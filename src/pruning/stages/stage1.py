import json
import logging
import torch
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy
from functools import partial
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from typing import Any, Dict, List, Union

from data.CountDataset import CountDataset
from data.CustomCollator import CustomCollator
from pruning.core.hooks import GPT2ComponentHooks
from pruning.core.OptimalAblationVectors import OptimalAblationVectors
from pruning.core.mask_samplers import ComponentMaskSampler
from pruning.stages.base import PruningStage
from pruning.utilities.metrics_logger import MetricsLogger

class CausalPruningStage1(PruningStage):
    def __init__(
        self,
        config,
        stage_name: str,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ):
        super().__init__(config, stage_name, logger, metrics_logger)
        
        # -------------- Stage-specific setup --------------
        
        # If first stage, convert model config to dependency graph
        self.model_config = self._get_full_possible_config_for_pruning(self.num_heads_per_layer)
        self.mask_sampler = ComponentMaskSampler(self.model_config).to(self.config.torch_device)
        if self.config.init_sample_param:
            self.mask_sampler.sample_params.data *= self.config.init_sample_param
            
        self.linear_LN = self.stage_config.linear_ln
        self.hooked_model = GPT2ComponentHooks(
            self.model,
            self.model_config,
            self.mask_sampler.mapping_to_param_idx,
            linearLN=self.linear_LN
        )
        self.logger.info(f"Mask Param Names: {[n for n, p in self.mask_sampler.named_parameters()]}")
        self.logger.info(f"Total Edge Count: {sum(p.numel() for p in self.mask_sampler.parameters())}")
        
        # Compute the expected variance estimates of LayerNorm inputs
        converted_ln_var = torch.zeros(len(self.mask_sampler.output_vertex))
        if self.linear_LN:
            # Compute expected variance estimates
            ln_var = self._estimate_ln_var(
                self.train_dataset,
                self.original_model,
                self.collator,
                self.config.torch_device,
                [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
            )
            
            # For each output nodes, map transformer graph vertices to the appropriate
            # LN variance estimate
            for i, output_v in enumerate(self.mask_sampler.output_vertex):
                match output_v:
                    # LN1
                    case (layer, act, head):
                        converted_ln_var[i] = ln_var[layer * 2]
                    # LN2
                    case (layer, "mlp"):
                        converted_ln_var[i] = ln_var[layer * 2 + 1]
                    # LNF
                    case ("lm_head",):
                        converted_ln_var[i] = ln_var[-1]
            
            # Log the variance tensor
            converted_ln_var = converted_ln_var.log()
        else:
            self.logger.info("USING REAL LAYERNORM")

        # Initialize the Optimal Ablation Vectors
        self.oa_vecs = OptimalAblationVectors(
            input_vertex=self.mask_sampler.input_vertex,
            output_vertex=self.mask_sampler.output_vertex,
            ln_vertex=self.mask_sampler.output_vertex,
            mlp_vertex=None,
            model_config=self.model.config,
            init_var=converted_ln_var
        ).to(self.config.torch_device)
        self._accumulate_biases_oa()
        
        # Initialize lamb and num_steps
        self.lamb = self.stage_config.lamb
        self.num_steps = self.stage_config.num_steps

    
    def _get_full_possible_config_for_pruning(
        self,
        num_heads_per_layer: List[int]
    ) -> Dict[str, List[Any]]:
        """
        This function builds a dependency map for pruning, as defined by the authors in Appendix F. In other words,
        we map which activations each module depends on.

        Args:
            num_heads_per_layer (List[int]): number of heads for each transformer layer in the GPT2 model

        Returns:
            Dict[str, List[Any]]: dependency map for each of the model layers
        """
        
        # Add transformer layer dependencies (for keys, queries, values and MLP)
        # Example: keys depend on WTE, WPE, all previous attention head outputs and all previous MLP outputs
        full_config = {
            layer: {
                "k": {
                    head: ["wte", "wpe"] + 
                        [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                        [f"mlp-{l}" for l in range(layer)]
                    for head in range(num_heads_per_layer[layer])
                },
                "q": {
                    head: ["wte", "wpe"] + 
                        [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                        [f"mlp-{l}" for l in range(layer)]
                    for head in range(num_heads_per_layer[layer])
                },
                "v": {
                    head: ["wte", "wpe"] + 
                        [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                        [f"mlp-{l}" for l in range(layer)]
                    for head in range(num_heads_per_layer[layer])
                },
                "mlp": ["wte", "wpe"] + 
                    [f"attn_output-{l}-{h}" for l in range(layer + 1) for h in range(num_heads_per_layer[l])] + 
                    [f"mlp-{l}" for l in range(layer)] 
            }
            for layer in range(len(num_heads_per_layer))
        }
        
        # Add the LM head dependencies
        full_config.update({
            "lm_head": ["wte", "wpe"] + 
                [f"attn_output-{l}-{h}" for l in range(len(num_heads_per_layer)) for h in range(num_heads_per_layer[l])] + 
                [f"mlp-{l}" for l in range(len(num_heads_per_layer))]
        })
        
        return full_config
        
    @torch.no_grad()
    def _estimate_ln_var(
        self,
        dataset: Union[CountDataset],
        model: GPT2LMHeadModel,
        collator: CustomCollator,
        device: torch.device,
        ignore_ids: List[int]
    ):
        """
        Estimate average input variance of every LayerNorm in the model.
        We do this by running 10 forward passes with the original model,
        each with batch size 64. The final variance of each input tensor
        is computed as the average of these runs

        Args:
            dataset (CountDataset | TODO): the dataset used for the task
            model (GPT2LMHeadModel): the original unpruned model
            collator (CustomCollator): the collator used to batch the task data
            device (torch.device): PyTorch device used for training
            ignore_ids (List[int]): token ids that should be ignored

        Returns:
            torch.Tensor: a tensor with the target variance for each of the input tensors to the model's LayerNorms
        """
        num_layers = len(model.transformer.h)
        batch_size = 64
        hooks = []
        var = torch.zeros(num_layers * 2 + 1, device=device)
        
        def save_hook(module, input, output, idx):
            """Accumulates the variance for this batch along the hidden dim. We mask out PAD/BOS/EOS/SEP tokens"""
            var[idx] += input[0].var(dim=-1)[~mask].mean()
        
        hooks.append(model.transformer.ln_f.register_forward_hook(partial(
            save_hook, idx=-1
        )))
        
        for layer in range(num_layers):
            hooks.append(model.transformer.h[layer].ln_1.register_forward_hook(partial(
                save_hook, idx=layer*2
            )))
            
            hooks.append(model.transformer.h[layer].ln_2.register_forward_hook(partial(
                save_hook, idx=layer*2+1
            )))
        
        inputs = []
        step_idx = 0
        for item in dataset:
            inputs.append(item)
            if len(inputs) == batch_size:
                batch = collator(inputs)
                batch = {k : v.to(device) for k, v in batch.items()}
                mask = torch.stack([batch['input_ids'] == i for i in ignore_ids]).any(dim=0)
                batch.pop("labels")
                model(**batch)
                
                inputs = []
                step_idx += 1
                if step_idx == 10:
                    break
                
        for hook in hooks:
            hook.remove()
        
        var /= 10
        return var

    def _accumulate_biases_oa(self):
        """
        This function accumulates biases from previous attentions layers. This gives a baseline
        residual value entering each node and makes pruning less destructive.
        Without this, pruning might incorrectly keep some edges to recreate the missing
        bias offset
        """
        
        for layer in range(self.num_layers):
            
            # Attention Layers Previous Biases
            for head in range(self.num_heads_per_layer[layer]):
                for act in ["q", "k", "v"]:
                    for i in range(layer):
                        oa_vec_idx = self.oa_vecs.to_out_oa_idx[(layer, act, head)]
                        local_bias = self.model.transformer.h[i].attn.resid_dropout(self.model.transformer.h[i].attn.c_proj.bias)
                        self.oa_vecs.output_vertex_oa.data[oa_vec_idx] += local_bias
            
            # MLP Layers Previous Biases
            for i in range(layer + 1):
                oa_vec_idx = self.oa_vecs.to_out_oa_idx[(layer, "mlp")]
                local_bias = self.model.transformer.h[i].attn.resid_dropout(self.model.transformer.h[i].attn.c_proj.bias)
                self.oa_vecs.output_vertex_oa.data[oa_vec_idx] += local_bias
        
        # LM Head Previous Biases
        for i in range(self.num_layers):
            oa_vec_idx = self.oa_vecs.to_out_oa_idx[("lm_head",)]
            local_bias = self.model.transformer.h[i].attn.resid_dropout(self.model.transformer.h[i].attn.c_proj.bias)
            self.oa_vecs.output_vertex_oa.data[oa_vec_idx] += local_bias

    def run(self):
        """
        Convenience method to execute stage 1 pruning. This includes...
        
        1. Training
        2. Validation
        3. Graph Transformation
        4. Intermediate Model Saving
        """
        
        param_groups = [
            {"params": [self.oa_vecs.ln_var], "lr": self.config.lr_ln_var_for_pruning},
            {"params": [self.oa_vecs.input_vertex_oa], "lr": self.config.lr_oa_for_pruning},
            {"params": [self.oa_vecs.output_vertex_oa], "lr": 1e-4},
        ]
        
        self.logger.info(f"1. Pruning Stage {self.stage_idx + 1} training...")
        self.train(oa_param_groups=param_groups, loss_type='algo')
        
        self.logger.info(f"2. Pruning Stage {self.stage_idx + 1} validation...")
        self.test()
        
        self.logger.info(f"3. Pruning Stage {self.stage_idx + 1} graph transformation...")
        self.transform_graph()
        
        self.logger.info(f"4. Pruning Stage {self.stage_idx + 1} saving...")
        self.save()
        
        self.logger.info(f"Pruning Stage {self.stage_idx + 1} complete!\n")

    def transform_graph(self):
        # Get final trained masks
        masks = self.mask_sampler.sample_binary_masks(1).squeeze(0)
        model_config = self.model_config
        mapping_to_param_idx = self.mask_sampler.mapping_to_param_idx
        
        assert masks.dim() == 1
        assert ((masks != 0) & (masks != 1)).sum().item() == 0
        
        def walk(config, masks, mapping_to_param_idx, path=()):
            if isinstance(config, dict):
                for key in config:
                    config[key] = walk(
                        config[key],
                        masks,
                        mapping_to_param_idx,
                        path + (key,)
                    )
                return config
            
            elif isinstance(config, list):
                new_list = []
                for item in config:
                    idx = mapping_to_param_idx[path + (item,)]
                    if masks[idx].item() == 1:
                        new_list.append(item)
                
                return new_list
            else:
                return config
        
        walk(model_config, masks, mapping_to_param_idx)
        
        # Save new config
        self.model_config = model_config
        self.output_dict['result_patching_config_global_iteration_0'] = deepcopy(self.model_config)