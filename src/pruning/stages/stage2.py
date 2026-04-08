import json
import logging
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Any, Dict, List

from data.CountDataset import CountDataset
from data.CustomCollator import CustomCollator
from pruning.core.hooks import GPT2FullPathHooks
from pruning.core.OptimalAblationVectors import OptimalAblationVectors
from pruning.core.mask_samplers import FullPathsMaskSampler
from pruning.stages.base import PruningStage
from pruning.utilities.pruning_utils import int_key_hook
from pruning.utilities.metrics_logger import MetricsLogger

class CausalPruningStage2(PruningStage):
    
    def __init__(
        self,
        config,
        stage_name: str,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ):
        super().__init__(config, stage_name, logger, metrics_logger)
        
        # ------------------ Stage-specific setup -------------------        
        
        # acc_match = output_dict["acc_match"]
        # acc_task = output_dict["acc_task"]
        # kl_div = output_dict["kl_div"]
        # task_loss = output_dict["task_loss"]
        
        # Load config depending on whether we split MLPs or not
        self.model_config = self._convert_config_fullpaths_(self.model_config, self.num_heads_per_layer, split_mlp=self.stage_config.split_mlp)
        self.logger.info(f"After conversion to full paths: {self.model_config}")
        
        self.mask_sampler = FullPathsMaskSampler(self.model_config, split_mlp=self.stage_config.split_mlp).to(self.config.torch_device)
        if self.config.init_sample_param:
            self.mask_sampler.sample_params.data *= self.config.init_sample_param
        
        self.hooked_model = GPT2FullPathHooks(
            model=self.model,
            config=self.model_config,
            mapping_to_param_idx=self.mask_sampler.mapping_to_param_idx,
            split_mlp=self.stage_config.split_mlp,
            logger=self.logger
        )
        self.logger.info(f"Mask Param Names: {[n for n, p in self.mask_sampler.named_parameters()]}")
        self.logger.info(f"Total Edge Count: {sum(p.numel() for p in self.mask_sampler.parameters())}")
        
        # Recover lambdas from LN linearization
        converted_ln_var = torch.zeros(len(self.mask_sampler.all_output_vertex))
        for i, output_v in enumerate(self.mask_sampler.all_output_vertex):
            match output_v:
                case (layer, v, head, inp_v):
                    converted_ln_var[i] = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, v, head)]].item()
                case (layer, "mlp", inp_v):
                    converted_ln_var[i] = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, "mlp")]].item()
                case _:
                    converted_ln_var[i] = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[output_v]].item()
        
        # Initialize OA Vectors
        self.oa_vecs = OptimalAblationVectors(
            input_vertex=self.mask_sampler.input_vertex,
            output_vertex=self.mask_sampler.output_vertex,
            ln_vertex=self.mask_sampler.all_output_vertex,
            mlp_vertex=self.mask_sampler.mlp_output_vertex,
            model_config=self.model.config,
            init_var=converted_ln_var
        ).to(self.config.torch_device)
        
        # Initialize lamb and num_steps
        self.lamb = self.stage_config.lamb
        self.num_steps = self.stage_config.num_steps

    def _convert_config_fullpaths_(
        self,
        model_config: Dict[Any, Any],
        num_heads_per_layer: List[int],
        split_mlp: bool = True
    ):
        """
        Utility transformation to convert model config dictionary into full-path config
        as required by stage 2 pruning.

        Args:
            model_config (Dict[Any, Any]): original model configuration dictionary
            num_heads_per_layer (List[int]): number of attention heads in each layer
            split_mlp (bool, optional): whether or not to linearly decompose MLPs. Defaults to True.
        """
        
        num_layers = len(num_heads_per_layer)
        
        for layer in range(num_layers):
            for head in range(num_heads_per_layer[layer]):
                for attn_act in ["k", "q", "v"]:
                    
                    # If splitting MLPs, create distinct value-input paths
                    mlp_dependencies = (
                        [f"mlp-{l}"
                            for l in range(layer)
                        if f"mlp-{l}" in model_config[layer][attn_act][head]] if not split_mlp else
                        
                        [f"mlp-{l}-{prev_path}"
                            for l in range(layer)
                            for prev_path in model_config[l]["mlp"]
                        if f"mlp-{l}" in model_config[layer][attn_act][head]]
                    )
                    
                    # Update attention head dependencies
                    model_config[layer][attn_act][head] = [
                        act for act in ["wte", "wpe"] if act in model_config[layer][attn_act][head]
                    ] + \
                    [f"attn_output-{l}-{h}-{prev_path}"
                        for l in range(layer)
                        for h in range(num_heads_per_layer[l])
                        for prev_path in model_config[l]["v"][h]
                    if f"attn_output-{l}-{h}" in model_config[layer][attn_act][head]] + \
                    mlp_dependencies
            
            # Update MLP dependencies
            mlp_dependencies = (
                [f"mlp-{l}"
                    for l in range(layer)
                if f"mlp-{l}" in model_config[layer]["mlp"]] if not split_mlp else
                
                [f"mlp-{l}-{prev_path}"
                    for l in range(layer)
                    for prev_path in model_config[l]["mlp"]
                if f"mlp-{l}" in model_config[layer]["mlp"]]
            )
            model_config[layer]["mlp"] = [
                act for act in ["wte", "wpe"] if act in model_config[layer]["mlp"]
            ] + \
            [f"attn_output-{l}-{h}-{prev_path}"
                for l in range(layer + 1)
                for h in range(num_heads_per_layer[l])
                for prev_path in model_config[l]["v"][h]
            if f"attn_output-{l}-{h}" in model_config[layer]["mlp"]] + \
            mlp_dependencies
        
        # Update unembedding layer dependencies
        mlp_dependencies = (
            [f"mlp-{l}"
                for l in range(num_layers)
            if f"mlp-{l}" in model_config["lm_head"]] if not split_mlp else
            
            [f"mlp-{l}-{prev_path}"
                for l in range(num_layers)
                for prev_path in model_config[l]["mlp"]
            if f"mlp-{l}" in model_config["lm_head"]]
        )
        model_config["lm_head"] = [
            act for act in ["wte", "wpe"] if act in model_config["lm_head"]
        ] + \
        [f"attn_output-{l}-{h}-{prev_path}"
            for l in range(num_layers)
            for h in range(num_heads_per_layer[l])
            for prev_path in model_config[l]['v'][h]
        if f"attn_output-{l}-{h}" in model_config["lm_head"]] + \
        mlp_dependencies
        
        # Write the resulting model_config to output directory
        output_file = self.config.full_output_dir / f'stage{self.stage_idx + 1}' / 'transformed.json'
        with open(output_file, 'w') as f:
            json.dump(model_config, f, indent=4)
        
        return model_config

    def _pretrain_oa_vecs(self):
        self.logger.info("PRETRAINING OPTIMAL ABLATION VECTORS FOR OUTPUT NODE")
        num_pretrain_steps = 500
        log_interval = 50
        batch_size = 64
        mini_batch_size = 16  #TODO: maybe we want to add a config key for this
        accumulation_steps = batch_size // mini_batch_size if mini_batch_size > 0 else 0
        
        # Disable gradients, define optimizers and dataloader
        self.oa_vecs.input_vertex_oa.requires_grad_(False)
        self.oa_vecs.ln_var.requires_grad_(False)
        param_groups = [
            {"params": [self.oa_vecs.output_vertex_oa], "lr": self.config.lr_oa_for_pruning}
        ]
        
        if self.stage_config.split_mlp:
            param_groups.append({"params": self.oa_vecs.mlps.parameters(), "lr": self.config.lr_mlp_for_pruning})
        
        oa_optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collator)
        training_logs = defaultdict(list)

        # Pretraining loop
        for current_step, batch in enumerate(dataloader):
            print(f"STEP: {current_step} | mem: {(torch.mps.current_allocated_memory() / 1e9):.2f}GB / {(torch.mps.driver_allocated_memory() / 1e9):.2f}GB")
            batch = {k: v.to(self.config.torch_device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            oa_optimizer.zero_grad()
            
            # Gradient Accumulation
            if mini_batch_size != 0:
                
                for i in range(0, batch_size, mini_batch_size):
                    # Slice the current chunk
                    mini_batch = {k: v[i : i + mini_batch_size] for k, v in batch.items()}
                    mini_labels = labels[i : i + mini_batch_size]
                    
                    # Get target logits
                    with torch.no_grad():
                        target_logits = self.original_model(**mini_batch).logits
                    
                    masks = self.mask_sampler.sample_binary_masks(mini_batch_size)
                    logits = self.hooked_model(
                        masks=masks,
                        oa_vecs=self.oa_vecs,
                        **mini_batch
                    ).logits
                    
                    # Compute task loss and KL divergence loss
                    task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), mini_labels[:, 1:].flatten()).item()
                    target_logits = target_logits[:, :-1][mini_labels[:, 1:] != -100]
                    logits = logits[:, :-1][mini_labels[:, 1:] != -100]
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1),
                        log_target=True
                    )
                    
                    # Log, compute gradients and take step
                    training_logs["kl_div"].append(loss.item())
                    training_logs["task_loss"].append(task_loss)
                    
                    loss /= accumulation_steps
                    loss.backward()
                
            else:
                
                # Get target logits, masks and current logits
                with torch.no_grad():
                    target_logits = self.original_model(**batch).logits

                masks = self.mask_sampler.sample_binary_masks(batch_size)
                logits = self.hooked_model(
                    masks=masks,
                    oa_vecs=self.oa_vecs,
                    **batch
                ).logits
                
                # Compute KL divergence loss
                task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()
                target_logits = target_logits[:, :-1][labels[:, 1:] != -100]
                logits = logits[:, :-1][labels[:, 1:] != -100]
                loss = F.kl_div(
                    F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1),
                    log_target=True
                )
                
                # Log, compute gradients and take step
                training_logs["kl_div"].append(loss.item())
                training_logs["task_loss"].append(task_loss)
                
                loss.backward()
            
            # 1. Clear the previous activations to free up VRAM
            self.hooked_model.activations.clear() 

            if self.config.device == 'mps':
                # 2. Release the actual GPU memory (Releases Metal handles)
                # This tells the MPS driver to reclaim memory from deleted tensors.
                torch.mps.empty_cache()

                # 3. Synchronize (Optional but recommended for profiling)
                # This ensures the GPU has actually finished the "empty_cache" 
                # command before you check the memory stats for the next loop.
                torch.mps.synchronize()
            
            oa_optimizer.step()
            
            self.metrics_logger.log(
                stage=f"Stage {self.stage_idx + 1} (Pretrain)",
                timestamp=time.time(),
                step=current_step,
                current_maxstep=num_pretrain_steps,
                split="train",
                kl_div=loss.item(),
                task_loss=task_loss,
                loss=loss.item()
            )

            if (current_step + 1) % log_interval == 0:
                self.logger.info({k: sum(v) / len(v) for k, v in training_logs.items()})
                training_logs = defaultdict(list)
            
            if (current_step + 1) == num_pretrain_steps:
                break
        
        # Reset gradients
        self.oa_vecs.input_vertex_oa.requires_grad_(True)
        self.oa_vecs.ln_var.requires_grad_(True)
        for p in self.oa_vecs.parameters():
            p.grad = None

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
                        path + (key, )
                    )
                return config
            
            elif isinstance(config, list):
                new_list = []
                for item in config:
                    idx = mapping_to_param_idx[path + (item, )]
                    if masks[idx].item() == 1:
                        new_list.append(item)
                return new_list

            else:
                return config
        
        walk(model_config, masks, mapping_to_param_idx)
        
        # Save new config
        self.model_config = model_config
        self.output_dict['result_patching_config_global_iteration_1'] = deepcopy(self.model_config)

    def run(self):
        """
        Convenience method to execute stage 2 pruning. This includes...
        
        0. Pretraining of Optimal Ablation Output Vectors
        1. Training
        2. Validation
        3. Graph Transformation
        4. Intermediate Model Saving
        """
        
        # Pretrain the output optimal ablation vectors
        self.logger.info(f"0. Pruning Stage {self.stage_idx + 1} pre-training...")
        self._pretrain_oa_vecs()
        
        param_groups = [
            {"params": [self.oa_vecs.ln_var], "lr": self.config.lr_ln_var_for_pruning},
            {"params": [self.oa_vecs.input_vertex_oa], "lr": self.config.lr_oa_for_pruning},
            {"params": [self.oa_vecs.output_vertex_oa], "lr": 1e-4},
        ]
        
        if self.stage_config.split_mlp:
            param_groups.append({"params": self.oa_vecs.mlps.parameters(), "lr": self.config.lr_mlp_for_pruning})

        self.logger.info(f"1. Pruning Stage {self.stage_idx + 1} training...")
        self.train(
            oa_param_groups=param_groups,
            loss_type='algo'
        )
        
        self.logger.info(f"2. Pruning Stage {self.stage_idx + 1} validation...")
        self.test()
        
        self.logger.info(f"3. Pruning Stage {self.stage_idx + 1} graph transformation...")
        self.transform_graph()
        
        self.logger.info(f"4. Pruning Stage {self.stage_idx + 1} saving...")
        self.save()
        
        self.logger.info(f"Pruning Stage {self.stage_idx + 1} complete!\n")