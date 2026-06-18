import json
import logging
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from copy import deepcopy
from functools import partial
from transformers import GPT2LMHeadModel
from typing import Any, Dict, List, Union

from data.CountDataset import CountDataset
from data.CustomCollator import CustomCollator
from pruning.core.hooks import GPT2QKHooks
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from pruning.core.mask_samplers import QKMaskSampler
from pruning.stages.base import PruningStage
from utilities.metrics_logger import MetricsLogger

class CausalPruningStage3(PruningStage):
    
    def __init__(
        self,
        config,
        stage_name: str,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ):
        super().__init__(config, stage_name, logger, metrics_logger)
        
        print(self.model_config)
        
        # -------------- Stage-specific setup --------------
        self.model_config = self._convert_config_kq_(self.model_config, self.num_heads_per_layer)
        self.logger.info(f"After per-head Cartesian-product explosion: {self.model_config}")
        
        self.mask_sampler = QKMaskSampler(self.model_config).to(self.config.torch_device)
        if self.config.init_sample_param:
            self.mask_sampler.sample_params.data *= self.config.init_sample_param
        
        self.hooked_model = GPT2QKHooks(
            model=self.model,
            config=self.model_config,
            mapping_to_param_idx=self.mask_sampler.mapping_to_param_idx,
            split_mlp=hasattr(self.oa_vecs, "mlps"),
            logger=self.logger
        )
        self.logger.info(f"Mask Param names: {[n for n, p in self.mask_sampler.named_parameters()]}")
        self.logger.info(f"Total Edge Count: {sum(p.numel() for p in self.mask_sampler.parameters())}")
        
        self.oa_vecs = OptimalQueryBiasVectors(
            self.mask_sampler.key_names,
            self.model.transformer.h[0].attn.head_dim,
            self.oa_vecs
        ).to(self.config.torch_device)
        
        # Initialize lamb and num_steps
        self.lamb = self.stage_config.lamb
        self.num_steps = self.stage_config.num_steps
        self.split_mlp = hasattr(self.oa_vecs, "mlps")
    
    def _convert_config_kq_(
        self,
        model_config: Dict[Any, Any],
        num_heads_per_layer: List[isinstance]
    ):
        """
        Utility transformation to convert model config dictionary from full-path version to QK pruning
        input with Cartesian products of all interacting variables in each head

        Args:
            model_config (Dict[Any, Any]): full-paths model configuration from stage 2
            num_heads_per_layer (List[int]): number of attention heads in each layer
        """
        
        for layer in range(len(num_heads_per_layer)):
            model_config[layer]["qk"] = {}
            for head in range(num_heads_per_layer[layer]):
                
                # Cartesian Explosion
                model_config[layer]["qk"][head] = [
                    (path_in_q, path_in_k)
                    for path_in_q in model_config[layer]["q"][head]
                    for path_in_k in model_config[layer]["k"][head]
                ]
            
            del model_config[layer]["q"]
        
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
        mini_batch_size = 16
        accumulation_steps = batch_size // mini_batch_size if mini_batch_size > 0 else 0
        
        # Disable gradients, define optimizers and dataloader
        self.oa_vecs.ln_var.requires_grad_(False)
        if hasattr(self.oa_vecs, "mlps"):
            self.oa_vecs.mlps.requires_grad_(False)
    
        oa_optimizer = torch.optim.AdamW([self.oa_vecs.output_vertex_oa, self.oa_vecs.q_bias_term], lr=self.config.lr_oa_for_pruning, weight_decay=0)
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
                    
                    with torch.no_grad():
                        target_logits = self.original_model(**mini_batch).logits

                    masks = self.mask_sampler.sample_binary_masks(mini_batch_size)
                    logits = self.hooked_model(
                        masks=masks,
                        oa_vecs=self.oa_vecs,
                        **mini_batch
                    ).logits

                    result = self.loss_module.compute_batch(
                        logits, target_logits, mini_labels, mini_batch["input_ids"]
                    )
                    task_loss = result.task_loss
                    loss = result.distillation_loss
                    
                    # Log, compute gradients and take step
                    training_logs["kl_div"].append(loss.item())
                    training_logs["task_loss"].append(task_loss)
                    
                    loss /= accumulation_steps
                    loss.backward()
                
            else:
                
                with torch.no_grad():
                    target_logits = self.original_model(**batch).logits

                masks = self.mask_sampler.sample_binary_masks(batch_size)
                logits = self.hooked_model(
                    masks=masks,
                    oa_vecs=self.oa_vecs,
                    **batch
                ).logits

                result = self.loss_module.compute_batch(
                    logits, target_logits, labels, batch["input_ids"]
                )
                task_loss = result.task_loss
                loss = result.distillation_loss
                
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
        self.oa_vecs.ln_var.requires_grad_(True)
        if hasattr(self.oa_vecs, "mlps"):
            self.oa_vecs.mlps.requires_grad_(True)
        
        for p in self.oa_vecs.parameters():
            p.grad = None
    
    def transform_graph(self):
        # Get final trained masks
        masks = self.mask_sampler.sample_binary_masks(1).squeeze(0)
        num_edges = (masks == 1).sum().item()
        mapping_to_param_idx = self.mask_sampler.mapping_to_param_idx
        self.logger.info(f"After Pruning Product Count: {num_edges}")
        
        # convert_mask_to_config_qk_
        assert masks.dim() == 1
        assert ((masks != 0) & (masks != 1)).sum().item() == 0

        # For each qk and k vertex, keep paths that haven't been dropped
        for k1 in self.model_config:
            if type(self.model_config[k1]) == dict:
                for k2 in self.model_config[k1]:
                    
                    if k2 == "qk" or k2 == "k":
                        for k3 in self.model_config[k1][k2]:
                            new_lis = []
                            for item in self.model_config[k1][k2][k3]:
                                
                                # If the mask is active, keep this item
                                if masks[mapping_to_param_idx[(k1, k3, item)]].item() == 1:
                                    new_lis.append(item)
                            
                            self.model_config[k1][k2][k3] = new_lis
        
        # remove_other_edges_after_qk_pruning_
        # 1. Determine nodes needed
        nodes_needed = set()
        def find_needed(node):
            nonlocal nodes_needed
            
            if type(node) == dict:
                for key in node:
                    find_needed(node[key])
            
            elif type(node) == list:
                for item in node:
                    if type(item) != str:
                        nodes_needed.add(item[0])
                        nodes_needed.add(item[1])
                    else:
                        nodes_needed.add(item)
            else:
                raise RuntimeError("Invalid configuration format!")
            
        find_needed(self.model_config)
        
        # 2. Delete unneeded nodes
        for k1 in self.model_config:
            if type(self.model_config[k1]) == dict:  # k1=layer
                for k2 in self.model_config[k1]:
                    
                    if k2 == "v":
                        for k3 in self.model_config[k1][k2]:  # k3=head
                            assert type(self.model_config[k1][k2][k3]) == list
                            new_lis = []
                            for item in self.model_config[k1][k2][k3]:
                                node = f"attn_output-{k1}-{k3}-{item}"
                                if node in nodes_needed:
                                    new_lis.append(item)
                                else:
                                    self.logger.info(f"{node} is removed")
                            
                            self.model_config[k1][k2][k3] = new_lis
                    
                    elif k2 == "mlp":
                        if self.split_mlp:
                            new_lis = []
                            for item in self.model_config[k1][k2]:
                                node = f"mlp-{k1}-{item}"
                                if node in nodes_needed:
                                    new_lis.append(item)
                                else:
                                    self.logger.info(f"{node} is removed")
                        else:
                            node = f"mlp-{k1}"
                            if node not in nodes_needed and len(self.model_config[k1][k2]) > 0:
                                self.model_config[k1][k2] = []
                                self.logger.info(f"{node} is removed")
        
        # Save new config
        self.output_dict['result_patching_config_global_iteration_2'] = deepcopy(self.model_config)

    def run(self):
        """
        Convenience method to execute stage 3 pruning. This includes...
        
        1. Pre-training
        2. Training
        3. Validation
        4. Graph Transformation
        5. Intermediate Model Saving
        """
        
        self.logger.info(f"0. Pruning Stage {self.stage_idx + 1} pre-training...")
        self._pretrain_oa_vecs()
        
        param_groups = [
            {"params": [self.oa_vecs.ln_var], "lr": self.config.lr_ln_var_for_pruning},
            {"params": [self.oa_vecs.q_bias_term], "lr": self.config.lr_oa_for_pruning},
            {"params": [self.oa_vecs.output_vertex_oa], "lr": 1e-4},
        ]
        
        if hasattr(self.oa_vecs, "mlps"):
            param_groups.append({"params": self.oa_vecs.mlps.parameters(), "lr": self.config.lr_mlp_for_pruning})
        
        self.logger.info(f"1. Pruning Stage {self.stage_idx + 1} training...")
        self.train(oa_param_groups=param_groups)
        
        self.logger.info(f"2. Pruning Stage {self.stage_idx + 1} validation...")
        self.test()
        
        self.logger.info(f"3. Pruning Stafe {self.stage_idx + 1} graph transformation...")
        self.transform_graph()
        
        self.logger.info(f"4. Pruning Stage {self.stage_idx + 1} saving...")
        self.save()
        
        self.logger.info(f"Pruning Stage {self.stage_idx + 1} complete!\n")