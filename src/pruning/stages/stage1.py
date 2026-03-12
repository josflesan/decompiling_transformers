import json
import logging
import torch
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy
from functools import partial
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from typing import List, Union

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
        stage_idx: int,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ):
        super().__init__(config, stage_idx, logger, metrics_logger)
        
        # -------------- Stage-specific setup --------------
        self.mask_sampler = ComponentMaskSampler(self.model_config)
        if self.config.init_sample_param:
            self.mask_sampler.sample_params.data *= self.config.init_sample_param
        
        self.linear_LN = self.config.pruning_stages[self.stage_idx].linear_ln
        self.hooked_model = GPT2ComponentHooks(
            self.model,
            self.model_config,
            self.mask_sampler.mapping_to_param_idx,
            linearLN=self.linear_LN
        )
        
        # Compute the expected variance estimates of LayerNorm inputs
        self.converted_ln_var = torch.zeros(len(self.mask_sampler.output_vertex))
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
                        self.converted_ln_var[i] = ln_var[layer * 2]
                    # LN2
                    case (layer, mlp):
                        self.converted_ln_var[i] = ln_var[layer * 2 + 1]
                    # LNF
                    case (lm_head,):
                        self.converted_ln_var[i] = ln_var[-1]
                
            # Log the variance tensor
            self.converted_ln_var = self.converted_ln_var.log()
        else:
            self.logger.info("USING REAL LAYERNORM")

        # Initialize the Optimal Ablation Vectors
        self.oa_vecs = OptimalAblationVectors(
            input_vertex=self.mask_sampler.input_vertex,
            output_vertex=self.mask_sampler.output_vertex,
            ln_vertex=self.mask_sampler.output_vertex,
            mlp_vertex=None,
            model_config=self.model.config,
            init_var=self.converted_ln_var
        ).to(self.config.torch_device)
        self._accumulate_biases_oa()
        
        # Initialize lamb and num_steps
        self.lamb = self.config.pruning_stages[self.stage_idx].lamb
        self.num_steps = self.config.pruning_stages[self.stage_idx].num_steps
        
        #TODO: figure out how to implement this check in refactor    
        # if mask_sampler.sample_params.numel() == 0:
        #     hooked_model.remove_hooks()
        #     # output_dict["result_patching_performance_global_iteration_0"] = {
        #     #     "acc_match": acc_match,
        #     #     "acc_task": acc_task,
        #     #     "task_loss": task_loss,
        #     #     "kl_div": kl_div,
        #     #     "num_edges": 0,
        #     #     "coef": lamb
        #     # }
        #     output_dict['result_patching_config_global_iteration_0'] = deepcopy(model_config)

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
        
    def train(self):
        gamma = 0.0
        log_interval = 50
        batch_size = self.config.pruning_stages[self.stage_idx].batch_size
        num_repeat = self.config.pruning_stages[self.stage_idx].num_repeat
        unique_input_per_batch = batch_size // num_repeat
        countdown = self.num_steps
        patience = 3
        training_logs = defaultdict(list)
        dataloader = DataLoader(self.train_dataset, batch_size=unique_input_per_batch, shuffle=False, collate_fn=self.collator)
        
        assert unique_input_per_batch * num_repeat == batch_size
        
        sampling_opt = torch.optim.AdamW(
            self.mask_sampler.parameters(),
            lr=self.config.lr_sampler_for_pruning,
            weight_decay=0,
            betas=(0.9, 0.995)
        )
        oa_optimizer = torch.optim.AdamW(
            [
                {"params": [self.oa_vecs.ln_var], "lr": self.config.lr_ln_var_for_pruning},
                {"params": [self.oa_vecs.input_vertex_oa], "lr": self.config.lr_oa_for_pruning},
                {"params": [self.oa_vecs.output_vertex_oa], "lr": 1e-4},
            ],
            weight_decay=0
        )
        
        for current_step, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.config.torch_device).repeat(num_repeat, *([1] * (v.dim() - 1))) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            with torch.no_grad():
                target_logits = self.original_model(**batch).logits
            
            masks = self.mask_sampler.sample_masks(batch_size)
            logits = self.hooked_model(masks=masks, oa_vecs=self.oa_vecs, **batch).logits
            
            # Compute task loss and pruning loss
            task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()
            target_logits = target_logits[:, :-1][labels[:, 1:] != -100]
            logits = logits[:, :-1][labels[:, 1:] != -100]
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1),
                log_target=True,
                
                #TODO: is this correct?
                reduction='batchmean'
            )
            
            training_logs['kl_div'].append(loss.item())
            training_logs['task_loss'].append(task_loss)
            penalty, (reg_edge, reg_node) = self.mask_sampler.get_penalty(gamma)
            training_logs['reg_edge'].append(reg_edge)
            training_logs['reg_node'].append(reg_node)        
            loss = loss + self.lamb * penalty
            
            sampling_opt.zero_grad()
            oa_optimizer.zero_grad()
            loss.backward()
            
            # Clip gradient norms and take step
            oa_grad_norm = torch.nn.utils.clip_grad_norm_(self.oa_vecs.parameters(), max_norm=float('inf')).item()
            sampler_grad_norm = torch.nn.utils.clip_grad_norm_(self.mask_sampler.parameters(), max_norm=float('inf')).item()
            training_logs["oa_grad_norm"].append(oa_grad_norm)
            training_logs["sampler_grad_norm"].append(sampler_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.mask_sampler.parameters(), 5)
            sampling_opt.step()
            oa_optimizer.step()
            
            # Log warning if any mask parameters are NaN
            nan_count = sum(p.isnan().sum().item() for p in self.mask_sampler.parameters())
            if nan_count > 0:
                self.logger.warning("sum NaN ", nan_count)
            
            self.metrics_logger.log(
                stage=self.stage_idx,
                step=current_step,
                kl_div=loss.item(),
                task_loss=task_loss,
                reg_edge=reg_edge,
                reg_node=reg_node,
                loss=loss.item(),
                oa_grad_norm=oa_grad_norm,
                sampler_grad_norm=sampler_grad_norm
            )
            
            # Logging
            if (current_step + 1) % log_interval == 0:
                self.logger({k: sum(v) / len(v) for k, v in training_logs.items()})
                all_sample_p = torch.cat([p.data.detach().view(-1) for p in self.mask_sampler.parameters()], dim=0)
                hist, bin_edges = torch.histogram(all_sample_p.cpu(), bins=5)
                self.logger("Histogram of Sampling Params", "\nhist", hist, "\nbin edges", bin_edges)
                
                #TODO: log histogram of sampling parameters
                
                if all_sample_p.max().item() < -2:
                    self.logger.error("All pruned, training failed. Stopping early...")
                    break
                
                if self.config.baseline_loss and (sum(training_logs['kl_div']) / len(training_logs['kl_div'])) > run_config.baseline_loss:
                    patience -= 1
                    if patience == 0:
                        self.logger.error("Loss stuck at high value, training failed. Stopping early...")
                        break
                else:
                    patience = 3
                
                if ((all_sample_p > -1) & (all_sample_p < 1)).sum().item():
                    count_down = self.num_steps + 1
            
            # Early Stopping
            countdown -= 1
            if countdown == 0:
                break
            if (current_step + 1) == 5000:
                break
            
            break #TODO: remove this
    
    def val(self):
        num_test_step = 200
        num_correct = 0
        num_match = 0
        task_loss = 0
        kl_div = 0
        batch_size = self.config.pruning_stages[self.stage_idx].batch_size
        dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collator)
        loss_func = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for current_step, batch in enumerate(dataloader):
                # Move batch tensors to device
                batch = {k: v.to(self.config.torch_device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                target_logits = self.original_model(**batch).logits
                
                masks = self.mask_sampler.sample_binary_masks(batch_size)
                logits = self.hooked_model(masks=masks, oa_vecs=self.oa_vecs, **batch).logits
                
                shift_target_logits = target_logits[:, :-1]
                shift_logits = logits[:, :-1]
                shift_labels = labels[:, 1:]
                target_predictions = shift_target_logits.argmax(dim=-1)
                predictions = shift_logits.argmax(dim=-1)
                
                match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                num_match += match.sum().item()
                
                correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                num_correct += correct.sum().item()
                
                task_loss += loss_func(shift_logits.flatten(end_dim=1), shift_labels.flatten()).item()
                kl_div += F.kl_div(
                    F.log_softmax(shift_logits[shift_labels != -100], dim=-1), F.log_softmax(shift_target_logits[shift_labels != -100], dim=-1),
                    log_target=True,
                    
                    #TODO: is this correct?
                    reduction='batchmean'
                ).item()
                
                if current_step + 1 == num_test_step:
                    break
                
                break  #TODO: Remove this
            
        acc_match = num_match / (num_test_step * batch_size)
        acc_task = num_correct / (num_test_step * batch_size)
        task_loss /= num_test_step
        kl_div /= num_test_step
        
        masks = self.mask_sampler.sample_binary_masks(1).squeeze(0)
        num_edges = (masks == 1).sum().item()
        self.logger.info(f"After Pruning Edge Count: {num_edges}")
        
        # Prepare output dictionary
        self.output_dict[f"result_patching_performance_global_iteration_0"] = {
            "acc_match": acc_match,
            "acc_task": acc_task,
            "task_loss": task_loss,
            "kl_div": kl_div,
            "num_edges": num_edges,
            "coef": self.lamb
        }
        
        self.output_dict['result_patching_config_global_iteration_0'] = deepcopy(self.model_config)
        self.hooked_model.remove_hooks()
        
        # Save optimal ablation vectors and output training results
        self.output_dict['acc_match'] = acc_match
        self.output_dict['acc_task'] = acc_task
        self.output_dict['kl_div'] = kl_div
        self.output_dict['task_loss'] = task_loss

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
    
    def save(self):
        # Save optimal ablation vectors and output training JSON
        torch.save(self.oa_vecs, self.config.full_output_dir / 'oa_vecs.pt')
        output_file = self.config.full_output_dir / 'output.json'
        with open(output_file, "w") as f:
            json.dump(self.output_dict, f, indent=4)
        
        # Save the pruned model
        model_dir = self.config.full_output_dir / 'pruned_model_stage1'
        model_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_dir)