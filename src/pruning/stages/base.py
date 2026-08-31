import json
import logging
import time
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from typing import Any, Dict, List, Optional

from tasks.registry import get_task
from pruning.core.lambda_search import (
    LambdaTrialResult,
    bracket_exists,
    compute_threshold_acc,
    resolve_probe_num_steps,
    select_best_trial,
    suggest_probe_lambdas,
    suggest_refined_lambdas,
)
from pruning.utilities.pruning_dataclasses import LambSearchConfig, PruningRunConfig, StageConfig
from utilities.core import LossModule, int_key_hook
from utilities.metrics_logger import MetricsLogger

class PruningStage(ABC):
    """
    Base abstract class defining a pruning stage and containing methods for stage-specific
    training, validation and graph transformation. Each pruning stage will be an child instance of
    this abstract class.
    """
    
    def __init__(
        self,
        config: PruningRunConfig,
        stage_name: str,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ): 
        self.config = config
        self.stage_name = stage_name  # stageX
        self.stage_idx = int(stage_name[-1]) - 1
        self.stage_config: StageConfig = self.config.pruning_stages[stage_name]
        self.logger = logger
        self.metrics_logger = metrics_logger
        self._lambda_trial_label: Optional[str] = None
        
        # Initialize output dict and models
        self.output_config_path = config.full_output_dir / 'args.json'
        self.output_dict = {}
        self.model = GPT2LMHeadModel.from_pretrained(Path(config.model_path)).to(config.torch_device)
        self.original_model = GPT2LMHeadModel.from_pretrained(Path(config.model_path)).to(config.torch_device)
        self.model.eval()
        self.original_model.eval()
        
        # Initialize task tokenizers, datasets and collators
        self.task_config = config.task_config
        self.task = get_task(self.task_config.name, self.task_config)
        self.tokenizer, datasets = self.task.build()
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.loss_module = LossModule.from_dataset(self.train_dataset, self.tokenizer)
        self.collator = self.loss_module.collator()
        
        # Initialize model variables
        self.num_layers = len(self.model.transformer.h)
        self.num_heads_per_layer = {
            layer: self.model.transformer.h[layer].attn.num_heads for layer in range(self.num_layers)
        }
        
        # Load state from previous stage if it exists
        if self.stage_idx != 0:
            self.oa_vecs = torch.load(self.config.full_output_dir / f"stage{self.stage_idx}" / "oa_vecs.pt", map_location=self.config.torch_device, weights_only=False)
            with open(self.config.full_output_dir / f"stage{self.stage_idx}" / "output.json") as f:
                self.output_dict = json.load(f, object_hook=int_key_hook)
                self.model_config = self.output_dict[f"result_patching_config_global_iteration_{self.stage_idx - 1}"]
    
    def _get_mps_memory_map(self):
        """Returns current MPS memory usage in GB."""
        # current_allocated is the memory currently held by tensors
        return {
            "allocated": torch.mps.current_allocated_memory() / 1e9,
            "driver_allocated": torch.mps.driver_allocated_memory() / 1e9, # Total memory the driver is using
        }
    
    def _format_lambda_trial_label(self, lamb: float, *, probe: bool) -> str:
        kind = "probe" if probe else "final"
        return f"{kind} λ={lamb:.2e}"

    def _set_lambda_trial_label(self, lamb: float, *, probe: bool) -> None:
        self._lambda_trial_label = self._format_lambda_trial_label(lamb, probe=probe)

    def _clear_lambda_trial_label(self) -> None:
        self._lambda_trial_label = None

    def _current_num_edges(self) -> int:
        with torch.no_grad():
            masks = self.mask_sampler.sample_binary_masks(1).squeeze(0)
            return int((masks == 1).sum().item())

    def train(
        self,
        oa_param_groups: Dict[str, Any],
        num_steps_override: Optional[int] = None,
    ):
        # Set up optimizer, dataloader and other parameters
        gamma = 0.0
        log_interval = 50
        step_budget = num_steps_override if num_steps_override is not None else self.num_steps
        countdown = step_budget
        patience = 3
        batch_size = self.stage_config.train_batch_size
        mini_batch_size = self.stage_config.mini_batch_size
        accumulation_steps = batch_size // mini_batch_size if mini_batch_size > 0 else 0
        
        num_repeat = self.stage_config.num_repeat
        unique_input_per_batch = batch_size // num_repeat
        assert unique_input_per_batch * num_repeat == batch_size
        
        sampling_opt = torch.optim.AdamW(
            self.mask_sampler.parameters(),
            lr=self.config.lr_sampler_for_pruning,
            weight_decay=0, 
            betas=(0.9, 0.995)
        )
        oa_optimizer = torch.optim.AdamW(oa_param_groups, weight_decay=0)
        dataloader = DataLoader(self.train_dataset, batch_size=unique_input_per_batch, shuffle=False, collate_fn=self.collator)
        training_logs = defaultdict(list)
        
        for current_step, batch in enumerate(dataloader):
            print(f"STEP: {current_step} | mem: {(torch.mps.current_allocated_memory() / 1e9):.2f}GB / {(torch.mps.driver_allocated_memory() / 1e9):.2f}GB")
            
            #TODO: figure out how to save metrics upon failure in refactor
            if self.mask_sampler.sample_params.numel() == 0:
                self.hooked_model.remove_hooks()
                self.output_dict[f'result_patching_config_global_iteration_{self.stage_idx}'] = deepcopy(self.model_config)
                self.logger.info("Nothing to prune, exiting...")
                break
            
            # Move to device
            batch = {k: v.to(self.config.torch_device).repeat(num_repeat, *([1] * (v.dim() - 1))) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            sampling_opt.zero_grad()
            oa_optimizer.zero_grad()
            
            # Gradient Accumulation
            if mini_batch_size != 0:
                
                for i in range(0, batch_size, mini_batch_size):
                    # Slice the current chunk
                    mini_batch = {k: v[i : i + mini_batch_size] for k, v in batch.items()}
                    mini_labels = labels[i : i + mini_batch_size]
                    
                    # Compute task loss and pruning loss
                    with torch.no_grad():
                        target_logits = self.original_model(**mini_batch).logits

                    masks = self.mask_sampler.sample_masks(mini_batch_size)
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
                    
                    training_logs['kl_div'].append(loss.item())
                    training_logs['task_loss'].append(task_loss)
                    penalty, (reg_edge, reg_node) = self.mask_sampler.get_penalty(gamma)
                    training_logs['reg_node'].append(reg_node)
                    training_logs['reg_edge'].append(reg_edge)
                    
                    loss = (loss + self.lamb * penalty) / accumulation_steps
                    loss.backward()
                    
                    # 1. Clear the previous activations to free up VRAM
                    self.hooked_model.activations.clear() 
                    
            else:
                with torch.no_grad():
                    target_logits = self.original_model(**batch).logits

                masks = self.mask_sampler.sample_masks(batch_size)
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
                
                training_logs['kl_div'].append(loss.item())
                training_logs['task_loss'].append(task_loss)
                penalty, (reg_edge, reg_node) = self.mask_sampler.get_penalty(gamma)
                training_logs['reg_node'].append(reg_node)
                training_logs['reg_edge'].append(reg_edge)
                
                loss = loss + self.lamb * penalty
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
                    
            # Clip gradient norms and take step
            oa_grad_norm = torch.nn.utils.clip_grad_norm_(self.oa_vecs.parameters(), max_norm=float('inf')).item()
            sampler_grad_norm = torch.nn.utils.clip_grad_norm_(self.mask_sampler.parameters(), max_norm=float('inf')).item()
            training_logs['oa_grad_norm'].append(oa_grad_norm)
            training_logs['sampler_grad_norm'].append(sampler_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.mask_sampler.parameters(), 5)
            
            # Take optimizer steps
            sampling_opt.step()
            oa_optimizer.step()
            
            # Log warning if any mask parameters are NaN
            nan_count = sum(p.isnan().sum().item() for p in self.mask_sampler.parameters())
            if nan_count > 0:
                self.logger.warning(f"sum NaN: {nan_count}")
            
            all_sample_p = torch.cat([p.data.detach().view(-1) for p in self.mask_sampler.parameters()], dim=0)
            train_metrics = {
                "stage": f"Stage {self.stage_idx + 1}",
                "timestamp": time.time(),
                "step": current_step,
                "current_maxstep": min(current_step + countdown, 5000),
                "split": "train",
                "kl_div": loss.item(),
                "task_loss": task_loss,
                "reg_edge": reg_edge,
                "reg_node": reg_node,
                "loss": loss.item(),
                "oa_grad_norm": oa_grad_norm,
                "sampler_grad_norm": sampler_grad_norm,
                "sampler_params": all_sample_p.cpu().tolist(),
            }
            if self._lambda_trial_label is not None:
                train_metrics["lambda_trial"] = self._lambda_trial_label
                train_metrics["lamb"] = self.lamb
                train_metrics["num_edges"] = self._current_num_edges()
            self.metrics_logger.log(**train_metrics)
            
            # Logging
            if (current_step + 1) % log_interval == 0:
                mem_current = self._get_mps_memory_map() if self.config.device == 'mps' else {}
                self.logger.info(f"STEP {current_step} | Mem: {mem_current['allocated']:.2f}GB / Reserved: {mem_current['driver_allocated']:.2f}GB")
                self.logger.info({k: sum(v) / len(v) for k, v in training_logs.items()})
                
                if all_sample_p.max().item() < -2:
                    self.logger.error("All pruned, training failed. Stopping early...")
                    break
                
                if self.config.baseline_loss and (sum(training_logs['kl_div']) / len(training_logs['kl_div'])) > self.config.baseline_loss:
                    patience -= 1
                    if patience == 0:
                        self.logger.error("Loss stuck at high value, training failed. Stopping early...")
                        break
                else:
                    patience = 3
                
                # If there are ambivalent masks left, increase number of steps
                if ((all_sample_p > -1) & (all_sample_p < 1)).sum().item():
                    countdown = step_budget + 1
                
                print(f"COUNTDOWN STEPS LEFT: {countdown}")
                
                training_logs = defaultdict(list)
            
            # Early Stopping
            countdown -= 1
            if countdown == 0:
                break
            if (current_step + 1) == 5000:
                break
            
        self.logger.info(f"Finished training ({current_step + 1} steps)")

    def _get_lamb_search_config(self) -> LambSearchConfig:
        return self.stage_config.lamb_search or LambSearchConfig()

    def _capture_trial_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "mask_sampler": {
                key: value.clone() for key, value in self.mask_sampler.state_dict().items()
            },
            "oa_vecs": {
                key: value.clone() for key, value in self.oa_vecs.state_dict().items()
            },
        }

    def _restore_trial_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.mask_sampler.load_state_dict(state["mask_sampler"])
        self.oa_vecs.load_state_dict(state["oa_vecs"])

    def _evaluate_match_acc(
        self,
        num_test_step: int = 200,
        *,
        log_metrics: bool = False,
    ) -> Dict[str, float]:
        num_correct = 0.0
        num_match = 0.0
        task_loss = 0.0
        kl_div = 0.0
        batch_size = self.stage_config.test_batch_size
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collator)

        total_batches = 0
        total_examples = 0

        with torch.no_grad():
            for current_step, batch in enumerate(dataloader):
                batch = {k: v.to(self.config.torch_device) for k, v in batch.items()}
                current_bz = batch["input_ids"].size(0)
                total_examples += current_bz
                total_batches += 1
                labels = batch.pop("labels")

                target_logits = self.original_model(**batch).logits

                masks = self.mask_sampler.sample_binary_masks(current_bz)
                logits = self.hooked_model(
                    masks=masks,
                    oa_vecs=self.oa_vecs,
                    **batch
                ).logits

                result = self.loss_module.compute_batch(
                    logits, target_logits, labels, batch["input_ids"]
                )
                num_correct += result.acc_task * current_bz
                num_match += result.acc_match * current_bz
                task_loss += result.task_loss
                kl_div += result.distillation_loss.item()

                if log_metrics:
                    val_metrics = {
                        "stage": f"Stage {self.stage_idx + 1}",
                        "timestamp": time.time(),
                        "step": current_step,
                        "split": "val",
                        "kl_div": kl_div,
                        "task_loss": task_loss,
                        "acc_task": num_correct / total_examples,
                        "acc_match": num_match / total_examples,
                    }
                    if self._lambda_trial_label is not None:
                        val_metrics["lambda_trial"] = self._lambda_trial_label
                        val_metrics["lamb"] = self.lamb
                    self.metrics_logger.log(**val_metrics)

                if current_step + 1 == num_test_step:
                    break

        masks = self.mask_sampler.sample_binary_masks(1).squeeze(0)
        num_edges = (masks == 1).sum().item()

        return {
            "acc_match": num_match / total_examples,
            "acc_task": num_correct / total_examples,
            "task_loss": task_loss / total_batches,
            "kl_div": kl_div / total_batches,
            "num_edges": float(num_edges),
        }

    def _get_baseline_acc(self) -> float:
        if self.config.baseline_acc is not None:
            return self.config.baseline_acc

        if "pruning_baseline_acc" in self.output_dict:
            self.config.baseline_acc = float(self.output_dict["pruning_baseline_acc"])
            return self.config.baseline_acc

        metrics = self._evaluate_match_acc()
        self.config.baseline_acc = metrics["acc_match"]
        self.logger.info(f"Pruning baseline acc_match={self.config.baseline_acc:.4f}")
        return self.config.baseline_acc

    def _reference_acc_for_stage(self) -> float:
        if self.stage_idx == 0:
            return self._evaluate_match_acc()["acc_match"]

        return float(self.output_dict["acc_match"])

    def _log_lambda_probe(
        self,
        trial: LambdaTrialResult,
        threshold_acc: float,
        baseline_acc: float,
        reference_acc: float,
    ) -> None:
        self.metrics_logger.log(
            task="lambda_probe",
            stage=f"Stage {self.stage_idx + 1}",
            lamb=trial.lamb,
            probe=trial.probe,
            acc_match=trial.acc_match,
            num_edges=trial.num_edges,
            threshold_acc=threshold_acc,
            baseline_acc=baseline_acc,
            reference_acc=reference_acc,
            feasible=trial.feasible,
        )

    def run_trial(
        self,
        lamb: float,
        num_steps: int,
        probe: bool,
        oa_param_groups: List[Dict[str, Any]],
        trial_state: Dict[str, Dict[str, torch.Tensor]],
        threshold_acc: float,
        baseline_acc: float,
        reference_acc: float,
    ) -> LambdaTrialResult:
        self._restore_trial_state(trial_state)
        self.lamb = lamb
        self._set_lambda_trial_label(lamb, probe=probe)
        try:
            self.train(oa_param_groups=oa_param_groups, num_steps_override=num_steps)
            metrics = self._evaluate_match_acc()
            trial = LambdaTrialResult(
                acc_match=metrics["acc_match"],
                num_edges=int(metrics["num_edges"]),
                lamb=lamb,
                probe=probe,
                feasible=metrics["acc_match"] >= threshold_acc,
            )
            self._log_lambda_probe(trial, threshold_acc, baseline_acc, reference_acc)
            self.logger.info(
                f"Lambda probe lamb={lamb:.2e} probe={probe} "
                f"acc_match={trial.acc_match:.4f} num_edges={trial.num_edges} "
                f"feasible={trial.feasible}"
            )
            return trial
        finally:
            self._clear_lambda_trial_label()

    def run_with_lambda_search(self, oa_param_groups: List[Dict[str, Any]]) -> None:
        search_cfg = self._get_lamb_search_config()
        baseline_acc = self._get_baseline_acc()
        reference_acc = self._reference_acc_for_stage()
        threshold_acc = compute_threshold_acc(
            baseline_acc,
            reference_acc,
            self.config.relative_gap,
        )
        probe_steps = resolve_probe_num_steps(self.num_steps, search_cfg.probe_num_steps)
        trial_state = self._capture_trial_state()

        self.logger.info(
            f"Lambda search stage {self.stage_idx + 1}: "
            f"baseline_acc={baseline_acc:.4f} reference_acc={reference_acc:.4f} "
            f"threshold_acc={threshold_acc:.4f} probe_steps={probe_steps}"
        )

        probe_lambdas = suggest_probe_lambdas(
            center_lamb=self.stage_config.lamb,
            probe_lambdas=search_cfg.probe_lambdas,
            scale_factor=search_cfg.probe_scale_factor,
            max_probes=search_cfg.max_probes,
        )

        trials: List[LambdaTrialResult] = []
        for lamb in probe_lambdas:
            trials.append(
                self.run_trial(
                    lamb=lamb,
                    num_steps=probe_steps,
                    probe=True,
                    oa_param_groups=oa_param_groups,
                    trial_state=trial_state,
                    threshold_acc=threshold_acc,
                    baseline_acc=baseline_acc,
                    reference_acc=reference_acc,
                )
            )

        if search_cfg.enable_refinement and bracket_exists(trials, threshold_acc):
            above = [t.lamb for t in trials if t.acc_match >= threshold_acc]
            below = [t.lamb for t in trials if t.acc_match < threshold_acc]
            refined_lamb = suggest_refined_lambdas(above, below)
            if refined_lamb is not None:
                trials.append(
                    self.run_trial(
                        lamb=refined_lamb,
                        num_steps=probe_steps,
                        probe=True,
                        oa_param_groups=oa_param_groups,
                        trial_state=trial_state,
                        threshold_acc=threshold_acc,
                        baseline_acc=baseline_acc,
                        reference_acc=reference_acc,
                    )
                )

        chosen, used_fallback = select_best_trial(trials, threshold_acc)
        if used_fallback:
            self.logger.warning(
                f"No lambda probe met threshold {threshold_acc:.4f}; "
                f"falling back to smallest lambda {chosen.lamb:.2e}"
            )
        else:
            self.logger.info(
                f"Selected lambda {chosen.lamb:.2e} with acc_match={chosen.acc_match:.4f} "
                f"and num_edges={chosen.num_edges}"
            )

        self.metrics_logger.log(
            task="lambda_search_selected",
            stage=f"Stage {self.stage_idx + 1}",
            lamb=chosen.lamb,
            acc_match=chosen.acc_match,
            num_edges=chosen.num_edges,
            threshold_acc=threshold_acc,
            used_fallback=used_fallback,
        )

        self._restore_trial_state(trial_state)
        self.lamb = chosen.lamb
        self.logger.info(f"Final training with lamb={self.lamb:.2e} for {self.num_steps} steps...")
        self._set_lambda_trial_label(chosen.lamb, probe=False)
        try:
            self.train(oa_param_groups=oa_param_groups)
            self.test()
        finally:
            self._clear_lambda_trial_label()
        self.transform_graph()
        self.save()

    def test(self):
        metrics = self._evaluate_match_acc(log_metrics=True)
        acc_match = metrics["acc_match"]
        acc_task = metrics["acc_task"]
        task_loss = metrics["task_loss"]
        kl_div = metrics["kl_div"]
        num_edges = int(metrics["num_edges"])
        self.logger.info(f"After Pruning Edge Count: {num_edges}")
        
        # Prepare output dictionary
        self.output_dict[f"result_patching_performance_global_iteration_{self.stage_idx}"] = {
            "acc_match": acc_match,
            "acc_task": acc_task,
            "task_loss": task_loss,
            "kl_div": kl_div,
            "num_edges": num_edges,
            "coef": self.lamb
        }
        
        self.hooked_model.remove_hooks()
        
        # Save optimal ablation vectors and output training results
        self.output_dict['acc_match'] = acc_match
        self.output_dict['acc_task'] = acc_task
        self.output_dict['kl_div'] = kl_div
        self.output_dict['task_loss'] = task_loss
        if self.stage_idx == 0 and self.config.baseline_acc is not None:
            self.output_dict['pruning_baseline_acc'] = self.config.baseline_acc

    def save(self):
        # Save optimal ablation vectors and output training JSON
        torch.save(self.oa_vecs, self.config.full_output_dir / f'stage{self.stage_idx + 1}' / 'oa_vecs.pt')
        output_file = self.config.full_output_dir / f'stage{self.stage_idx + 1}' / 'output.json'
        with open(output_file, "w") as f:
            json.dump(self.output_dict, f, indent=4)
        
        # Save the pruned model
        model_dir = self.config.full_output_dir / f'stage{self.stage_idx + 1}' / 'pruned_model'
        model_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_dir)
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def transform_graph(self):
        pass
