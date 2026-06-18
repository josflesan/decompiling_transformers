"""
Evaluates model behaviour with attention/unembedding primitive replacements.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from math import isnan
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from data.CustomTokenizer import CustomTokenizer
from primitives_att.core.MatrixRounder import MatrixRounder
from primitives_att.core.hooks import attention_primitive_forward, lm_head_primitive_hook
from primitives_att.utilities.att_primitive_dataclasses import EvalMetrics
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from pruning.core.hooks import GPT2QKHooks
from pruning.core.mask_samplers import QKMaskSampler
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from utilities.core import LossModule
from utilities.metrics_logger import MetricsLogger

PrimitivesMap = Dict[Any, Any]
ConvertedMlp = Dict[str, PrimitiveSearchOutput]


def _save_wte_inputs(module, input, output, model):
    model.wte_inputs = input[0].detach()


def _save_wpe_inputs(module, input, output, model):
    model.wpe_inputs = input[0].detach()


class PrimitiveExecutionEngine:
    def __init__(
        self,
        hooked_model: GPT2QKHooks,
        original_model: GPT2LMHeadModel,
        dataloader: DataLoader,
        mask_sampler: QKMaskSampler,
        oa_vecs: OptimalQueryBiasVectors,
        tokenizer: CustomTokenizer,
        converted_mlp: ConvertedMlp,
        loss_module: LossModule,
        logger: Optional[logging.Logger] = None,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.hooked_model = hooked_model
        self.original_model = original_model
        self.dataloader = dataloader
        self.mask_sampler = mask_sampler
        self.oa_vecs = oa_vecs
        self.tokenizer = tokenizer
        self.converted_mlp = converted_mlp
        self.loss_module = loss_module
        self.logger = logger
        self.metrics_logger = metrics_logger

    def evaluate(
        self,
        primitives_map: PrimitivesMap,
        num_steps: int,
        batch_size: int,
        replace_attention: bool = True,
        replace_logits: bool = True,
    ) -> EvalMetrics:
        hooks = []
        hooks.append(
            self.hooked_model.model.transformer.wte.register_forward_hook(
                partial(_save_wte_inputs, model=self.hooked_model)
            )
        )
        hooks.append(
            self.hooked_model.model.transformer.wpe.register_forward_hook(
                partial(_save_wpe_inputs, model=self.hooked_model)
            )
        )

        prev_attn_forward = self.hooked_model.attention_forward
        if replace_attention:
            self.hooked_model.attention_forward = lambda module, layer: attention_primitive_forward(
                self.hooked_model,
                module,
                layer,
                primitives=primitives_map,
                tokenizer=self.tokenizer,
                converted_mlp=self.converted_mlp,
                oa_vecs=self.oa_vecs,
            )

        lm_head_hook_handle = None
        if replace_logits:
            for hook in self.hooked_model.hooks:
                if (
                    hook.id in hook.__getstate__()[0]
                    and hook.__getstate__()[0][hook.id] == self.hooked_model.lm_head_hook
                ):
                    hook.remove()
                    break
            
            lm_head_hook_handle = self.hooked_model.model.lm_head.register_forward_hook(
                partial(
                    lm_head_primitive_hook,
                    hooked_model=self.hooked_model,
                    primitives=primitives_map,
                    tokenizer=self.tokenizer,
                    converted_mlp=self.converted_mlp,
                    oa_vecs=self.oa_vecs,
                )
            )
            hooks.append(lm_head_hook_handle)

        dataset = self.dataloader.dataset
        collator = self.dataloader.collate_fn
        num_correct_items = 0
        num_match_items = 0
        sum_kl = 0.0
        sum_task_loss = 0.0
        current_step = 0
        inputs = []

        with torch.no_grad():
            for item in dataset:
                inputs.append(item)
                if len(inputs) < batch_size:
                    continue

                batch = collator(inputs)
                batch = {k: v.to(self.hooked_model.device) for k, v in batch.items()}
                labels = batch["labels"]
                batch.pop("labels")

                masks = self.mask_sampler.sample_binary_masks(batch_size)
                self.hooked_model.input_ids = batch["input_ids"]
                self.hooked_model.position_ids = batch["position_ids"]

                result = self.hooked_model(masks=masks, oa_vecs=self.oa_vecs, **batch)
                logits = result.logits.detach()
                target_logits = self.original_model(**batch).logits.detach()

                result = self.loss_module.compute_batch(
                    logits, target_logits, labels, batch["input_ids"]
                )
                sum_task_loss += result.task_loss
                sum_kl += result.distillation_loss.item()
                num_correct_items += result.acc_task * batch_size
                num_match_items += result.acc_match * batch_size

                inputs = []
                current_step += 1
                if current_step == num_steps:
                    break

        total_items = num_steps * batch_size
        metrics = EvalMetrics(
            acc=num_correct_items / total_items,
            kl=sum_kl / num_steps,
            acc_match=num_match_items / total_items,
            task_loss=sum_task_loss / num_steps,
        )

        if replace_attention:
            self.hooked_model.attention_forward = prev_attn_forward
        for hook in hooks:
            hook.remove()
        
        if replace_logits and lm_head_hook_handle is not None:
            self.hooked_model.hooks.append(
                self.hooked_model.model.lm_head.register_forward_hook(
                    self.hooked_model.lm_head_hook
                )
            )

        return metrics

    def _empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def train_rounders(
        self,
        primitives_map: PrimitivesMap,
        rounders: Dict[str, MatrixRounder],
        training_lambda: float,
        num_steps: int,
        batch_size: int,
        lr: float,
        log_interval: int,
        match_acc_threshold: float,
        stage: Optional[int] = None,
    ) -> None:
        if len(rounders) == 0:
            return

        hooks = []
        hooks.append(
            self.hooked_model.model.transformer.wte.register_forward_hook(
                partial(_save_wte_inputs, model=self.hooked_model)
            )
        )
        hooks.append(
            self.hooked_model.model.transformer.wpe.register_forward_hook(
                partial(_save_wpe_inputs, model=self.hooked_model)
            )
        )

        prev_attn_forward = self.hooked_model.attention_forward
        self.hooked_model.attention_forward = lambda module, layer: attention_primitive_forward(
            self.hooked_model,
            module,
            layer,
            primitives=primitives_map,
            tokenizer=self.tokenizer,
            converted_mlp=self.converted_mlp,
            oa_vecs=self.oa_vecs,
        )

        for hook in self.hooked_model.hooks:
            if (
                hook.id in hook.__getstate__()[0]
                and hook.__getstate__()[0][hook.id] == self.hooked_model.lm_head_hook
            ):
                hook.remove()
                break

        hooks.append(
            self.hooked_model.model.lm_head.register_forward_hook(
                partial(
                    lm_head_primitive_hook,
                    hooked_model=self.hooked_model,
                    primitives=primitives_map,
                    tokenizer=self.tokenizer,
                    converted_mlp=self.converted_mlp,
                    oa_vecs=self.oa_vecs,
                )
            )
        )

        for rounder in rounders.values():
            rounder.set_stage(stage)

        all_params = [param for rounder in rounders.values() for param in rounder.parameters()]
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0, betas=(0.9, 0.995))
        dataset = self.dataloader.dataset
        collator = self.dataloader.collate_fn
        device = self.hooked_model.device

        current_step = 0
        inputs = []
        training_logs: dict[str, list[float]] = defaultdict(list)
        steps_below_match_threshold = 0
        match_acc_patience = 1

        def save_valid_checkpoint(step: int) -> dict[str, Any]:
            return {
                "rounder_states": {
                    name: {k: v.detach().clone() for k, v in rounder.state_dict().items()}
                    for name, rounder in rounders.items()
                },
                "step": step,
            }

        def revert_to_valid_checkpoint(checkpoint: Optional[dict[str, Any]]) -> bool:
            if checkpoint is None:
                if self.logger is not None:
                    self.logger.info("No valid checkpoint available to revert to")
                return False
            if self.logger is not None:
                self.logger.info(
                    f"Reverting to valid checkpoint from step {checkpoint['step']}"
                )
            for name, rounder in rounders.items():
                rounder.load_state_dict(checkpoint["rounder_states"][name])
            optimizer.zero_grad()
            return True

        valid_checkpoint = save_valid_checkpoint(0)
        if self.metrics_logger is not None:
            self.metrics_logger.log(
                task="att_round_training_start",
                stage=stage,
                num_steps=num_steps,
                batch_size=batch_size,
                training_lambda=training_lambda,
                lr=lr,
                log_interval=log_interval,
                match_acc_threshold=match_acc_threshold,
                rounder_count=len(rounders),
            )

        stop_reason = "completed"

        for item in dataset:
            inputs.append(item)
            if len(inputs) < batch_size:
                continue

            batch = collator(inputs)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            batch.pop("labels")

            masks = self.mask_sampler.sample_binary_masks(batch_size)
            self.hooked_model.input_ids = batch["input_ids"]
            self.hooked_model.position_ids = batch["position_ids"]

            result = self.hooked_model(masks=masks, oa_vecs=self.oa_vecs, **batch)
            logits = result.logits

            with torch.no_grad():
                target_logits = self.original_model(**batch).logits.detach()

            result = self.loss_module.compute_batch(
                logits, target_logits, labels, batch["input_ids"]
            )
            task_loss = result.task_loss
            loss = result.distillation_loss
            acc = result.acc_task
            match_acc = result.acc_match

            if isnan(loss.item()):
                if self.logger is not None:
                    self.logger.info("Stop learning because loss is nan")
                revert_to_valid_checkpoint(valid_checkpoint)
                stop_reason = "nan_loss"
                break

            training_logs["task_loss"].append(task_loss)
            training_logs["loss"].append(loss.item())
            training_logs["acc"].append(acc)
            training_logs["match_acc"].append(match_acc)

            penalty_tensor = sum(rounder.get_penalty() for rounder in rounders.values())
            if next(iter(rounders.values())).average_over_pixels:
                num_pixels = sum(rounder.indep_prod.numel() for rounder in rounders.values())
                penalty_tensor = penalty_tensor / num_pixels

            penalty = float(penalty_tensor.item())
            training_logs["penalty"].append(penalty)
            if isnan(penalty):
                if self.logger is not None:
                    self.logger.info("Stop learning because penalty is nan")
                revert_to_valid_checkpoint(valid_checkpoint)
                stop_reason = "nan_penalty"
                break

            loss = loss + training_lambda * penalty_tensor
            training_logs["full_loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inputs = []
            current_step += 1
            self._empty_cache()

            if current_step % log_interval == 0:
                avg_logs = {k: sum(v) / len(v) for k, v in training_logs.items()}
                if self.logger is not None:
                    self.logger.info(f"Round training step {current_step}: {avg_logs}")
                if self.metrics_logger is not None:
                    self.metrics_logger.log(
                        task="att_round_training_step",
                        stage=stage,
                        step=current_step,
                        total_steps=num_steps,
                        progress=(current_step / num_steps) if num_steps > 0 else 0.0,
                        task_loss=avg_logs.get("task_loss"),
                        loss=avg_logs.get("loss"),
                        full_loss=avg_logs.get("full_loss"),
                        penalty=avg_logs.get("penalty"),
                        acc=avg_logs.get("acc"),
                        match_acc=avg_logs.get("match_acc"),
                        match_acc_threshold=match_acc_threshold,
                        below_threshold=avg_logs.get("match_acc", 0.0) < match_acc_threshold,
                        steps_below_threshold=steps_below_match_threshold,
                        rounder_count=len(rounders),
                    )
                cur_match_acc = avg_logs["match_acc"]
                training_logs = defaultdict(list)

                if cur_match_acc < match_acc_threshold:
                    steps_below_match_threshold += 1
                    if steps_below_match_threshold >= match_acc_patience:
                        if self.logger is not None:
                            self.logger.info(
                                "Stop learning: match accuracy below threshold "
                                f"{match_acc_threshold:.4f} for {match_acc_patience} "
                                f"consecutive intervals (current={cur_match_acc:.4f})"
                            )
                        revert_to_valid_checkpoint(valid_checkpoint)
                        stop_reason = "match_acc_below_threshold"
                        break
                else:
                    steps_below_match_threshold = 0
                    valid_checkpoint = save_valid_checkpoint(current_step)

            if current_step == num_steps:
                break

        self.hooked_model.attention_forward = prev_attn_forward
        for hook in hooks:
            hook.remove()
        
        self.hooked_model.hooks.append(
            self.hooked_model.model.lm_head.register_forward_hook(
                self.hooked_model.lm_head_hook
            )
        )
        if self.metrics_logger is not None:
            self.metrics_logger.log(
                task="att_round_training_complete",
                stage=stage,
                final_step=current_step,
                total_steps=num_steps,
                progress=(current_step / num_steps) if num_steps > 0 else 0.0,
                stop_reason=stop_reason,
                match_acc_threshold=match_acc_threshold,
                rounder_count=len(rounders),
            )
