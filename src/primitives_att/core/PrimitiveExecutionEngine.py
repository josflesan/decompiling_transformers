"""
Evaluates model behaviour with attention/unembedding primitive replacements.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from data.CustomCollator import CustomCollator
from data.CustomTokenizer import CustomTokenizer
from primitives_att.core.hooks import attention_primitive_forward, lm_head_primitive_hook
from primitives_att.utilities.att_primitive_dataclasses import EvalMetrics
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from pruning.core.hooks import GPT2QKHooks
from pruning.core.mask_samplers import QKMaskSampler
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors

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
        logger: Optional[logging.Logger] = None,
        use_bce: bool = False,
    ):
        self.hooked_model = hooked_model
        self.original_model = original_model
        self.dataloader = dataloader
        self.mask_sampler = mask_sampler
        self.oa_vecs = oa_vecs
        self.tokenizer = tokenizer
        self.converted_mlp = converted_mlp
        self.logger = logger
        self.use_bce = use_bce

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

                if not self.use_bce:
                    sum_task_loss += F.cross_entropy(
                        logits[:, :-1].flatten(end_dim=1),
                        labels[:, 1:].flatten(),
                    ).item()

                    target_shift_logits = target_logits[:, :-1]
                    shift_logits = logits[:, :-1]
                    shift_labels = labels[:, 1:]

                    target_predictions = target_shift_logits.argmax(dim=-1)
                    predictions = shift_logits.argmax(dim=-1)

                    match = ((predictions == target_predictions) | (shift_labels == -100)).all(
                        dim=1
                    )
                    num_match_items += match.sum().item()
                    correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                    num_correct_items += correct.sum().item()

                    valid = shift_labels != -100
                    if valid.any():
                        sum_kl += F.kl_div(
                            F.log_softmax(shift_logits[valid], dim=-1),
                            F.log_softmax(target_shift_logits[valid], dim=-1),
                            log_target=True,
                        ).item()
                else:
                    mask = (batch["input_ids"] != self.tokenizer.pad_token_id) & (
                        batch["input_ids"] != self.tokenizer.eos_token_id
                    )
                    task_loss = F.binary_cross_entropy_with_logits(
                        logits, labels.float(), reduction="none"
                    )
                    sum_task_loss += task_loss[mask].mean().item()

                    kl_loss = F.binary_cross_entropy_with_logits(
                        logits, torch.sigmoid(target_logits), reduction="none"
                    )
                    sum_kl += kl_loss[mask].mean().item()

                    target_predictions = (target_logits > 0).long()
                    predictions = (logits > 0).long()
                    pad_mask = (batch["input_ids"] == self.tokenizer.pad_token_id) | (
                        batch["input_ids"] == self.tokenizer.eos_token_id
                    )
                    match = ((predictions == target_predictions).all(dim=-1) | pad_mask).all(dim=1)
                    num_match_items += match.sum().item()
                    correct = ((predictions == labels).all(dim=-1) | pad_mask).all(dim=1)
                    num_correct_items += correct.sum().item()

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
