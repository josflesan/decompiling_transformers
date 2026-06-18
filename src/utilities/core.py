"""This file contains a range of basic utilities used throughout the program"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def int_key_hook(d):
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


def dataset_use_bce(dataset) -> bool:
    return getattr(dataset, "bce", getattr(dataset, "BCE", False))


@dataclass
class TaskConfig:
    name: str
    train_length_range: List[int]
    val_length_range: List[int]
    max_test_length: int


@dataclass
class BatchLossResult:
    task_loss: float
    distillation_loss: Tensor
    acc_task: float
    acc_match: float


class LossModule:
    def __init__(self, use_bce: bool, pad_token_id: int, eos_token_id: int | None = None):
        self._use_bce = use_bce
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_dataset(cls, dataset, tokenizer) -> LossModule:
        return cls(
            dataset_use_bce(dataset),
            tokenizer.pad_token_id,
            getattr(tokenizer, "eos_token_id", None),
        )

    @property
    def use_bce(self) -> bool:
        return self._use_bce

    def collator(self):
        if self.use_bce:
            from data.BCECollator import BCECollator

            return BCECollator(self.pad_token_id)
        from data.CustomCollator import CustomCollator

        return CustomCollator(self.pad_token_id)

    def _position_mask(self, input_ids: Tensor) -> Tensor:
        mask = input_ids != self.pad_token_id
        if self.eos_token_id is not None:
            mask = mask & (input_ids != self.eos_token_id)
        return mask

    def task_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        input_ids: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        if not self.use_bce:
            return F.cross_entropy(
                logits[:, :-1].flatten(end_dim=1),
                labels[:, 1:].flatten(),
                reduction=reduction,
            )

        mask = self._position_mask(input_ids)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        if reduction == "none":
            return loss
        return loss[mask].mean()

    def distillation_loss(
        self,
        logits: Tensor,
        target_logits: Tensor,
        labels: Tensor,
        input_ids: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        if not self.use_bce:
            shift_labels = labels[:, 1:]
            shift_logits = logits[:, :-1]
            shift_target_logits = target_logits[:, :-1]
            valid = shift_labels != -100
            return F.kl_div(
                F.log_softmax(shift_logits[valid], dim=-1),
                F.log_softmax(shift_target_logits[valid], dim=-1),
                log_target=True,
                reduction=reduction,
            )

        mask = self._position_mask(input_ids)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            torch.sigmoid(target_logits),
            reduction="none",
        )
        if reduction == "none":
            return loss
        return loss[mask].mean()

    def compute_batch(
        self,
        logits: Tensor,
        target_logits: Tensor,
        labels: Tensor,
        input_ids: Tensor,
    ) -> BatchLossResult:
        distillation_loss = self.distillation_loss(
            logits, target_logits, labels, input_ids
        )

        with torch.no_grad():
            task_loss = self.task_loss(logits, labels, input_ids).item()
            acc_task, acc_match = self.batch_accuracy(
                logits, labels, target_logits, input_ids
            )

        return BatchLossResult(
            task_loss=task_loss,
            distillation_loss=distillation_loss,
            acc_task=acc_task,
            acc_match=acc_match,
        )

    def batch_accuracy(
        self,
        logits: Tensor,
        labels: Tensor,
        target_logits: Tensor,
        input_ids: Tensor,
    ) -> tuple[float, float]:
        batch_size = input_ids.size(0)

        if not self.use_bce:
            shift_logits = logits[:, :-1]
            shift_target_logits = target_logits[:, :-1]
            shift_labels = labels[:, 1:]

            target_predictions = shift_target_logits.argmax(dim=-1)
            predictions = shift_logits.argmax(dim=-1)

            match = ((predictions == target_predictions) | (shift_labels == -100)).all(
                dim=1
            )
            correct = ((predictions == shift_labels) | (shift_labels == -100)).all(
                dim=1
            )
            return correct.sum().item() / batch_size, match.sum().item() / batch_size

        mask = self._position_mask(input_ids)
        pad_mask = ~mask

        target_predictions = (target_logits > 0).long()
        predictions = (logits > 0).long()

        match = ((predictions == target_predictions).all(dim=-1) | pad_mask).all(dim=1)
        correct = ((predictions == labels).all(dim=-1) | pad_mask).all(dim=1)
        return correct.sum().item() / batch_size, match.sum().item() / batch_size
