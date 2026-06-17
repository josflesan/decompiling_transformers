"""
Learnable matrix replacement that sparsifies and rounds attention/unembedding products.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch


class MatrixRounder(torch.nn.Module):
    possible_params_and_default_values: Dict[str, Any] = {
        "train_scalar": False,
        "round_loss_coef": 1.0,
        "scalar_loss_coef": 1.0,
        "to_zero_loss_coef": 1.0,
        "two_stages": True,
        "average_over_pixels": True,
        "to_zero_loss_penalty": "l1",
        "round_loss_penalty": "l1",
    }

    def __init__(self, indep_prod: torch.Tensor, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.indep_prod = torch.nn.Parameter(copy.deepcopy(indep_prod))
        self.scalar = torch.nn.Parameter(torch.tensor(1.0, device=indep_prod.device))
        self.do_round = False

        for attr, value in MatrixRounder.possible_params_and_default_values.items():
            setattr(self, attr, value)
        if params:
            for attr in MatrixRounder.possible_params_and_default_values:
                if attr in params:
                    setattr(self, attr, params[attr])

        if not self.train_scalar:
            self.scalar.requires_grad = False

        self.stage = 1 if self.two_stages else None

    def set_stage(self, stage: Optional[int]) -> None:
        self.stage = stage

    def enable_rounding(self) -> None:
        self.do_round = True

    def disable_rounding(self) -> None:
        self.do_round = False

    def get_matrix(self) -> torch.Tensor:
        if self.do_round:
            return self.scalar * torch.round(self.indep_prod)
        
        return self.scalar * self.indep_prod

    def get_penalty(self) -> torch.Tensor:
        rounded_diff = torch.round(self.indep_prod) - self.indep_prod
        
        # Compute the different parts of the loss (rounding and to-zero)
        if self.round_loss_penalty == "l2":
            if self.average_over_pixels:
                reg_round = self.round_loss_coef * (rounded_diff**2).sum()
            else:
                reg_round = self.round_loss_coef * (rounded_diff**2).mean()
        
        elif self.round_loss_penalty == "l1":
            if self.average_over_pixels:
                reg_round = self.round_loss_coef * torch.abs(rounded_diff).sum()
            else:
                reg_round = self.round_loss_coef * torch.abs(rounded_diff).mean()
        else:
            raise NotImplementedError(f"Unknown round_loss_penalty: {self.round_loss_penalty}")

        if self.to_zero_loss_penalty == "l2":
            if self.average_over_pixels:
                reg_to_zero = self.to_zero_loss_coef * (self.indep_prod**2).sum()
            else:
                reg_to_zero = self.to_zero_loss_coef * (self.indep_prod**2).mean()
        elif self.to_zero_loss_penalty == "l1":
            if self.average_over_pixels:
                reg_to_zero = self.to_zero_loss_coef * torch.abs(self.indep_prod).sum()
            else:
                reg_to_zero = self.to_zero_loss_coef * torch.abs(self.indep_prod).mean()
        else:
            raise NotImplementedError(f"Unknown to_zero_loss_penalty: {self.to_zero_loss_penalty}")

        if self.stage is None:
            return reg_round + reg_to_zero
        if self.stage == 1:
            return reg_to_zero
        if self.stage == 2:
            return reg_round
        
        raise NotImplementedError(f"Stage {self.stage} not defined")
