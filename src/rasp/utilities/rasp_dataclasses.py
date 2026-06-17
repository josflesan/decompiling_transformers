from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CircledLabel:
    label: str
    code_var: str
    layer: Optional[int] = None
    head: Optional[int] = None
    q_path: Optional[str] = None
    k_path: Optional[str] = None
    inp_path: Optional[str] = None


@dataclass
class DecompilationResult:
    lines: List[str]
    selector_to_config: Dict[str, Any]
    var_mapping: Dict[str, str]
    circled_labels: Dict[str, CircledLabel] = field(default_factory=dict)
    deleted_lines: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RaspRunConfig:
    seed: int
    device: str
    exp_name: str
    output_dir: str
    convert_to_primitives: bool = True
    show_logits_for_unconverted_mlp: bool = False
    skip_convert: bool = False
    mlp_failure_threshold: float = 0.9

    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self.full_output_dir = Path(self.output_dir) / self.exp_name / "rasp"
        self.full_output_dir.mkdir(parents=True, exist_ok=True)

        self.torch_device = (
            torch.device("cuda")
            if self.device == "cuda" and torch.cuda.is_available()
            else torch.device("mps")
            if self.device == "mps" and torch.backends.mps.is_available()
            else torch.device("cpu")
        )


@dataclass
class RaspInputs:
    pruning_config: Dict[Any, Any]
    pruning_metrics: Dict[str, Any]
    split_mlp: bool
    converted_mlp: Dict[str, Any]
    interaction_map: Dict[Any, Any]
    att_stats: Dict[str, Any]
    mlp_input_output: Optional[Dict[Any, Any]] = None
