import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from decompilation.utilities.decompilation_dataclasses import DecompilationRunConfig
from primitives_att.utilities.att_primitive_utils import build_config_from_dict as build_att_config
from primitives_mlp.utilities.mlp_primitive_utils import build_config_from_dict as build_mlp_config
from rasp.utilities.rasp_utils import build_config_from_dict as build_rasp_config
from pruning.utilities.pruning_utils import build_config_from_dict as build_pruning_config

SHARED_CONFIG_KEYS = (
    "seed",
    "device",
    "exp_name",
    "output_dir",
    "model_path",
    "task_config",
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _shared_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {key: raw[key] for key in SHARED_CONFIG_KEYS if key in raw}


def _merge_stage_config(raw: Dict[str, Any], stage_key: str) -> Dict[str, Any]:
    stage_raw = raw.get(stage_key, {})
    if not isinstance(stage_raw, dict):
        raise ValueError(f"Expected '{stage_key}' to be a mapping in the config file.")

    return {**_shared_config(raw), **stage_raw}


def load_config(
    config_path: str,
    *,
    run_pruning: Optional[bool] = None,
    run_mlp_primitives: Optional[bool] = None,
    run_att_primitives: Optional[bool] = None,
    run_conversion: Optional[bool] = None,
) -> DecompilationRunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if "task_config" not in raw:
        raise ValueError("Global config must define 'task_config'.")

    set_seed(raw.get("seed", 0))
    torch.set_printoptions(sci_mode=False, precision=5)

    return DecompilationRunConfig(
        pruning=build_pruning_config(_merge_stage_config(raw, "pruning")),
        mlp_primitives=build_mlp_config(_merge_stage_config(raw, "mlp_primitives")),
        att_primitives=build_att_config(_merge_stage_config(raw, "att_primitives")),
        rasp=build_rasp_config(_merge_stage_config(raw, "rasp")),
        run_pruning=run_pruning if run_pruning is not None else raw.get("run_pruning", True),
        run_mlp_primitives=(
            run_mlp_primitives
            if run_mlp_primitives is not None
            else raw.get("run_mlp_primitives", True)
        ),
        run_att_primitives=(
            run_att_primitives
            if run_att_primitives is not None
            else raw.get("run_att_primitives", True)
        ),
        run_conversion=run_conversion if run_conversion is not None else raw.get("run_conversion", True),
    )

def validate_stage_prerequisites(config: DecompilationRunConfig) -> None:
    output_dir = Path(config.pruning.output_dir)
    exp_name = config.pruning.exp_name
    pruning_dir = output_dir / exp_name / "pruning" / "stage3"
    needs_pruning_output = (
        not config.run_pruning
        and (config.run_mlp_primitives or config.run_att_primitives)
    )
    if needs_pruning_output:
        _require_artifact(pruning_dir / "output.json", "pruning")
        _require_artifact(pruning_dir / "oa_vecs.pt", "pruning")
    if not config.run_mlp_primitives and config.run_att_primitives:
        _require_artifact(
            output_dir / exp_name / "mlp_primitives" / "converted_mlp.pt",
            "MLP primitive replacement",
        )

def _require_artifact(path: Path, stage_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot skip {stage_name}: required artifact missing at {path}"
        )