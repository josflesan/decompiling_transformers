import os
import random
from typing import Any

import numpy as np
import torch
import yaml
from dacite import from_dict

from rasp.utilities.rasp_dataclasses import RaspRunConfig


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def convert_keys_to_int(obj: Any) -> Any:
    """Recursively convert string keys that are integer-like to int."""
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [convert_keys_to_int(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_keys_to_int(v) for v in obj)
    return obj


def load_config(config_path: str) -> RaspRunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = from_dict(RaspRunConfig, raw)
    set_seed(config.seed)
    torch.set_printoptions(sci_mode=False, precision=5)
    return config
