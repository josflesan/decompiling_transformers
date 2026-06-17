import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import yaml
from enum import Enum
from dacite import from_dict

from primitives_mlp.utilities.registry import PRIMITIVE_REGISTRY
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveConfig, MLPPrimitivesConfig
from utilities.core import TaskConfig

class PrimitiveType(Enum):
    # Single-Input Primitives
    EQUALS = ("equal", True)  # Name, single_input
    ERASE = ("erase", True)
    EXISTS = ("exists", True)
    FORALL = ("forall", True)
    HARDEN = ("harden", True)
    NOOP = ("noop", True)
    SHARPEN = ("sharpen", True)
    ZEROONE = ("zeroone", True)
    
    # Multi-Input Primitives
    ERASE_MULTI = ("erase", False)
    COMBINE = ("combine", False)
    KEEPONE = ("keepone", False)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_config_from_dict(raw: dict) -> MLPPrimitivesConfig:
    raw = dict(raw)

    raw["mlp_primitives"] = [
        PrimitiveConfig(**primitive) for primitive in raw["mlp_primitives"]
    ]
    raw["task_config"] = TaskConfig(**raw["task_config"])

    return from_dict(MLPPrimitivesConfig, raw)


def load_config(config_path: str) -> MLPPrimitivesConfig:
    """Loads YAML file for MLP primitive run and uses dataclasses to build structured output"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = build_config_from_dict(raw)
    set_seed(config.seed)
    torch.set_printoptions(sci_mode=False, precision=5)

    return config

def build_primitive(ptype: PrimitiveType, **kwargs):
    pname, psingle = ptype.value
    return PRIMITIVE_REGISTRY[pname](type=ptype, name=pname, single_input=psingle, **kwargs)

def get_primitives(mlp_primitives_config: "MLPPrimitivesConfig"):
    """
    Reads the primitives defined in the config and returns a list of built Primitive instances.
    """
    type_lookup = {ptype.value[0]: ptype for ptype in PrimitiveType}
    built_primitives = []
    
    for config in mlp_primitives_config.mlp_primitives:
        ptype = type_lookup.get(config.type)
        if ptype is None:
            raise ValueError(f"Unknown primitive type: {config.type}")
        
        # Extract hyperparameters (pow, center, threshold) while ignoring None values
        kwargs = {
            k: v for k, v in vars(config).items() 
            if k != 'type' and v is not None
        }
        
        built_primitives.append(build_primitive(ptype, **kwargs))
        
    return built_primitives