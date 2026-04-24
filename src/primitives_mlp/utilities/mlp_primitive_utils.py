'''
- RegressionSolver: Utility Module to perform linear mapping recovery. This uses the activation
tensors v and w and returns C using torch.linalg.pinv
'''

import torch
import random
import numpy as np
import os
import yaml
from enum import Enum
from dacite import from_dict

from primitives_mlp.utilities.registry import PRIMITIVE_REGISTRY
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveConfig, MLPPrimitivesConfig

class PrimitiveType(Enum):
    EQUALS = ("equal", True)  # Name, single_input
    ERASE = ("erase", True)
    EXISTS = ("exists", True)
    FORALL = ("forall", True)
    HARDEN = ("harden", True)
    NOOP = ("noop", True)
    SHARPEN = ("sharpen", True)
    ZEROONE = ("zeroone", True)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_path: str) -> None:
    """Loads YAML file for MLP primitive run and uses dataclasses to build structured output"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    # Convert list of primitives into PrimitiveConfig objects
    raw['mlp_primitives'] = [
        PrimitiveConfig(**primitive) for primitive in raw['mlp_primitives']
    ]
    
    # Instantiate MLPPrimitivesConfig
    config = from_dict(MLPPrimitivesConfig, raw)
    set_seed(config.seed)
    
    # Set Pytorch print options
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
