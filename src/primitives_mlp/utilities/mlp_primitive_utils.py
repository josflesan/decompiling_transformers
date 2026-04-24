'''
- RegressionSolver: Utility Module to perform linear mapping recovery. This uses the activation
tensors v and w and returns C using torch.linalg.pinv
'''

import torch
import random
import numpy as np
import os
import yaml
from dacite import from_dict

from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveConfig, MLPPrimitivesConfig

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