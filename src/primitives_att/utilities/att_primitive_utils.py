import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import yaml
from enum import Enum
from dacite import from_dict

from primitives_att.utilities.att_primitive_dataclasses import PrimitiveConfig, AttPrimitivesConfig
from utilities.core import TaskConfig

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_path: str) -> None:
    """Loads YAML file for Att primitive run and uses dataclasses to build structured output"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    # Convert list of primitives into PrimitiveConfig objects
    raw['att_primitives_matrix'] = [
        PrimitiveConfig(**primitive) for primitive in raw['att_primitives_matrix']
    ]
    raw['att_primitives_bias'] = [
        PrimitiveConfig(**primitive) for primitive in raw['att_primitives_bias']
    ]
    
    raw['unembedding_primitives_matrix'] = [
        PrimitiveConfig(**primitive) for primitive in raw['unembedding_primitives_matrix']
    ]
    raw['unembedding_primitives_bias'] = [
        PrimitiveConfig(**primitive) for primitive in raw['unembedding_primitives_bias']
    ]
    
    # Convert task config dict into TaskConfig object
    raw['task_config'] = TaskConfig(**raw['task_config'])
    
    # Instantiate AttPrimitivesConfig
    config = from_dict(AttPrimitivesConfig, raw)
    set_seed(config.seed)
    
    # Set Pytorch print options
    torch.set_printoptions(sci_mode=False, precision=5)
    
    return config
