import torch
import random
import numpy as np
import os
import json
import yaml
from typing import Any, Dict, List

from pathlib import Path
from transformers import GPT2LMHeadModel
from dacite import from_dict
from .pruning_dataclasses import PruningRunConfig, StageConfig, TaskConfig

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def load_config(config_path: str) -> None:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
        
    # Convert stage_config dict into StageConfig
    raw['stage_config'] = StageConfig(**raw['stage_config'])
    
    # Convert task_config dict into TaskConfig
    raw['task_config'] = TaskConfig(**raw['task_config'])
        
    # Instantiate RunConfig
    config = from_dict(PruningRunConfig, raw)
    set_seed(config.seed)
    
    # Set Pytorch print options
    torch.set_printoptions(sci_mode=False, precision=5)
    
    return config

def output_model_arch_json(
    model: GPT2LMHeadModel,
    out_dir: Path
) -> None:
    layer_configs = []
    for block in model.transformer.h:
        layer_info = {
            "attn": str(block.attn),
            "mlp": str(block.mlp),
            "ln_1": str(block.ln_1),
            "ln_2": str(block.ln_2),
        }
        layer_configs.append(layer_info)
        
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(layer_configs, f, indent=4)
        
def get_full_possible_config_for_pruning(num_heads_per_layer: List[int]) -> Dict[str, List[Any]]:
    """
    This function builds a dependency map for pruning, as defined by the authors in Appendix F. In other words,
    we map which activations each module depends on.

    Args:
        num_heads_per_layer (List[int]): number of heads for each transformer layer in the GPT2 model

    Returns:
        Dict[str, List[Any]]: dependency map for each of the model layers
    """
    
    # Add transformer layer dependencies (for keys, queries, values and MLP)
    # Example: keys depend on WTE, WPE, all previous attention head outputs and all previous MLP outputs
    full_config = {
        layer: {
            "k": {
                head: ["wte", "wpe"] + 
                    [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                    [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "q": {
                head: ["wte", "wpe"] + 
                    [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                    [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "v": {
                head: ["wte", "wpe"] + 
                    [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + 
                    [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "mlp": ["wte", "wpe"] + 
                [f"attn_output-{l}-{h}" for l in range(layer + 1) for h in range(num_heads_per_layer[l])] + 
                [f"mlp-{l}" for l in range(layer)] 
        }
        for layer in range(len(num_heads_per_layer))
    }
    
    # Add the LM head dependencies
    full_config.update({
        "lm_head": ["wte", "wpe"] + 
            [f"attn_output-{l}-{h}" for l in range(len(num_heads_per_layer)) for h in range(num_heads_per_layer[l])] + 
            [f"mlp-{l}" for l in range(len(num_heads_per_layer))]
    })
    
    return full_config
