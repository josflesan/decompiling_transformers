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
from .pruning_dataclasses import LambSearchConfig, PruningRunConfig, StageConfig
from utilities.core import TaskConfig


def _build_stage_config(stage_raw: Dict[str, Any]) -> StageConfig:
    stage_raw = dict(stage_raw)
    lamb_search = stage_raw.get("lamb_search")
    if lamb_search is not None:
        stage_raw["lamb_search"] = LambSearchConfig(**lamb_search)
    return StageConfig(**stage_raw)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def build_config_from_dict(raw: Dict[str, Any]) -> PruningRunConfig:
    raw = dict(raw)

    raw["pruning_stages"] = {
        key: _build_stage_config(value) for key, value in raw["pruning_stages"].items()
    }
    raw["task_config"] = TaskConfig(**raw["task_config"])

    return from_dict(PruningRunConfig, raw)


def load_config(config_path: str) -> PruningRunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = build_config_from_dict(raw)
    set_seed(config.seed)
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
