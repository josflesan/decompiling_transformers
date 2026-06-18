"""
Helpers for determining activation dimensionality and filtering primitive candidates.
"""

from __future__ import annotations

import re
from itertools import product
from typing import Any, Dict, List, Optional

from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import AttentionInteraction, LogitsInteraction
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput

def expand_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid_keys = [k for k, v in params.items() if isinstance(v, list)]
    fixed_keys = [k for k, v in params.items() if not isinstance(v, list)]
    grid_values = [params[k] for k in grid_keys]

    combos = []
    for combo in product(*grid_values):
        conf = {k: v for k, v in zip(grid_keys, combo)}
        conf.update({k: params[k] for k in fixed_keys})
        combos.append(conf)
    return combos


def is_token_dim_activation(
    act_name: str,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
) -> bool:
    split_nodes = re.compile(r"attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe").findall(act_name)

    for idx, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            continue
        if node == "wte":
            return True
        if node == "wpe":
            return False
        if node.startswith("mlp"):
            converted_mlp_name = "-".join(split_nodes[idx:])
            if converted_mlp_name not in converted_mlp:
                return False
            mlp_primitive = converted_mlp[converted_mlp_name].best_primitive
            
            # MLP primitives that preserve token-index dimensionality along the path
            if mlp_primitive.name in set(["noop", "sharpen", "harden"]):
                continue
            
            return False
        raise RuntimeError(f"Unrecognized node in activation path: {node}")

    raise RuntimeError(f"Could not resolve token dimensionality for path: {act_name}")


def _filter_by_scalar_default(
    primitives: List[Primitive],
    only_default_scalars: bool,
) -> List[Primitive]:
    if not only_default_scalars:
        return primitives
    return [p for p in primitives if p.info.has_default_scalar]


def _filter_non_token_primitives(primitives: List[Primitive]) -> List[Primitive]:
    return [p for p in primitives if not p.info.is_only_token]


def filter_attention_candidates(
    interaction: AttentionInteraction,
    matrix_primitives: List[Primitive],
    bias_primitives: List[Primitive],
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    only_default_scalars: bool,
) -> Dict[str, Optional[List[Primitive]]]:
    if interaction.activation_name_to_keep_q is None:
        candidates = bias_primitives
        k_token_dim = is_token_dim_activation(
            interaction.activation_name_to_keep_k, converted_mlp
        )
        if not k_token_dim:
            candidates = _filter_non_token_primitives(candidates)
        
        return {
            "primitive": _filter_by_scalar_default(candidates, only_default_scalars),
            "special_primitive": None,
        }

    q_token_dim = is_token_dim_activation(
        interaction.activation_name_to_keep_q, converted_mlp
    )
    k_token_dim = is_token_dim_activation(
        interaction.activation_name_to_keep_k, converted_mlp
    )
    candidates = matrix_primitives

    if k_token_dim:
        if q_token_dim:
            special = _filter_by_scalar_default(candidates, only_default_scalars)
            return {
                "primitive": _filter_by_scalar_default(candidates, only_default_scalars),
                "special_primitive": special,
            }
        return {
            "primitive": _filter_by_scalar_default(candidates, only_default_scalars),
            "special_primitive": None,
        }

    if q_token_dim:
        filtered = [
            p
            for p in _filter_by_scalar_default(candidates, only_default_scalars)
            if not p.info.is_only_token
        ]
        return {"primitive": filtered, "special_primitive": filtered.copy()}

    filtered = _filter_non_token_primitives(
        _filter_by_scalar_default(candidates, only_default_scalars)
    )
    return {"primitive": filtered, "special_primitive": None}


def filter_logits_candidates(
    interaction: LogitsInteraction,
    matrix_primitives: List[Primitive],
    bias_primitives: List[Primitive],
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    only_default_scalars: bool,
) -> Dict[str, Optional[List[Primitive]]]:
    if interaction.activation_name_to_keep == "vocab_bias":
        return {
            "primitive": _filter_by_scalar_default(bias_primitives, only_default_scalars),
            "special_primitive": None,
        }

    candidates = _filter_by_scalar_default(matrix_primitives, only_default_scalars)
    token_dim = is_token_dim_activation(
        interaction.activation_name_to_keep, converted_mlp
    )
    if token_dim:
        return {"primitive": candidates, "special_primitive": candidates.copy()}
    
    return {"primitive": candidates, "special_primitive": None}
