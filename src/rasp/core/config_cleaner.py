from __future__ import annotations

import copy
import re
from typing import Any, Dict, Tuple

import torch

from primitives_att.utilities.att_primitive_dataclasses import (
    AbstractPrimitive,
    AttentionInteraction,
    LogitsInteraction,
)

PrimitivesMap = Dict[Any, Any]


def _is_uniform_zero(abstract: AbstractPrimitive) -> bool:
    if abstract.primitive is None:
        return False
    prim_name = getattr(abstract.primitive, "info", None)
    if prim_name is None or prim_name.name != "zero":
        return False
    if abstract.special_primitive is None:
        return True
    special_name = getattr(abstract.special_primitive, "info", None)
    return special_name is not None and special_name.name == "zero"


def _matrix_is_all_zero(abstract: AbstractPrimitive) -> bool:
    if abstract.replacement_matrix is None:
        return False
    matrix = abstract.replacement_matrix.get_matrix()
    return bool((matrix == 0).all())


def _remove_qk_interaction(
    config: Dict[Any, Any],
    interaction_map: PrimitivesMap,
    layer: int,
    head: int,
    act_q: str,
    act_k: str,
) -> Tuple[Dict[Any, Any], PrimitivesMap]:
    new_config = copy.deepcopy(config)
    new_map = copy.deepcopy(interaction_map)

    new_config[layer]["qk"][head] = [
        pair
        for pair in new_config[layer]["qk"][head]
        if not (pair[0] == act_q and pair[1] == act_k)
    ]

    to_remove = [
        interaction
        for interaction in new_map[layer][head]
        if isinstance(interaction, AttentionInteraction)
        and interaction.activation_name_to_keep_q == act_q
        and interaction.activation_name_to_keep_k == act_k
    ]
    for interaction in to_remove:
        del new_map[layer][head][interaction]

    return new_config, new_map


def _remove_k_interaction(
    config: Dict[Any, Any],
    interaction_map: PrimitivesMap,
    layer: int,
    head: int,
    act_k: str,
) -> Tuple[Dict[Any, Any], PrimitivesMap]:
    new_config = copy.deepcopy(config)
    new_map = copy.deepcopy(interaction_map)

    new_config[layer]["k"][head] = [
        k for k in new_config[layer]["k"][head] if k != act_k
    ]

    to_remove = [
        interaction
        for interaction in new_map[layer][head]
        if isinstance(interaction, AttentionInteraction)
        and interaction.activation_name_to_keep_q is None
        and interaction.activation_name_to_keep_k == act_k
    ]
    for interaction in to_remove:
        del new_map[layer][head][interaction]

    return new_config, new_map


def _remove_lm_head_interaction(
    config: Dict[Any, Any],
    interaction_map: PrimitivesMap,
    act: str,
) -> Tuple[Dict[Any, Any], PrimitivesMap]:
    new_config = copy.deepcopy(config)
    new_map = copy.deepcopy(interaction_map)

    new_config["lm_head"] = [item for item in new_config["lm_head"] if item != act]
    to_remove = [
        interaction
        for interaction in new_map["lm_head"]
        if isinstance(interaction, LogitsInteraction)
        and interaction.activation_name_to_keep == act
    ]
    for interaction in to_remove:
        del new_map["lm_head"][interaction]

    return new_config, new_map


def _collect_used_paths(config: Dict[Any, Any]) -> set[str]:
    all_used_paths: set[str] = set()
    pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"

    for layer in config:
        if layer == "lm_head":
            for act in config["lm_head"]:
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))
            continue

        for head in config[layer]["k"]:
            for act in config[layer]["k"][head]:
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))

        for head in config[layer]["qk"]:
            for act_pair in config[layer]["qk"][head]:
                for act in act_pair:
                    matches = re.findall(pattern, act)
                    for m_i in range(len(matches)):
                        all_used_paths.add("-".join(matches[m_i:]))

    has_unsplitted_mlps = any(
        layer != "lm_head" and f"mlp-{layer}" in all_used_paths for layer in config
    )
    if has_unsplitted_mlps:
        for layer in config:
            if layer == "lm_head":
                continue
            for act in config[layer]["mlp"]:
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))

    return all_used_paths


def remove_unnecessary_primitives(
    config: Dict[Any, Any],
    interaction_map: PrimitivesMap,
) -> Tuple[Dict[Any, Any], PrimitivesMap]:
    """Drop zero/uniform selectors and prune unused v/mlp paths from the pruning config."""
    for layer in interaction_map:
        if not isinstance(layer, int):
            continue
        for head in interaction_map[layer]:
            for interaction, abstract in list(interaction_map[layer][head].items()):
                if not isinstance(interaction, AttentionInteraction):
                    continue
                if _is_uniform_zero(abstract) or _matrix_is_all_zero(abstract):
                    if interaction.activation_name_to_keep_q is None:
                        config, interaction_map = _remove_k_interaction(
                            config,
                            interaction_map,
                            layer,
                            head,
                            interaction.activation_name_to_keep_k,
                        )
                    else:
                        config, interaction_map = _remove_qk_interaction(
                            config,
                            interaction_map,
                            layer,
                            head,
                            interaction.activation_name_to_keep_q,
                            interaction.activation_name_to_keep_k,
                        )

    for interaction, abstract in list(interaction_map["lm_head"].items()):
        if _is_uniform_zero(abstract) or _matrix_is_all_zero(abstract):
            config, interaction_map = _remove_lm_head_interaction(
                config,
                interaction_map,
                interaction.activation_name_to_keep,
            )

    all_used_paths = _collect_used_paths(config)

    for layer in config:
        if layer == "lm_head":
            continue
        for head in config[layer]["v"]:
            config[layer]["v"][head] = [
                act
                for act in config[layer]["v"][head]
                if f"attn_output-{layer}-{head}-{act}" in all_used_paths
            ]
        config[layer]["mlp"] = [
            act
            for act in config[layer]["mlp"]
            if f"mlp-{layer}-{act}" in all_used_paths
            or f"mlp-{layer}" in all_used_paths
        ]

    return config, interaction_map
