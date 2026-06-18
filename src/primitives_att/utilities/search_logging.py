"""Helpers for serializing attention primitive search state to metrics.jsonl."""

from __future__ import annotations

from typing import Any, Dict, Optional

from primitives_att.utilities.att_primitive_dataclasses import (
    AbstractPrimitive,
    AttentionInteraction,
    LogitsInteraction,
    PrimitiveEval,
)

PrimitivesMap = Dict[Any, Any]


def is_successfully_converted(abstract: Optional[AbstractPrimitive]) -> bool:
    if abstract is None:
        return False
    if abstract.primitive is not None:
        return True
    if abstract.replacement_matrix is not None:
        return bool(abstract.replacement_matrix.do_round)
    return False


def interaction_id(
    interaction: AttentionInteraction | LogitsInteraction,
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> str:
    if isinstance(interaction, LogitsInteraction):
        return f"lm_head-{interaction.activation_name_to_keep}"
    if interaction.activation_name_to_keep_q is None:
        return f"L{layer}-H{head}-bias-{interaction.activation_name_to_keep_k}"
    return (
        f"L{layer}-H{head}-"
        f"{interaction.activation_name_to_keep_q}-"
        f"{interaction.activation_name_to_keep_k}"
    )


def primitive_label(primitive: Any) -> Optional[str]:
    if primitive is None:
        return None
    return str(primitive)


def abstract_primitive_fields(abstract: Optional[AbstractPrimitive]) -> Optional[Dict[str, Any]]:
    if abstract is None:
        return None
    
    return {
        "name": abstract.name,
        "primitive": primitive_label(abstract.primitive),
        "special_primitive": primitive_label(abstract.special_primitive),
        "scaling_factor": abstract.scaling_factor,
        "replacement_matrix": abstract.replacement_matrix is not None,
        "rounded": (
            abstract.replacement_matrix.do_round
            if abstract.replacement_matrix is not None
            else None
        ),
    }


def count_interactions(interaction_map: PrimitivesMap) -> int:
    total = len(interaction_map["lm_head"])
    for layer in interaction_map:
        if isinstance(layer, int):
            for head in interaction_map[layer]:
                total += len(interaction_map[layer][head])
    return total


def count_converted(interaction_map: PrimitivesMap) -> int:
    converted = sum(
        1 for p in interaction_map["lm_head"].values() if is_successfully_converted(p)
    )
    for layer in interaction_map:
        if isinstance(layer, int):
            for head in interaction_map[layer]:
                converted += sum(
                    1
                    for p in interaction_map[layer][head].values()
                    if is_successfully_converted(p)
                )
    return converted


def interaction_eval(interaction_map: PrimitivesMap) -> PrimitiveEval:
    converted = count_converted(interaction_map)
    total = count_interactions(interaction_map)
    return PrimitiveEval(
        zero_parameters=[converted],
        total_parameters=[total],
        is_fully_replaced=[converted == total and total > 0],
    )


def head_interaction_state(
    interaction_map: PrimitivesMap,
    layer: int,
    head: int,
) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        interaction_id(interaction, layer, head): abstract_primitive_fields(abstract)
        for interaction, abstract in interaction_map[layer][head].items()
    }
