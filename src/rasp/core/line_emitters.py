from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from primitives_att.utilities.att_primitive_dataclasses import AbstractPrimitive


def _rasp_op(primitive: Any) -> str:
    return primitive.info.rasp_op


def emit_predefined_line(
    select_name: str,
    abstract: AbstractPrimitive,
    var_mapping_q: Optional[str],
    var_mapping_k: Optional[str],
    var_mapping_inp: Optional[str],
    layer_idx: Optional[str],
    head_idx: Optional[str],
) -> Tuple[str, str]:
    assert abstract.primitive is not None
    assert abstract.scaling_factor is not None

    if var_mapping_q is None and var_mapping_k is None and var_mapping_inp == "bias":
        code_line = f"project(op=({_rasp_op(abstract.primitive)})"
    elif var_mapping_q is None and var_mapping_k is None:
        code_line = (
            f"project(inp={var_mapping_inp}, op=({_rasp_op(abstract.primitive)})"
        )
    elif var_mapping_q == "bias":
        code_line = f"select(k={var_mapping_k}, op=({_rasp_op(abstract.primitive)})"
    else:
        code_line = (
            f"select(q={var_mapping_q}, k={var_mapping_k}, "
            f"op=({_rasp_op(abstract.primitive)})"
        )

    if (
        abstract.special_primitive is not None
        and abstract.special_primitive is not abstract.primitive
    ):
        code_line += f",\n\t\tspecial_op=({_rasp_op(abstract.special_primitive)})"
    code_line += ")"

    if (
        var_mapping_q is not None
        and var_mapping_k is not None
        and var_mapping_q != "bias"
    ):
        code_line += f"\t# layer {layer_idx} head {head_idx}"

    return select_name, code_line


def emit_replacement_line(
    select_name: str,
    count_heatmaps: int,
    var_mapping_q: Optional[str],
    var_mapping_k: Optional[str],
    var_mapping_inp: Optional[str],
    layer_idx: Optional[str],
    head_idx: Optional[str],
    q: Optional[str],
    k: Optional[str],
    inp: Optional[str],
    converted_mlp: Dict[str, Any],
    show_logits_for_unconverted_mlp: bool,
) -> Tuple[str, str, bool, str]:
    is_after_mlp = False
    op_letter = chr(count_heatmaps + ord("a"))

    if var_mapping_q is None and var_mapping_k is None and var_mapping_inp == "bias":
        code_line = f"project(op=\\circled{{{op_letter}}})"
    elif var_mapping_q is None and var_mapping_k is None:
        if (
            inp is not None
            and "mlp" in inp
            and inp[inp.find("mlp") :] not in converted_mlp
            and not show_logits_for_unconverted_mlp
        ):
            code_line = f"project(inp={var_mapping_inp}, op=(inp==out))"
            is_after_mlp = True
        else:
            code_line = f"project(inp={var_mapping_inp}, op=\\circled{{{op_letter}}})"
    elif var_mapping_q == "bias":
        code_line = f"select(k={var_mapping_k}, op=\\circled{{{op_letter}}})"
    else:
        q_unconverted = (
            q is not None
            and "mlp" in q
            and q[q.find("mlp") :] not in converted_mlp
        )
        k_unconverted = (
            k is not None
            and "mlp" in k
            and k[k.find("mlp") :] not in converted_mlp
        )
        if (q_unconverted or k_unconverted) and not show_logits_for_unconverted_mlp:
            code_line = (
                f"select(q={var_mapping_q}, k={var_mapping_k}, op=(q==k))"
            )
            is_after_mlp = True
        else:
            code_line = (
                f"select(q={var_mapping_q}, k={var_mapping_k}, "
                f"op=\\circled{{{op_letter}}})"
            )

    if (
        var_mapping_q is not None
        and var_mapping_k is not None
        and var_mapping_q != "bias"
    ):
        code_line += f"\t# layer {layer_idx} head {head_idx}"

    return select_name, code_line, is_after_mlp, op_letter


def is_replacement_matrix(abstract: AbstractPrimitive) -> bool:
    if abstract.replacement_matrix is not None:
        return True
    return abstract.name is not None and "projection[" in abstract.name
