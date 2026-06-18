from __future__ import annotations

from typing import Any, Tuple, Union

from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput

MlpPrimitiveKey = Union[str, Tuple[str, Any]]


def map_primitive_to_names(primitive: MlpPrimitiveKey) -> Tuple[str, str]:
    """Map an MLP primitive identifier to (var_prefix, op_name) D-RASP tokens."""
    match primitive:
        case "unknown":
            return "new", "element_wise_op"
        case "no_op" | "noop":
            return "", ""
        case "erase":
            return "erased", "erase"
        case "harden":
            return "hardened", "harden"
        case ("sharpen", _):
            return "sharpened", "sharpen"
        case ("exists", idx):
            return f"is_{idx}_exists", f"is_{idx}_exists"
        case ("forall", _):
            return "is_pure", "is_pure"
        case ("zeroone", _, _):
            return "is_01_balance", "is_01_balance"
        case ("equal", token1, token2):
            return f"diff_{token1}{token2}", f"diff_{token1}{token2}"
        case "combine":
            return "_x_", "Cartesian_product"
        case _:
            raise RuntimeError(f"Unsupported MLP primitive: {primitive!r}")


def mlp_search_output_to_primitive_key(
    search_output: PrimitiveSearchOutput | None,
) -> MlpPrimitiveKey:
    if search_output is None or search_output.best_primitive is None:
        return "unknown"

    prim = search_output.best_primitive
    name = prim.name

    if name == "noop":
        return "noop"
    if name == "keepone":
        return ("keep_one", getattr(prim, "keep_n", 0))
    if name == "sharpen":
        return ("sharpen", getattr(prim, "pow", None))
    if name == "exists":
        return ("exists", getattr(prim, "idx", 0))
    if name == "forall":
        return ("forall", getattr(prim, "threshold", None))
    if name == "zeroone":
        return ("zeroone", getattr(prim, "pow", None), getattr(prim, "center", None))
    if name == "equal":
        indices = getattr(prim, "indices", [])
        if len(indices) >= 2:
            return ("equal", indices[0], indices[1])
        return ("equal", 0, 1)

    return name
