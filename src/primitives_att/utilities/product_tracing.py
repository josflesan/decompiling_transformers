"""
Trace dependent and independent path products for attention and unembedding primitive hooks.
"""

import re
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from primitives_mlp.utilities.parameter_getters import (
    get_attn_weights_for_head,
    get_ln_matrix_for_node,
    get_ov_for_head,
    get_qk_for_head,
)
from pruning.core.hooks import GPT2QKHooks
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors


def _uses_split_mlps(oa_vecs: OptimalQueryBiasVectors) -> bool:
    return hasattr(oa_vecs, "mlps")


@torch.no_grad()
def get_product_for_one_side(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = hooked_model.model
    d_model = model.config.hidden_size

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path)
    split_nodes = split_nodes[::-1].copy()
    dep_prod = None
    indep_prod = None

    for idx, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            attn = get_attn_weights_for_head(hooked_model, layer, head).squeeze(0)
            dep_prod = attn @ dep_prod

            past_path = "-".join(split_nodes[idx - 1::-1])
            w_ln = get_ln_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
            w_v, w_o = get_ov_for_head(model, layer, head)
            indep_prod = indep_prod @ w_ln @ w_v @ w_o

        elif node == "wte":
            dep_prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            indep_prod = model.transformer.wte.weight.data

        elif node == "wpe":
            dep_prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
            indep_prod = model.transformer.wpe.weight.data

        elif node.startswith("mlp"):
            node_path = "-".join(split_nodes[idx::-1])
            
            if node_path in converted_mlp:
                search_results = converted_mlp[node_path]
                dep_prod = search_results.best_primitive.apply(dep_prod)
                indep_prod = search_results.best_C
            else:
                dep_prod = hooked_model.activations[node_path].squeeze(0)
                indep_prod = torch.eye(d_model, device=model.device)

        else:
            raise RuntimeError(f"Node not recognized: {node}")

    return dep_prod, indep_prod


@torch.no_grad()
def get_product_for_one_side_multi_source(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = hooked_model.model
    d_model = model.config.hidden_size
    config = hooked_model.config

    if path.startswith("mlp"):
        if path in converted_mlp:
            prods = [
                get_product_for_one_side_multi_source(
                    hooked_model, oa_vecs, converted_mlp, mlp_inp, input_ids, position_ids
                )[0]
                for mlp_inp in config[int(path[4:])]["mlp"]
            ]
            search_results = converted_mlp[path]
            dep_prod = search_results.best_primitive.apply(prods)
            return dep_prod, search_results.best_C

        dep_prod = hooked_model.activations[path].squeeze(0)
        indep_prod = torch.eye(d_model, device=model.device)
        return dep_prod, indep_prod

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path)
    dep_prod = None
    indep_prod = None

    for idx, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            attn = get_attn_weights_for_head(hooked_model, layer, head).squeeze(0)
            dep_prod = dep_prod @ attn if dep_prod is not None else attn

            past_path = "-".join(split_nodes[idx + 1:])
            w_ln = get_ln_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
            w_v, w_o = get_ov_for_head(model, layer, head)
            absorbed = w_ln @ w_v @ w_o
            indep_prod = absorbed @ indep_prod if indep_prod is not None else absorbed

        elif node == "wte":
            tok = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            dep_prod = dep_prod @ tok if dep_prod is not None else tok
            wte = model.transformer.wte.weight.data
            indep_prod = wte @ indep_prod if indep_prod is not None else wte

        elif node == "wpe":
            pos = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
            dep_prod = dep_prod @ pos if dep_prod is not None else pos
            wpe = model.transformer.wpe.weight.data
            indep_prod = wpe @ indep_prod if indep_prod is not None else wpe

        elif node.startswith("mlp"):
            dep_p, indep_p = get_product_for_one_side_multi_source(
                hooked_model, oa_vecs, converted_mlp, node, input_ids, position_ids
            )
            dep_prod = dep_prod @ dep_p
            indep_prod = indep_p @ indep_prod

        else:
            raise RuntimeError(f"Node not recognized: {node}")

    return dep_prod, indep_prod


def _get_product_for_one_side(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _uses_split_mlps(oa_vecs):
        return get_product_for_one_side(
            hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids
        )
    return get_product_for_one_side_multi_source(
        hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids
    )


@torch.no_grad()
def get_product_for_one_side_for_head(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    attn_layer_idx: int,
    attn_head_idx: int,
    qk_type: str,
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dep_prod, indep_prod = _get_product_for_one_side(
        hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids
    )
    w_ln = get_ln_matrix_for_node(
        hooked_model.model, oa_vecs, attn_layer_idx, qk_type, attn_head_idx
    )
    w_q_or_k = get_qk_for_head(hooked_model.model, attn_layer_idx, qk_type, attn_head_idx)
    indep_prod = indep_prod @ w_ln @ w_q_or_k
    return dep_prod, indep_prod


@torch.no_grad()
def get_product_for_one_side_for_unembed(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    if path != "vocab_bias":
        dep_prod, indep_prod = _get_product_for_one_side(
            hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids
        )
    else:
        indep_prod = oa_vecs.output_vertex_oa.data[
            oa_vecs.to_out_oa_idx[("lm_head",)]
        ].unsqueeze(0)
        dep_prod = None

    w_ln = get_ln_matrix_for_node(hooked_model.model, oa_vecs, None, "lm_head", None)
    indep_prod = indep_prod @ w_ln @ hooked_model.model.lm_head.weight.data.T
    return dep_prod, indep_prod
