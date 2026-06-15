"""
Forward-pass hooks that replace attention QK interactions and unembedding paths with primitives.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from data.CustomTokenizer import CustomTokenizer
from primitives_att.utilities.product_tracing import (
    get_product_for_one_side_for_head,
    get_product_for_one_side_for_unembed,
)
from pruning.core.hooks import GPT2QKHooks, gpt2_causal_attention_mask
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput

ConvertedMlp = Dict[str, PrimitiveSearchOutput]
PrimitivesMap = Dict[Any, Any]

SPECIAL_TOKEN_IDS = ("sep_token_id", "bos_token_id", "eos_token_id", "pad_token_id")


def _get_wte_wpe_inputs(hooked_model: GPT2QKHooks) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = hooked_model.wte_inputs.squeeze(0)
    position_ids = hooked_model.wpe_inputs.squeeze(0)
    
    return input_ids, position_ids


def _get_scaling_factor(interaction_primitive: Any) -> float | str:
    scaling = getattr(interaction_primitive, "scaling_factor", None)
    if scaling is None:
        scaling = getattr(interaction_primitive, "scaling_factor_primitive", 1.0)
    
    return scaling if scaling is not None else 1.0


def _apply_scaling(primitive_matrix: torch.Tensor, scaling: float | str) -> torch.Tensor:
    if isinstance(scaling, str) and scaling == "inf":
        mask_value = torch.finfo(primitive_matrix.dtype).max
        return torch.where(primitive_matrix > 0, mask_value, torch.zeros_like(primitive_matrix))
    
    return scaling * primitive_matrix


def _build_primitive_matrix(
    interaction_primitive: Any,
    indep_prod: torch.Tensor,
    activation_name_q: Optional[str],
    tokenizer: CustomTokenizer,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if interaction_primitive.primitive is not None:
        special_tokens = [getattr(tokenizer, attr) for attr in SPECIAL_TOKEN_IDS]
        if activation_name_q is not None:
            primitive_matrix = interaction_primitive.primitive.construct(
                indep_prod.shape[0], indep_prod.shape[1], tokenizer
            ).to(dtype=dtype, device=device)
            if interaction_primitive.special_primitive is not None:
                special_matrix = interaction_primitive.special_primitive.construct(
                    indep_prod.shape[0], indep_prod.shape[1], tokenizer
                ).to(dtype=dtype, device=device)
                primitive_matrix[special_tokens, :] = special_matrix[special_tokens, :]
        else:
            primitive_matrix = interaction_primitive.primitive.construct(
                None, indep_prod.shape[1], tokenizer
            ).to(dtype=dtype, device=device)
            assert interaction_primitive.special_primitive is None
        return _apply_scaling(primitive_matrix, _get_scaling_factor(interaction_primitive))

    if interaction_primitive.replacement_matrix is not None:
        return interaction_primitive.replacement_matrix.get_matrix().to(dtype=dtype, device=device)

    raise RuntimeError("AbstractPrimitive has neither primitive nor replacement_matrix set")


def _compute_side_products(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: ConvertedMlp,
    layer: int,
    head: int,
    activation_name_q: Optional[str],
    activation_name_k: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    qk_types = ["q", "k"] if activation_name_q is not None else ["k"]
    activation_names = {
        "q": activation_name_q,
        "k": activation_name_k,
    }

    products = {
        qk_type: get_product_for_one_side_for_head(
            hooked_model,
            oa_vecs,
            converted_mlp,
            layer,
            head,
            qk_type,
            activation_names[qk_type],
            input_ids,
            position_ids,
        )
        for qk_type in qk_types
    }

    act_soft: dict[str, torch.Tensor] = {"k": products["k"][0]}
    if activation_name_q is not None:
        act_soft["q"] = products["q"][0]
        indep_prod = products["q"][1] @ products["k"][1].transpose(-1, -2)
    else:
        alpha = oa_vecs.q_bias_term.data[
            oa_vecs.to_q_bias[(layer, head, activation_name_k)]
        ].unsqueeze(0)
        indep_prod = alpha @ products["k"][1].transpose(-1, -2)

    return act_soft, indep_prod


def _interaction_logits(
    act_soft: dict[str, torch.Tensor],
    indep_prod: torch.Tensor,
    primitive_matrix: Optional[torch.Tensor],
    activation_name_q: Optional[str],
) -> torch.Tensor:
    matrix = indep_prod if primitive_matrix is None else primitive_matrix
    if activation_name_q is not None:
        logits = act_soft["q"] @ matrix @ act_soft["k"].transpose(-1, -2)
    else:
        logits = matrix.squeeze().unsqueeze(0) @ act_soft["k"].transpose(-1, -2)

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    return logits


def attention_primitive_forward(
    self: GPT2QKHooks,
    module: torch.nn.Module,
    layer: Optional[int] = None,
    primitives: Optional[PrimitivesMap] = None,
    tokenizer: Optional[CustomTokenizer] = None,
    oa_vecs: Optional[OptimalQueryBiasVectors] = None,
    converted_mlp: Optional[ConvertedMlp] = None,
):
    assert self.current_layer == layer
    assert primitives is not None
    assert tokenizer is not None
    assert oa_vecs is not None
    assert converted_mlp is not None
    assert hasattr(self, "activation_name_to_keep") and self.activation_name_to_keep is not None

    num_heads = module.num_heads
    head_dim = module.head_dim
    split_size = module.split_size
    batch_size, seq_len, d_model = self.activations["wte"].size()
    device = self.activations["wte"].device
    input_ids, position_ids = _get_wte_wpe_inputs(self)

    input_activations = []
    for head in range(num_heads):
        if self.activation_name_to_keep[head] is not None:
            activation_name = self.activation_name_to_keep[head]
            input_act = self.activations[activation_name]
            ln_var = self.oa_vecs.ln_var[
                self.oa_vecs.to_ln_idx[(layer, "v", head, activation_name)]
            ].exp()
            input_act = self._linear_layer_norm(
                self.ln_1[layer], input_act, ln_var, bias=False
            )
        else:
            input_act = torch.zeros_like(self.activations["wte"])
        input_activations.append(input_act)

    input_activations = torch.stack(input_activations)
    output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ module.c_attn.weight[
        :, split_size * 2 :
    ].view(d_model, num_heads, head_dim).transpose(0, 1)
    value_states = (
        output_activations.transpose(0, 1)
        .contiguous()
        .view(batch_size, seq_len, num_heads, head_dim)
        .transpose(1, 2)
    )

    attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
    for head in range(num_heads):
        for interaction, interaction_primitive in primitives[layer][head].items():
            activation_name_q = interaction.activation_name_to_keep_q
            activation_name_k = interaction.activation_name_to_keep_k
            act_soft, indep_prod = _compute_side_products(
                self,
                oa_vecs,
                converted_mlp,
                layer,
                head,
                activation_name_q,
                activation_name_k,
                input_ids,
                position_ids,
            )

            if interaction_primitive is None or (
                interaction_primitive.primitive is None
                and interaction_primitive.replacement_matrix is None
            ):
                term = _interaction_logits(act_soft, indep_prod, None, activation_name_q)
            else:
                primitive_matrix = _build_primitive_matrix(
                    interaction_primitive,
                    indep_prod,
                    activation_name_q,
                    tokenizer,
                    attn_weights.dtype,
                    attn_weights.device,
                )
                term = _interaction_logits(
                    act_soft, indep_prod, primitive_matrix, activation_name_q
                )

            attn_weights[:, head, :, :] += term

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value_states.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    causal_mask = gpt2_causal_attention_mask(module, seq_len, device)
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(value_states.dtype)
    self.activations[f"attn_weights-{layer}"] = attn_weights.detach()

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)

    attention_by_head = attn_output.contiguous().view(batch_size * seq_len, num_heads, head_dim)
    attention_by_head_output = attention_by_head.transpose(0, 1) @ module.c_proj.weight.view(
        num_heads, head_dim, d_model
    )
    attention_by_head_output = attention_by_head_output.view(
        num_heads, batch_size, seq_len, d_model
    ).unbind(dim=0)

    for head, attn in enumerate(attention_by_head_output):
        if self.activation_name_to_keep[head] is not None:
            self.activations[
                f"attn_output-{layer}-{head}-{self.activation_name_to_keep[head]}"
            ] = attn

    return None


def lm_head_primitive_hook(
    module: torch.nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
    hooked_model: GPT2QKHooks,
    primitives: Optional[PrimitivesMap] = None,
    tokenizer: Optional[CustomTokenizer] = None,
    oa_vecs: Optional[OptimalQueryBiasVectors] = None,
    converted_mlp: Optional[ConvertedMlp] = None,
):
    assert primitives is not None
    assert tokenizer is not None
    assert oa_vecs is not None
    assert converted_mlp is not None

    logits_output = None
    input_ids = hooked_model.input_ids.to(hooked_model.device)
    position_ids = hooked_model.position_ids.to(hooked_model.device)

    for interaction, primitive in primitives["lm_head"].items():
        activation_name = interaction.activation_name_to_keep

        if primitive is None or (
            primitive.primitive is None and primitive.replacement_matrix is None
        ):
            dep_prod, indep_prod = get_product_for_one_side_for_unembed(
                hooked_model,
                oa_vecs,
                converted_mlp,
                activation_name,
                input_ids,
                position_ids,
            )
            if activation_name == "vocab_bias":
                prod = indep_prod
            else:
                prod = dep_prod @ indep_prod
        elif activation_name == "vocab_bias":
            if primitive.replacement_matrix is not None:
                prod = primitive.replacement_matrix.get_matrix()
            else:
                primitive_matrix = primitive.primitive.construct(
                    None, module.weight.shape[0], tokenizer
                ).to(dtype=module.weight.dtype, device=module.weight.device)
                assert primitive.special_primitive is None
                prod = _apply_scaling(primitive_matrix, _get_scaling_factor(primitive))
            prod = prod.unsqueeze(0)
        else:
            dep_prod, indep_prod = get_product_for_one_side_for_unembed(
                hooked_model,
                oa_vecs,
                converted_mlp,
                activation_name,
                input_ids,
                position_ids,
            )
            if primitive.primitive is not None:
                primitive_matrix = primitive.primitive.construct(
                    indep_prod.shape[0], indep_prod.shape[1], tokenizer
                ).to(dtype=indep_prod.dtype, device=indep_prod.device)
                special_tokens = [getattr(tokenizer, attr) for attr in SPECIAL_TOKEN_IDS]
                if primitive.special_primitive is not None:
                    special_matrix = primitive.special_primitive.construct(
                        indep_prod.shape[0], indep_prod.shape[1], tokenizer
                    ).to(dtype=primitive_matrix.dtype, device=primitive_matrix.device)
                    primitive_matrix[special_tokens, :] = special_matrix[special_tokens, :]
                primitive_matrix = _apply_scaling(
                    primitive_matrix, _get_scaling_factor(primitive)
                )
            else:
                primitive_matrix = primitive.replacement_matrix.get_matrix()
            prod = dep_prod @ primitive_matrix

        if prod.dim() == 2:
            prod = prod.unsqueeze(0)

        logits_output = prod if logits_output is None else logits_output + prod

    return logits_output
