"""
Helpers for saving attention and unembedding primitive heatmaps during hook execution.
"""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data.CustomTokenizer import CustomTokenizer
from primitives_att.utilities.activation_analysis import is_token_dim_activation
from primitives_att.utilities.heatmap_plotting import plot_and_save_primitives_matrices
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from pruning.core.hooks import GPT2QKHooks

ConvertedMlp = Dict[str, PrimitiveSearchOutput]


def is_cartesian_act(
    act_name: str,
    converted_mlp: ConvertedMlp,
    pruning_config: dict,
) -> Tuple[bool, Optional[int]]:
    pattern = r"attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe"
    split_nodes = re.findall(pattern, act_name)
    closest_non_attn = 0
    for node in split_nodes:
        if node.startswith("attn_output"):
            closest_non_attn += 1
        else:
            break
    split_nodes = split_nodes[closest_non_attn:]
    closest_non_attn_node = "-".join(split_nodes)

    if closest_non_attn_node not in converted_mlp:
        return False, None

    mlp_primitive = converted_mlp[closest_non_attn_node].best_primitive
    if mlp_primitive.name != "combine":
        return False, None

    mlp_match = re.match(r"mlp-(\d+)", closest_non_attn_node)
    if not mlp_match:
        return False, None
    mlp_layer = int(mlp_match.group(1))
    acts_incoming = pruning_config[mlp_layer]["mlp"]
    is_cartesian_incoming = [
        is_cartesian_act(act, converted_mlp, pruning_config)[0] for act in acts_incoming
    ]
    is_token_incoming = [
        is_token_dim_activation(act, converted_mlp) for act in acts_incoming
    ]
    
    if all(t or c[0] for t, c in zip(is_token_incoming, is_cartesian_incoming)):
        sum_incoming = sum(
            (1 if token else (c[1] if c[0] else 1))
            for token, c in zip(is_token_incoming, is_cartesian_incoming)
        )
        return True, sum_incoming
    
    return False, None


def vocab_ticks(tokenizer: CustomTokenizer) -> List[str]:
    return [tokenizer.vocab_inv[i] for i in range(len(tokenizer.vocab))]


def example_ticks(tokenizer: CustomTokenizer, input_ids: torch.Tensor) -> List[str]:
    return tokenizer.convert_ids_to_tokens(input_ids.tolist())


def activation_ticks(
    activation_name: str,
    converted_mlp: ConvertedMlp,
    pruning_config: dict,
    tokenizer: CustomTokenizer,
) -> Optional[List[str]]:
    if is_token_dim_activation(activation_name, converted_mlp):
        return vocab_ticks(tokenizer)
    
    is_cartesian, repeats = is_cartesian_act(activation_name, converted_mlp, pruning_config)
    if is_cartesian and repeats is not None:
        return ["-".join(tokens) for tokens in product(vocab_ticks(tokenizer), repeat=repeats)]
    return None


def interaction_save_name(
    activation_q: Optional[str],
    activation_k: str,
) -> str:
    if activation_q is None:
        return f"bias-{activation_k}"
    
    return f"{activation_q}-{activation_k}"


class MatrixSaver:
    def __init__(
        self,
        hooked_model: GPT2QKHooks,
        tokenizer: CustomTokenizer,
        converted_mlp: ConvertedMlp,
    ):
        self.hooked_model = hooked_model
        self.tokenizer = tokenizer
        self.converted_mlp = converted_mlp
        self.pruning_config = hooked_model.config

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.hooked_model, "save_matrices", False))

    def _base_dir(self) -> Path:
        return Path(self.hooked_model.save_matrices_path)

    def save(
        self,
        key: tuple[Any, ...],
        matrix: torch.Tensor,
        relative_path: str,
        ticks_x: Optional[Sequence[str]] = None,
        ticks_y: Optional[Sequence[str]] = None,
        add_causal_mask: bool = False,
    ) -> None:
        if not self.enabled:
            return
        
        saved = getattr(self.hooked_model, "saved_matrices", None)
        if saved is None:
            self.hooked_model.saved_matrices = set()
            saved = self.hooked_model.saved_matrices
        if key in saved:
            return
        
        saved.add(key)

        png_path = self._base_dir() / relative_path
        
        plot_and_save_primitives_matrices(
            matrix,
            png_path,
            ticks_x=ticks_x,
            ticks_y=ticks_y,
            add_causal_mask=add_causal_mask,
        )

    def save_pos_tok_once(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        key = ("attn", "pos-tok")
        saved = getattr(self.hooked_model, "saved_matrices", None)
        if saved is None:
            self.hooked_model.saved_matrices = set()
            saved = self.hooked_model.saved_matrices
        if key in saved:
            return
        saved.add(key)

        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        if position_ids.dim() > 1:
            position_ids = position_ids[0]

        example = example_ticks(self.tokenizer, input_ids)
        vocab = vocab_ticks(self.tokenizer)
        toks = torch.nn.functional.one_hot(
            input_ids.long(), num_classes=self.hooked_model.model.config.vocab_size
        ).float()
        poss = torch.nn.functional.one_hot(
            position_ids.long(),
            num_classes=self.hooked_model.model.config.max_position_embeddings,
        ).float()
        base = "pos-tok"
        
        plot_and_save_primitives_matrices(
            toks.mT, self._base_dir() / f"{base}/tok.png", ticks_x=example, ticks_y=vocab
        )
        plot_and_save_primitives_matrices(
            poss.mT, self._base_dir() / f"{base}/pos.png", ticks_x=example, ticks_y=None
        )

    def attention_ticks(
        self,
        activation_q: Optional[str],
        activation_k: str,
    ) -> tuple[Optional[List[str]], Optional[List[str]]]:
        ticks_y = None if activation_q is None else activation_ticks(
            activation_q, self.converted_mlp, self.pruning_config, self.tokenizer
        )
        ticks_x = activation_ticks(
            activation_k, self.converted_mlp, self.pruning_config, self.tokenizer
        )
        return ticks_x, ticks_y
