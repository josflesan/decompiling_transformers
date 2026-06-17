from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from primitives_att.utilities.att_primitive_dataclasses import (
    AbstractPrimitive,
    AttentionInteraction,
    LogitsInteraction,
)
from rasp.core.config_cleaner import remove_unnecessary_primitives
from rasp.core.line_emitters import (
    emit_predefined_line,
    emit_replacement_line,
    is_replacement_matrix,
)
from rasp.utilities.mlp_name_mapper import (
    map_primitive_to_names,
    mlp_search_output_to_primitive_key,
)
from rasp.utilities.rasp_dataclasses import CircledLabel, DecompilationResult
from rasp.utilities.rasp_utils import convert_keys_to_int

PrimitivesMap = Dict[Any, Any]


class DRASPConverter:
    def __init__(
        self,
        config: Dict[Any, Any],
        split_mlp: bool,
        converted_mlp: Dict[str, Any],
        interaction_map: Optional[PrimitivesMap] = None,
        convert_to_primitives: bool = True,
        show_logits_for_unconverted_mlp: bool = False,
    ):
        self.config = config
        self.split_mlp = split_mlp
        self.converted_mlp = converted_mlp
        self.interaction_map = interaction_map
        self.convert_to_primitives = convert_to_primitives
        self.show_logits_for_unconverted_mlp = show_logits_for_unconverted_mlp

    def _find_qk_abstract(
        self, layer: int, head: int, q: str, k: str
    ) -> AbstractPrimitive:
        assert self.interaction_map is not None
        for interaction, abstract in self.interaction_map[layer][head].items():
            if (
                isinstance(interaction, AttentionInteraction)
                and interaction.activation_name_to_keep_q == q
                and interaction.activation_name_to_keep_k == k
            ):
                return abstract
        raise AssertionError(f"Missing QK interaction for layer={layer} head={head} q={q} k={k}")

    def _find_k_abstract(self, layer: int, head: int, k: str) -> AbstractPrimitive:
        assert self.interaction_map is not None
        for interaction, abstract in self.interaction_map[layer][head].items():
            if (
                isinstance(interaction, AttentionInteraction)
                and interaction.activation_name_to_keep_q is None
                and interaction.activation_name_to_keep_k == k
            ):
                return abstract
        raise AssertionError(f"Missing K interaction for layer={layer} head={head} k={k}")

    def _find_lm_abstract(self, activation: str) -> AbstractPrimitive:
        assert self.interaction_map is not None
        for interaction, abstract in self.interaction_map["lm_head"].items():
            if (
                isinstance(interaction, LogitsInteraction)
                and interaction.activation_name_to_keep == activation
            ):
                return abstract
        raise AssertionError(f"Missing lm_head interaction for activation={activation}")

    def _emit_attention_line(
        self,
        select_name: str,
        abstract: AbstractPrimitive,
        var_mapping: Dict[str, str],
        layer_idx: str,
        head_idx: str,
        q: Optional[str],
        k: str,
        count_heatmaps: int,
        circled_labels: Dict[str, CircledLabel],
    ) -> Tuple[str, int, bool]:
        if abstract.primitive is not None:
            if q is None:
                _, code_line = emit_predefined_line(
                    select_name,
                    abstract,
                    "bias",
                    var_mapping[k],
                    None,
                    layer_idx,
                    head_idx,
                )
            else:
                _, code_line = emit_predefined_line(
                    select_name,
                    abstract,
                    var_mapping[q],
                    var_mapping[k],
                    None,
                    layer_idx,
                    head_idx,
                )
            return f"{select_name} = {code_line}", count_heatmaps + 1, False

        if is_replacement_matrix(abstract):
            if q is None:
                _, code_line, is_after_mlp, op_letter = emit_replacement_line(
                    select_name,
                    count_heatmaps,
                    "bias",
                    var_mapping[k],
                    None,
                    layer_idx,
                    head_idx,
                    "bias",
                    k,
                    None,
                    self.converted_mlp,
                    self.show_logits_for_unconverted_mlp,
                )
                circled_labels[op_letter] = CircledLabel(
                    label=op_letter,
                    code_var=select_name,
                    layer=int(layer_idx),
                    head=int(head_idx),
                    k_path=k,
                )
            else:
                _, code_line, is_after_mlp, op_letter = emit_replacement_line(
                    select_name,
                    count_heatmaps,
                    var_mapping[q],
                    var_mapping[k],
                    None,
                    layer_idx,
                    head_idx,
                    q,
                    k,
                    None,
                    self.converted_mlp,
                    self.show_logits_for_unconverted_mlp,
                )
                circled_labels[op_letter] = CircledLabel(
                    label=op_letter,
                    code_var=select_name,
                    layer=int(layer_idx),
                    head=int(head_idx),
                    q_path=q,
                    k_path=k,
                )
            count_delta = 0 if is_after_mlp else 1
            return f"{select_name} = {code_line}", count_heatmaps + count_delta, is_after_mlp

        if q is None:
            primitive_name = abstract.name or "select"
            line = (
                f"{select_name} = {primitive_name}(k={var_mapping[k]})"
                f"\t  # layer {layer_idx} head {head_idx}"
            )
        else:
            primitive_name = abstract.name or "select"
            line = (
                f"{select_name} = {primitive_name}(q={var_mapping[q]}, k={var_mapping[k]})"
                f"\t  # layer {layer_idx} head {head_idx}"
            )
        return line, count_heatmaps + 1, False

    def _emit_logits_line(
        self,
        var_name: str,
        abstract: AbstractPrimitive,
        var_mapping: Dict[str, str],
        inp: str,
        count_heatmaps: int,
        circled_labels: Dict[str, CircledLabel],
    ) -> Tuple[str, int, bool]:
        if abstract.primitive is not None:
            _, code_line = emit_predefined_line(
                var_name,
                abstract,
                None,
                None,
                var_mapping[inp] if inp != "vocab_bias" else "bias",
                None,
                None,
            )
            return f"{var_name} = {code_line}", count_heatmaps + 1, False

        if is_replacement_matrix(abstract):
            inp_var = "bias" if inp == "vocab_bias" else var_mapping[inp]
            _, code_line, is_after_mlp, op_letter = emit_replacement_line(
                var_name,
                count_heatmaps,
                None,
                None,
                inp_var,
                None,
                None,
                None,
                None,
                inp,
                self.converted_mlp,
                self.show_logits_for_unconverted_mlp,
            )
            circled_labels[op_letter] = CircledLabel(
                label=op_letter,
                code_var=var_name,
                inp_path=inp,
            )
            count_delta = 0 if is_after_mlp else 1
            return f"{var_name} = {code_line}", count_heatmaps + count_delta, is_after_mlp

        primitive_name = abstract.name or "not_converted_proj_to_vocab"
        if inp == "vocab_bias":
            return f"{var_name} = {primitive_name}(bias)", count_heatmaps + 1, False
        return f"{var_name} = {primitive_name}({var_mapping[inp]})", count_heatmaps + 1, False

    def convert(self) -> DecompilationResult:
        config = convert_keys_to_int(self.config)
        interaction_map = self.interaction_map

        if self.convert_to_primitives:
            assert interaction_map is not None
            config, interaction_map = remove_unnecessary_primitives(config, interaction_map)
            self.interaction_map = interaction_map

        code: list[str] = []
        selector_to_config: Dict[str, Any] = {}
        circled_labels: Dict[str, CircledLabel] = {}
        counter = {"s": 1, "a": 1, "m": 1}
        var_mapping = {"wpe": "pos", "wte": "token"}
        count_heatmaps = 0

        for layer_idx in range(len(config) - 1):
            layer_idx_str = str(layer_idx)
            for head_idx in config[layer_idx]["v"]:
                head_idx_str = str(head_idx)
                select_names: list[str] = []

                for prod in (
                    config[layer_idx]["qk"][head_idx]
                    + config[layer_idx]["k"][head_idx]
                ):
                    select_name = "s" + str(counter["s"])

                    if isinstance(prod, (tuple, list)):
                        q, k = prod
                        if self.convert_to_primitives:
                            abstract = self._find_qk_abstract(layer_idx, head_idx, q, k)
                            line, count_heatmaps, _ = self._emit_attention_line(
                                select_name,
                                abstract,
                                var_mapping,
                                layer_idx_str,
                                head_idx_str,
                                q,
                                k,
                                count_heatmaps,
                                circled_labels,
                            )
                            code.append(line)
                        else:
                            code.append(
                                f"{select_name} = select(q={var_mapping[q]}, k={var_mapping[k]})"
                                f"\t  # layer {layer_idx_str} head {head_idx_str}"
                            )
                    elif isinstance(prod, str):
                        k = prod
                        if self.convert_to_primitives:
                            abstract = self._find_k_abstract(layer_idx, head_idx, k)
                            line, count_heatmaps, _ = self._emit_attention_line(
                                select_name,
                                abstract,
                                var_mapping,
                                layer_idx_str,
                                head_idx_str,
                                None,
                                k,
                                count_heatmaps,
                                circled_labels,
                            )
                            code.append(line)
                        else:
                            code.append(
                                f"{select_name} = select(k={var_mapping[k]})"
                                f"\t  # layer {layer_idx_str} head {head_idx_str}"
                            )
                    else:
                        raise RuntimeError(prod)

                    selector_to_config[select_name] = (
                        layer_idx,
                        head_idx,
                        prod,
                    )
                    select_names.append(select_name)
                    counter["s"] += 1

                select_expr = "[]" if len(select_names) == 0 else "+".join(select_names)
                for v in config[layer_idx]["v"][head_idx]:
                    attn_out_name = "a" + str(counter["a"])
                    code.append(
                        f"{attn_out_name} = aggregate(s={select_expr}, v={var_mapping[v]})"
                        f"\t  # layer {layer_idx_str} head {head_idx_str}"
                    )
                    var_mapping[f"attn_output-{layer_idx_str}-{head_idx_str}-{v}"] = (
                        attn_out_name
                    )
                    counter["a"] += 1

            if config[layer_idx]["mlp"]:
                if not self.split_mlp:
                    path = f"mlp-{layer_idx_str}"
                    if path in self.converted_mlp:
                        primitive_key = mlp_search_output_to_primitive_key(
                            self.converted_mlp[path]
                        )
                        if primitive_key[0] == "keep_one":
                            mlp_inp = config[layer_idx]["mlp"][primitive_key[1]]
                            var_mapping[f"mlp-{layer_idx_str}"] = var_mapping[mlp_inp]
                        else:
                            connect_symbol, op_name = map_primitive_to_names(primitive_key)
                            mlp_inputs = [
                                var_mapping[x] for x in config[layer_idx]["mlp"]
                            ]
                            code.append(
                                f"{connect_symbol.join(mlp_inputs)} = {op_name}({', '.join(mlp_inputs)})"
                                f"\t  # layer {layer_idx_str} mlp"
                            )
                            var_mapping[f"mlp-{layer_idx_str}"] = connect_symbol.join(
                                mlp_inputs
                            )
                    else:
                        mlp_out_name = "m" + str(counter["m"])
                        mlp_inputs = "+".join(
                            var_mapping[x] for x in config[layer_idx]["mlp"]
                        )
                        code.append(
                            f"{mlp_out_name} = mlp({mlp_inputs})"
                            f"\t  # layer {layer_idx_str} mlp"
                        )
                        var_mapping[f"mlp-{layer_idx_str}"] = mlp_out_name
                        counter["m"] += 1
                else:
                    for inp in config[layer_idx]["mlp"]:
                        path = f"mlp-{layer_idx_str}-{inp}"
                        primitive_key = mlp_search_output_to_primitive_key(
                            self.converted_mlp.get(path)
                        )
                        if primitive_key in ("noop", "no_op"):
                            var_mapping[f"mlp-{layer_idx_str}-{inp}"] = var_mapping[inp]
                            continue
                        var_prefix, op_name = map_primitive_to_names(primitive_key)
                        code.append(
                            f"{var_prefix}_{var_mapping[inp]} = {op_name}({var_mapping[inp]})"
                            f"\t  # layer {layer_idx_str} mlp"
                        )
                        var_mapping[f"mlp-{layer_idx_str}-{inp}"] = (
                            f"{var_prefix}_{var_mapping[inp]}"
                        )

        distribution_to_config: Dict[str, Any] = {}
        i = 0
        for i, inp in enumerate(config["lm_head"], start=1):
            var_name = f"logits{i}"
            if self.convert_to_primitives:
                abstract = self._find_lm_abstract(inp)
                line, count_heatmaps, _ = self._emit_logits_line(
                    var_name,
                    abstract,
                    var_mapping,
                    inp,
                    count_heatmaps,
                    circled_labels,
                )
                code.append(line)
            else:
                code.append(f"{var_name} = proj_to_vocab({var_mapping[inp]})")
            distribution_to_config[var_name] = inp

        if self.convert_to_primitives:
            vocab_bias = [
                interaction
                for interaction in self.interaction_map["lm_head"]
                if isinstance(interaction, LogitsInteraction)
                and interaction.activation_name_to_keep == "vocab_bias"
            ]
            if len(vocab_bias) == 1:
                var_name = f"logits{i + 1}"
                abstract = self.interaction_map["lm_head"][vocab_bias[0]]
                line, count_heatmaps, _ = self._emit_logits_line(
                    var_name,
                    abstract,
                    var_mapping,
                    "vocab_bias",
                    count_heatmaps,
                    circled_labels,
                )
                code.append(line)
                distribution_to_config[var_name] = "vocab_bias"
        else:
            var_name = f"logits{i + 1}"
            code.append(f"{var_name} = proj_to_vocab(bias)")
            distribution_to_config[var_name] = "vocab_bias"

        logits_expr = "+\n\t\t\t".join(distribution_to_config.keys())
        code.append(f"prediction = softmax({logits_expr})")

        selector_to_config.update(distribution_to_config)
        return DecompilationResult(
            lines=code,
            selector_to_config=selector_to_config,
            var_mapping=var_mapping,
            circled_labels=circled_labels,
        )
