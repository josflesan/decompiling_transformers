"""
Search strategies for attention and unembedding primitive replacement.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from primitives_att.core.PrimitiveExecutionEngine import PrimitiveExecutionEngine
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.activation_analysis import (
    expand_grid,
    filter_attention_candidates,
    filter_logits_candidates,
)
from primitives_att.utilities.att_primitive_dataclasses import (
    AbstractPrimitive,
    AttentionInteraction,
    AttPrimitivesConfig,
    LogitsInteraction,
    PrimitiveEval,
)
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape
from primitives_att.utilities.search_logging import (
    abstract_primitive_fields,
    count_converted,
    count_interactions,
    head_interaction_state,
    interaction_id,
    primitive_label,
)
from utilities.metrics_logger import MetricsLogger

PrimitivesMap = Dict[Any, Any]
CandidatePrimitives = Dict[Tuple[PrimitiveDomain, PrimitiveShape], List[Primitive]]


class SearchStrategy(ABC):
    @abstractmethod
    def search(
        self,
        interaction_map: PrimitivesMap,
        acc_match_threshold: float,
    ) -> Tuple[PrimitivesMap, PrimitiveEval]:
        raise NotImplementedError


class GreedySearchStrategy(SearchStrategy):
    DEFAULT_SCALING_FACTORS = [1e4, 1.0, 0.01, 0.1, 10.0, 100.0]
    DEFAULT_ONLY_SCALING = [1e4]

    def __init__(
        self,
        config: AttPrimitivesConfig,
        execution_engine: PrimitiveExecutionEngine,
        candidate_primitives: CandidatePrimitives,
        logger: Optional[logging.Logger] = None,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.config = config
        self.execution_engine = execution_engine
        self.candidate_primitives = candidate_primitives
        self.logger = logger
        self.metrics_logger = metrics_logger
        self.scaling_factors = (
            self.DEFAULT_ONLY_SCALING
            if config.only_default_scalars
            else self.DEFAULT_SCALING_FACTORS
        )
        self._total_interactions = 0
        self._interaction_idx = 0
        self._candidate_idx = 0

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _metrics(self, **fields) -> None:
        if self.metrics_logger is not None:
            self.metrics_logger.log(**fields)

    def _try_candidate(
        self,
        interaction_map: PrimitivesMap,
        layer: Optional[int],
        head: Optional[int],
        interaction: AttentionInteraction | LogitsInteraction,
        abstract_primitive: AbstractPrimitive,
        acc_match_threshold: float,
        phase: str,
        interaction_key: str,
    ) -> bool:
        if layer is not None and head is not None:
            interaction_map[layer][head][interaction] = abstract_primitive
        else:
            interaction_map["lm_head"][interaction] = abstract_primitive

        metrics = self.execution_engine.evaluate(
            interaction_map,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.eval_batch_size,
        )
        accepted = metrics.acc_match >= acc_match_threshold

        self._log(
            f"Candidate acc_match={metrics.acc_match:.4f} "
            f"(threshold={acc_match_threshold:.4f}) "
            f"{'ACCEPTED' if accepted else 'rejected'}"
        )
        self._metrics(
            task="att_candidate_eval",
            phase=phase,
            interaction_idx=self._interaction_idx,
            total_interactions=self._total_interactions,
            candidate_idx=self._candidate_idx,
            interaction_id=interaction_key,
            layer=layer,
            head=head,
            primitive=primitive_label(abstract_primitive.primitive),
            special_primitive=primitive_label(abstract_primitive.special_primitive),
            scaling_factor=abstract_primitive.scaling_factor,
            acc_match=metrics.acc_match,
            acc=metrics.acc,
            kl=metrics.kl,
            task_loss=metrics.task_loss,
            accepted=accepted,
            acc_match_threshold=acc_match_threshold,
        )
        self._candidate_idx += 1

        if not accepted:
            if layer is not None and head is not None:
                interaction_map[layer][head][interaction] = None
            else:
                interaction_map["lm_head"][interaction] = None
            return False

        return True

    def _begin_interaction(
        self,
        phase: str,
        interaction: AttentionInteraction | LogitsInteraction,
        layer: Optional[int],
        head: Optional[int],
        num_candidates: int,
    ) -> str:
        self._interaction_idx += 1
        self._candidate_idx = 0
        key = interaction_id(interaction, layer, head)

        fields: Dict[str, Any] = {
            "task": "att_interaction_start",
            "phase": phase,
            "interaction_idx": self._interaction_idx,
            "total_interactions": self._total_interactions,
            "interaction_id": key,
            "num_candidates": num_candidates,
            "converted_so_far": count_converted(self._interaction_map_ref),
        }
        if isinstance(interaction, AttentionInteraction):
            fields["activation_q"] = interaction.activation_name_to_keep_q
            fields["activation_k"] = interaction.activation_name_to_keep_k
            fields["layer"] = layer
            fields["head"] = head
        else:
            fields["activation"] = interaction.activation_name_to_keep

        self._metrics(**fields)
        self._log(
            f"[{self._interaction_idx}/{self._total_interactions}] "
            f"Searching {key} ({num_candidates} candidate combos)"
        )
        return key

    def _finish_interaction(
        self,
        phase: str,
        interaction_key: str,
        layer: Optional[int],
        head: Optional[int],
        found: bool,
        abstract: Optional[AbstractPrimitive],
        interaction_map: PrimitivesMap,
    ) -> None:
        self._metrics(
            task="att_interaction_complete",
            phase=phase,
            interaction_idx=self._interaction_idx,
            total_interactions=self._total_interactions,
            interaction_id=interaction_key,
            layer=layer,
            head=head,
            found=found,
            best_primitive=abstract_primitive_fields(abstract),
            converted_so_far=count_converted(interaction_map),
        )

        if layer is not None and head is not None:
            self._metrics(
                task="att_head_state",
                layer=layer,
                head=head,
                interaction_idx=self._interaction_idx,
                total_interactions=self._total_interactions,
                interactions=head_interaction_state(interaction_map, layer, head),
            )

        status = abstract.name if found and abstract is not None else "none"
        self._log(f"Interaction {interaction_key} -> {status}")

    def _search_attention(self, interaction_map: PrimitivesMap, acc_match_threshold: float) -> None:
        matrix_primitives = self.candidate_primitives[
            (PrimitiveDomain.ATTENTION, PrimitiveShape.MATRIX)
        ]
        bias_primitives = self.candidate_primitives[
            (PrimitiveDomain.ATTENTION, PrimitiveShape.BIAS)
        ]
        num_layers = len(self.execution_engine.hooked_model.model.transformer.h)

        for layer in range(num_layers):
            num_heads = self.execution_engine.hooked_model.model.transformer.h[layer].attn.num_heads
            for head in range(num_heads):
                for interaction in interaction_map[layer][head]:
                    candidate_params = filter_attention_candidates(
                        interaction,
                        matrix_primitives,
                        bias_primitives,
                        self.execution_engine.converted_mlp,
                        self.config.only_default_scalars,
                    )
                    combos = expand_grid(candidate_params)
                    num_candidates = sum(
                        len(self.scaling_factors)
                        for combo in combos
                        if combo.get("primitive") is not None
                    )

                    interaction_key = self._begin_interaction(
                        phase="attention",
                        interaction=interaction,
                        layer=layer,
                        head=head,
                        num_candidates=num_candidates,
                    )

                    found = False
                    accepted_abstract: Optional[AbstractPrimitive] = None

                    for combo in combos:
                        if found:
                            break

                        primitive = combo["primitive"]
                        special_primitive = combo["special_primitive"]
                        if primitive is None:
                            continue

                        self._log(f"Trying primitive={primitive}")
                        if special_primitive is not None:
                            self._log(f"  special_primitive={special_primitive}")

                        for scaling in self.scaling_factors:
                            if found:
                                break

                            abstract = AbstractPrimitive(
                                name="predefined_primitive",
                                primitive=primitive,
                                special_primitive=special_primitive,
                                scaling_factor=scaling,
                            )

                            if self._try_candidate(
                                interaction_map,
                                layer,
                                head,
                                interaction,
                                abstract,
                                acc_match_threshold,
                                phase="attention",
                                interaction_key=interaction_key,
                            ):
                                found = True
                                abstract.name = self._describe_primitive(abstract)
                                accepted_abstract = abstract

                    self._finish_interaction(
                        phase="attention",
                        interaction_key=interaction_key,
                        layer=layer,
                        head=head,
                        found=found,
                        abstract=accepted_abstract,
                        interaction_map=interaction_map,
                    )

        self._metrics(
            task="att_phase_complete",
            phase="attention",
            converted_count=count_converted(interaction_map),
            total_interactions=self._total_interactions,
        )

    def _search_logits(self, interaction_map: PrimitivesMap, acc_match_threshold: float) -> None:
        matrix_primitives = self.candidate_primitives[
            (PrimitiveDomain.UNEMBEDDING, PrimitiveShape.MATRIX)
        ]
        bias_primitives = self.candidate_primitives[
            (PrimitiveDomain.UNEMBEDDING, PrimitiveShape.BIAS)
        ]

        for interaction in interaction_map["lm_head"]:
            candidate_params = filter_logits_candidates(
                interaction,
                matrix_primitives,
                bias_primitives,
                self.execution_engine.converted_mlp,
                self.config.only_default_scalars,
            )
            combos = expand_grid(candidate_params)
            num_candidates = sum(
                len(self.scaling_factors)
                for combo in combos
                if combo.get("primitive") is not None
            )

            interaction_key = self._begin_interaction(
                phase="lm_head",
                interaction=interaction,
                layer=None,
                head=None,
                num_candidates=num_candidates,
            )

            found = False
            accepted_abstract: Optional[AbstractPrimitive] = None

            for combo in combos:
                if found:
                    break

                primitive = combo["primitive"]
                special_primitive = combo["special_primitive"]
                if primitive is None:
                    continue

                self._log(f"Trying logits primitive={primitive}")
                for scaling in self.scaling_factors:
                    if found:
                        break

                    abstract = AbstractPrimitive(
                        name="predefined_primitive",
                        primitive=primitive,
                        special_primitive=special_primitive,
                        scaling_factor=scaling,
                    )

                    if self._try_candidate(
                        interaction_map,
                        None,
                        None,
                        interaction,
                        abstract,
                        acc_match_threshold,
                        phase="lm_head",
                        interaction_key=interaction_key,
                    ):
                        found = True
                        abstract.name = self._describe_primitive(abstract)
                        accepted_abstract = abstract

            self._finish_interaction(
                phase="lm_head",
                interaction_key=interaction_key,
                layer=None,
                head=None,
                found=found,
                abstract=accepted_abstract,
                interaction_map=interaction_map,
            )

        self._metrics(
            task="att_phase_complete",
            phase="lm_head",
            converted_count=count_converted(interaction_map),
            total_interactions=self._total_interactions,
        )

    @staticmethod
    def _describe_primitive(abstract: AbstractPrimitive) -> str:
        name = f"{abstract.scaling_factor} * predefined_primitive: {abstract.primitive}"
        if abstract.special_primitive is not None:
            name += f"; special_primitive={abstract.special_primitive}"
        return name

    def _build_eval(self, interaction_map: PrimitivesMap) -> PrimitiveEval:
        num_primitives = count_interactions(interaction_map)
        primitives_converted = count_converted(interaction_map)

        if num_primitives == 0:
            return PrimitiveEval(
                zero_parameters=[1],
                total_parameters=[-1],
                is_fully_replaced=[True],
            )

        return PrimitiveEval(
            zero_parameters=[primitives_converted],
            total_parameters=[num_primitives],
            is_fully_replaced=[num_primitives == primitives_converted],
        )

    def search(
        self,
        interaction_map: PrimitivesMap,
        acc_match_threshold: float,
    ) -> Tuple[PrimitivesMap, PrimitiveEval]:
        primitives_to_try = copy.deepcopy(interaction_map)
        self._interaction_map_ref = primitives_to_try
        self._total_interactions = count_interactions(interaction_map)
        self._interaction_idx = 0

        self._log("Starting greedy search for attention primitives")
        self._search_attention(primitives_to_try, acc_match_threshold)

        self._log("Starting greedy search for unembedding primitives")
        self._search_logits(primitives_to_try, acc_match_threshold)

        eval_result = self._build_eval(primitives_to_try)
        self._metrics(
            task="att_greedy_complete",
            converted_count=eval_result.zero_parameters[0],
            total_interactions=eval_result.total_parameters[0],
            fully_replaced=eval_result.is_fully_replaced[0],
        )

        return primitives_to_try, eval_result
