"""
Search strategies for attention and unembedding primitive replacement.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from primitives_att.core.PrimitiveExecutionEngine import PrimitiveExecutionEngine
from primitives_att.core.MatrixRounder import MatrixRounder
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
    RoundConfig,
)
from primitives_att.utilities.product_tracing import (
    get_product_for_one_side_for_head_ignore_dep_prod,
    get_product_for_one_side_for_unembed_ignore_dep_prod,
)
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape
from primitives_att.utilities.search_logging import (
    abstract_primitive_fields,
    count_converted,
    count_interactions,
    head_interaction_state,
    interaction_eval,
    interaction_id,
    primitive_label,
)
from utilities.metrics_logger import MetricsLogger

PrimitivesMap = Dict[Any, Any]
CandidatePrimitives = Dict[Tuple[PrimitiveDomain, PrimitiveShape], List[Primitive]]
VanillaInteraction = tuple[
    AttentionInteraction | LogitsInteraction,
    Optional[int],
    Optional[int],
]

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
        candidate_fields: Dict[str, Any] = {
            "task": "att_candidate_eval",
            "phase": phase,
            "interaction_idx": self._interaction_idx,
            "total_interactions": self._total_interactions,
            "candidate_idx": self._candidate_idx,
            "interaction_id": interaction_key,
            "primitive": primitive_label(abstract_primitive.primitive),
            "special_primitive": primitive_label(abstract_primitive.special_primitive),
            "scaling_factor": abstract_primitive.scaling_factor,
            "acc_match": metrics.acc_match,
            "acc": metrics.acc,
            "kl": metrics.kl,
            "task_loss": metrics.task_loss,
            "accepted": accepted,
            "acc_match_threshold": acc_match_threshold,
        }
        if layer is not None and head is not None:
            candidate_fields["layer"] = layer
            candidate_fields["head"] = head
        elif isinstance(interaction, LogitsInteraction):
            candidate_fields["activation"] = interaction.activation_name_to_keep
        self._metrics(**candidate_fields)
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
        complete_fields: Dict[str, Any] = {
            "task": "att_interaction_complete",
            "phase": phase,
            "interaction_idx": self._interaction_idx,
            "total_interactions": self._total_interactions,
            "interaction_id": interaction_key,
            "found": found,
            "best_primitive": abstract_primitive_fields(abstract),
            "converted_so_far": count_converted(interaction_map),
        }
        if layer is not None and head is not None:
            complete_fields["layer"] = layer
            complete_fields["head"] = head
        elif interaction_key.startswith("lm_head-"):
            complete_fields["activation"] = interaction_key[len("lm_head-") :]
        self._metrics(**complete_fields)

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


class RoundSearchStrategy(SearchStrategy):
    def __init__(
        self,
        config: AttPrimitivesConfig,
        execution_engine: PrimitiveExecutionEngine,
        candidate_primitives: CandidatePrimitives,
        logger: Optional[logging.Logger] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        only_unset: bool = False,
    ):
        self.config = config
        self.execution_engine = execution_engine
        self.candidate_primitives = candidate_primitives
        self.logger = logger
        self.metrics_logger = metrics_logger
        self.only_unset = only_unset

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _metrics(self, **fields) -> None:
        if self.metrics_logger is not None:
            self.metrics_logger.log(**fields)

    def _round_config(self) -> RoundConfig:
        if self.config.round is not None:
            return self.config.round
        return RoundConfig()

    @staticmethod
    def _rounder_key(
        interaction: AttentionInteraction | LogitsInteraction,
        layer: Optional[int],
        head: Optional[int],
    ) -> str:
        if isinstance(interaction, LogitsInteraction):
            return interaction.activation_name_to_keep
        return (
            f"{layer}-{head}-"
            f"{interaction.activation_name_to_keep_k}-"
            f"{interaction.activation_name_to_keep_q}"
        )

    def _collect_vanilla_interactions(
        self, interaction_map: PrimitivesMap
    ) -> list[VanillaInteraction]:
        vanilla: list[VanillaInteraction] = []
        num_layers = len(self.execution_engine.hooked_model.model.transformer.h)

        for layer in range(num_layers):
            num_heads = self.execution_engine.hooked_model.model.transformer.h[layer].attn.num_heads
            for head in range(num_heads):
                for interaction, abstract in interaction_map[layer][head].items():
                    if self.only_unset and abstract is not None:
                        continue
                    vanilla.append((interaction, layer, head))

        for interaction, abstract in interaction_map["lm_head"].items():
            if self.only_unset and abstract is not None:
                continue
            vanilla.append((interaction, None, None))

        return vanilla

    def _compute_indep_prod(
        self,
        interaction: AttentionInteraction | LogitsInteraction,
        layer: Optional[int],
        head: Optional[int],
    ) -> torch.Tensor:
        hooked_model = self.execution_engine.hooked_model
        oa_vecs = self.execution_engine.oa_vecs
        converted_mlp = self.execution_engine.converted_mlp

        if isinstance(interaction, LogitsInteraction):
            return get_product_for_one_side_for_unembed_ignore_dep_prod(
                hooked_model,
                oa_vecs,
                converted_mlp,
                interaction.activation_name_to_keep,
            ).detach()

        indep_prod_k = get_product_for_one_side_for_head_ignore_dep_prod(
            hooked_model,
            oa_vecs,
            converted_mlp,
            layer,
            head,
            "k",
            interaction.activation_name_to_keep_k,
        )
        
        if interaction.activation_name_to_keep_q is not None:
            indep_prod_q = get_product_for_one_side_for_head_ignore_dep_prod(
                hooked_model,
                oa_vecs,
                converted_mlp,
                layer,
                head,
                "q",
                interaction.activation_name_to_keep_q,
            )
            return (indep_prod_q @ indep_prod_k.transpose(-1, -2)).detach()

        alpha = oa_vecs.q_bias_term.data[
            oa_vecs.to_q_bias[(layer, head, interaction.activation_name_to_keep_k)]
        ].unsqueeze(0)
        return (alpha @ indep_prod_k.transpose(-1, -2)).squeeze().detach()

    @staticmethod
    def _describe_replacement_matrix(rounder: MatrixRounder) -> str:
        matrix = rounder.get_matrix()
        name = (
            f"projection[non-zero={(matrix != 0).sum().item()} "
            f"out of {matrix.numel()}]"
        )
        if rounder.do_round:
            name += "[round]"
        return name

    def _assign_rounder(
        self,
        interaction_map: PrimitivesMap,
        interaction: AttentionInteraction | LogitsInteraction,
        layer: Optional[int],
        head: Optional[int],
        rounder: MatrixRounder,
    ) -> None:
        abstract = AbstractPrimitive(name="replacement_matrix", replacement_matrix=rounder)
        if layer is not None and head is not None:
            interaction_map[layer][head][interaction] = abstract
        else:
            interaction_map["lm_head"][interaction] = abstract

    def _build_eval(
        self,
        interaction_map: PrimitivesMap,
        rounders: Dict[str, MatrixRounder],
        accepted_rounding: Dict[str, bool],
    ) -> PrimitiveEval:
        if len(rounders) == 0:
            return PrimitiveEval(
                zero_parameters=[1],
                total_parameters=[-1],
                is_fully_replaced=[True],
            )

        zero_parameters = []
        total_parameters = []
        for key, rounder in rounders.items():
            matrix = rounder.get_matrix()
            zero_parameters.append((matrix == 0).sum().item())
            total_parameters.append(matrix.numel())
            if key in accepted_rounding:
                abstract = self._find_abstract_by_key(interaction_map, key)
                if abstract is not None:
                    abstract.name = self._describe_replacement_matrix(rounder)

        is_fully_replaced = self._build_is_fully_replaced(interaction_map)
        
        return PrimitiveEval(
            zero_parameters=zero_parameters,
            total_parameters=total_parameters,
            is_fully_replaced=is_fully_replaced,
        )

    def _find_abstract_by_key(
        self, interaction_map: PrimitivesMap, key: str
    ) -> Optional[AbstractPrimitive]:
        for interaction, abstract in interaction_map["lm_head"].items():
            if self._rounder_key(interaction, None, None) == key:
                return abstract
        num_layers = len(self.execution_engine.hooked_model.model.transformer.h)
        for layer in range(num_layers):
            num_heads = self.execution_engine.hooked_model.model.transformer.h[layer].attn.num_heads
            for head in range(num_heads):
                for interaction, abstract in interaction_map[layer][head].items():
                    if self._rounder_key(interaction, layer, head) == key:
                        return abstract
        return None

    @staticmethod
    def _build_is_fully_replaced(interaction_map: PrimitivesMap) -> list[bool]:
        is_fully_replaced: list[bool] = []
        
        for abstract in interaction_map["lm_head"].values():
            if abstract is None:
                is_fully_replaced.append(False)
            elif abstract.replacement_matrix is not None:
                is_fully_replaced.append(abstract.replacement_matrix.do_round)
            elif abstract.primitive is not None:
                is_fully_replaced.append(True)
            else:
                is_fully_replaced.append(False)

        for layer in interaction_map:
            if not isinstance(layer, int):
                continue
            
            for head in interaction_map[layer]:
                for abstract in interaction_map[layer][head].values():
                    if abstract is None:
                        is_fully_replaced.append(False)
                    elif abstract.replacement_matrix is not None:
                        is_fully_replaced.append(abstract.replacement_matrix.do_round)
                    elif abstract.primitive is not None:
                        is_fully_replaced.append(True)
                    else:
                        is_fully_replaced.append(False)
        
        return is_fully_replaced

    def search(
        self,
        interaction_map: PrimitivesMap,
        acc_match_threshold: float,
    ) -> Tuple[PrimitivesMap, PrimitiveEval]:
        round_cfg = self._round_config()
        rounder_params = {
            key: getattr(round_cfg, key)
            for key in MatrixRounder.possible_params_and_default_values
            if hasattr(round_cfg, key)
        }

        vanilla_interactions = self._collect_vanilla_interactions(interaction_map)
        primitives_to_try = copy.deepcopy(interaction_map)
        rounders: Dict[str, MatrixRounder] = {}

        if len(vanilla_interactions) == 0:
            self._log("No unset interactions to round")
            return primitives_to_try, PrimitiveEval(
                zero_parameters=[1],
                total_parameters=[-1],
                is_fully_replaced=self._build_is_fully_replaced(primitives_to_try),
            )

        self._log(f"Initializing MatrixRounder for {len(vanilla_interactions)} interactions")
        device = self.execution_engine.hooked_model.device
        for interaction, layer, head in vanilla_interactions:
            indep_prod = self._compute_indep_prod(interaction, layer, head).to(device)
            key = self._rounder_key(interaction, layer, head)
            rounders[key] = MatrixRounder(indep_prod, params=rounder_params)

        for interaction, layer, head in vanilla_interactions:
            key = self._rounder_key(interaction, layer, head)
            self._assign_rounder(
                primitives_to_try, interaction, layer, head, rounders[key]
            )

        pre_metrics = self.execution_engine.evaluate(
            primitives_to_try,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.eval_batch_size,
        )
        self._log(
            f"acc_match before round training (continuous matrices): "
            f"{pre_metrics.acc_match:.4f}, kl={pre_metrics.kl:.4f}"
        )

        for rounder in rounders.values():
            rounder.disable_rounding()

        continuous_metrics = self.execution_engine.evaluate(
            primitives_to_try,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.eval_batch_size,
        )
        self._log(
            f"acc_match before round training (rounding disabled): "
            f"{continuous_metrics.acc_match:.4f}, kl={continuous_metrics.kl:.4f}"
        )

        stages: list[Optional[int]] = [1, 2] if round_cfg.two_stages else [None]
        for stage in stages:
            self._log(f"Round training stage {stage}")
            self.execution_engine.train_rounders(
                primitives_map=primitives_to_try,
                rounders=rounders,
                training_lambda=round_cfg.training_lambda,
                num_steps=round_cfg.num_steps,
                batch_size=self.config.eval_batch_size,
                lr=round_cfg.lr,
                log_interval=self.config.num_test_steps,
                match_acc_threshold=acc_match_threshold,
                stage=stage,
            )
            stage_metrics = self.execution_engine.evaluate(
                primitives_to_try,
                num_steps=self.config.num_test_steps,
                batch_size=self.config.eval_batch_size,
            )
            self._log(
                f"acc_match after stage {stage} (not rounded): "
                f"{stage_metrics.acc_match:.4f}, kl={stage_metrics.kl:.4f}"
            )
            self._metrics(
                task="att_round_stage_complete",
                stage=stage,
                acc_match=stage_metrics.acc_match,
                kl=stage_metrics.kl,
            )

        accepted_rounding: Dict[str, bool] = {}
        for key, rounder in rounders.items():
            rounder.enable_rounding()
            rounded_metrics = self.execution_engine.evaluate(
                primitives_to_try,
                num_steps=self.config.num_test_steps,
                batch_size=self.config.eval_batch_size,
            )
            accepted = rounded_metrics.acc_match >= acc_match_threshold
            if not accepted:
                self._log(
                    f"Reject rounding for {key}: "
                    f"{rounded_metrics.acc_match:.4f} < {acc_match_threshold:.4f}"
                )
                rounder.disable_rounding()
            else:
                self._log(
                    f"Accept rounding for {key}: "
                    f"{rounded_metrics.acc_match:.4f} >= {acc_match_threshold:.4f}"
                )
            accepted_rounding[key] = accepted
            matrix = rounder.get_matrix()
            self._metrics(
                task="att_round_acceptance",
                rounder_key=key,
                accepted=accepted,
                acc_match=rounded_metrics.acc_match,
                kl=rounded_metrics.kl,
                non_zero=int((matrix != 0).sum().item()),
                total_params=matrix.numel(),
            )

        eval_result = self._build_eval(primitives_to_try, rounders, accepted_rounding)
        interaction_summary = interaction_eval(primitives_to_try)
        self._metrics(
            task="att_round_complete",
            converted_count=interaction_summary.zero_parameters[0],
            total_interactions=interaction_summary.total_parameters[0],
            fully_replaced=interaction_summary.is_fully_replaced[0],
        )
        return primitives_to_try, eval_result


class GreedyThenRoundStrategy(SearchStrategy):
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

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def search(
        self,
        interaction_map: PrimitivesMap,
        acc_match_threshold: float,
    ) -> Tuple[PrimitivesMap, PrimitiveEval]:
        self._log("Starting greedy-then-round search")

        greedy_strategy = GreedySearchStrategy(
            config=self.config,
            execution_engine=self.execution_engine,
            candidate_primitives=self.candidate_primitives,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
        )
        primitives_after_greedy, _ = greedy_strategy.search(
            interaction_map, acc_match_threshold
        )

        self._log("Greedy search complete; starting round fallback for unset interactions")
        round_strategy = RoundSearchStrategy(
            config=self.config,
            execution_engine=self.execution_engine,
            candidate_primitives=self.candidate_primitives,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
            only_unset=True,
        )
        primitives_after_round, _ = round_strategy.search(
            primitives_after_greedy, acc_match_threshold
        )

        if self.metrics_logger is not None:
            final_eval = interaction_eval(primitives_after_round)
            self.metrics_logger.log(
                task="att_greedy_then_round_complete",
                converted_count=final_eval.zero_parameters[0],
                total_interactions=final_eval.total_parameters[0],
                fully_replaced=final_eval.is_fully_replaced[0],
            )

        return primitives_after_round, interaction_eval(primitives_after_round)
