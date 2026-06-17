"""
Orchestrates primitive search for attention and unembedding interactions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from primitives_att.core.PrimitiveExecutionEngine import PrimitiveExecutionEngine
from primitives_att.core.search_strategies import (
    GreedySearchStrategy,
    GreedyThenRoundStrategy,
    RoundSearchStrategy,
    SearchStrategy,
)
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import (
    AttPrimitiveSearchOutput,
    AttPrimitivesConfig,
    EvalMetrics,
)
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape
from primitives_att.utilities.search_logging import count_interactions, interaction_eval
from utilities.metrics_logger import MetricsLogger

PrimitivesMap = Dict[Any, Any]
CandidatePrimitives = Dict[tuple[PrimitiveDomain, PrimitiveShape], list[Primitive]]

SEARCH_STRATEGIES: Dict[str, type[SearchStrategy]] = {
    "greedy_search": GreedySearchStrategy,
    "round": RoundSearchStrategy,
    "greedy_search_then_round": GreedyThenRoundStrategy,
}


class PrimitiveSearchEngine:
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

    def _metrics(self, **fields) -> None:
        if self.metrics_logger is not None:
            self.metrics_logger.log(**fields)

    def search(self, interaction_map: PrimitivesMap) -> AttPrimitiveSearchOutput:
        total_interactions = count_interactions(interaction_map)

        baseline: EvalMetrics = self.execution_engine.evaluate(
            interaction_map,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.eval_batch_size,
        )
        self._log(
            f"Baseline acc_match={baseline.acc_match:.4f}, "
            f"kl={baseline.kl:.4f}, acc={baseline.acc:.4f}"
        )

        acc_match_threshold = baseline.acc_match * self.config.threshold_on_acc
        search_threshold = baseline.acc_match * (self.config.threshold_on_acc + 0.01)

        self._metrics(
            task="att_search_start",
            search_method=self.config.search_method,
            total_interactions=total_interactions,
            baseline_acc_match=baseline.acc_match,
            baseline_acc=baseline.acc,
            baseline_kl=baseline.kl,
            baseline_task_loss=baseline.task_loss,
            acc_match_threshold=acc_match_threshold,
            search_threshold=search_threshold,
            threshold_on_acc=self.config.threshold_on_acc,
            only_default_scalars=self.config.only_default_scalars,
            num_test_steps=self.config.num_test_steps,
            eval_batch_size=self.config.eval_batch_size,
        )

        strategy_cls = SEARCH_STRATEGIES.get(self.config.search_method)
        if strategy_cls is None:
            raise ValueError(f"Unknown search method: {self.config.search_method}")

        strategy = strategy_cls(
            config=self.config,
            execution_engine=self.execution_engine,
            candidate_primitives=self.candidate_primitives,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
        )
        primitives_after_search, _ = strategy.search(
            interaction_map, search_threshold
        )
        primitive_eval = interaction_eval(primitives_after_search)

        final_metrics = self.execution_engine.evaluate(
            primitives_after_search,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.eval_batch_size,
        )
        self._log(
            f"Post-search acc_match={final_metrics.acc_match:.4f}, kl={final_metrics.kl:.4f}"
        )

        if final_metrics.acc_match < acc_match_threshold:
            self._log(
                f"Final acc_match {final_metrics.acc_match:.4f} below threshold "
                f"{acc_match_threshold:.4f}; keeping search results anyway"
            )

        stats = {
            "acc": {"before": baseline.acc, "after": final_metrics.acc},
            "kl": {"before": baseline.kl, "after": final_metrics.kl},
            "acc_match": {"before": baseline.acc_match, "after": final_metrics.acc_match},
            "task_loss": {"before": baseline.task_loss, "after": final_metrics.task_loss},
        }

        self._metrics(
            task="att_search_complete",
            acc_match_before=baseline.acc_match,
            acc_match_after=final_metrics.acc_match,
            acc_before=baseline.acc,
            acc_after=final_metrics.acc,
            kl_before=baseline.kl,
            kl_after=final_metrics.kl,
            converted_count=primitive_eval.zero_parameters[0],
            total_interactions=primitive_eval.total_parameters[0],
            fully_replaced=primitive_eval.is_fully_replaced[0],
        )

        return AttPrimitiveSearchOutput(
            primitives=primitives_after_search,
            eval=primitive_eval,
            stats=stats,
        )
