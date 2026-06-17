from pathlib import Path

import primitives_att.primitives  # noqa: F401 — register primitives
from decompilation.utilities.decompilation_dataclasses import DecompilationRunConfig
from decompilation.utilities.decompilation_utils import validate_stage_prerequisites
from primitives_att.pipeline import AttPrimitivePipeline
from primitives_mlp.pipeline import MLPPrimitivePipeline
from pruning.pipeline import CausalPruningPipeline
from utilities.logger import setup_logger


class DecompilationPipeline:
    def __init__(self, config: DecompilationRunConfig):
        self.config = config
        self.logger = setup_logger(
            Path(config.pruning.output_dir) / config.pruning.exp_name / "decompilation",
            name="decompilation",
        )
        self.logger.info("Decompilation pipeline initialized")
        self._log_stage_plan()

    def _log_stage_plan(self) -> None:
        stages = [
            ("pruning", self.config.run_pruning),
            ("MLP primitives", self.config.run_mlp_primitives),
            ("attention primitives", self.config.run_att_primitives),
        ]
        enabled = [name for name, run in stages if run]
        skipped = [name for name, run in stages if not run]
        self.logger.info(f"Stages to run: {', '.join(enabled) or 'none'}")
        if skipped:
            self.logger.info(f"Stages skipped: {', '.join(skipped)}")

    def run(self) -> None:
        validate_stage_prerequisites(self.config)

        if self.config.run_pruning:
            self.logger.info("=" * 60)
            self.logger.info("Stage 1/3: Causal pruning")
            self.logger.info("=" * 60)
            CausalPruningPipeline(self.config.pruning).run()
        else:
            self.logger.info("Skipping causal pruning (run_pruning=False)")

        if self.config.run_mlp_primitives:
            self.logger.info("=" * 60)
            self.logger.info("Stage 2/3: MLP primitive replacement")
            self.logger.info("=" * 60)
            MLPPrimitivePipeline(self.config.mlp_primitives).run()
        else:
            self.logger.info("Skipping MLP primitive replacement (run_mlp_primitives=False)")

        if self.config.run_att_primitives:
            self.logger.info("=" * 60)
            self.logger.info("Stage 3/3: Attention primitive replacement")
            self.logger.info("=" * 60)
            AttPrimitivePipeline(self.config.att_primitives).run()
        else:
            self.logger.info("Skipping attention primitive replacement (run_att_primitives=False)")

        self.logger.info("Decompilation pipeline complete")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])
