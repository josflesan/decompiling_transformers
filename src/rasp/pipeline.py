from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from rasp.core.artifact_collector import ArtifactCollector
from rasp.core.DRASPConverter import DRASPConverter
from rasp.utilities.input_loaders import InputLoader
from rasp.utilities.rasp_dataclasses import DecompilationResult, RaspRunConfig
from utilities.logger import setup_logger
from utilities.metrics_logger import MetricsLogger


class RaspPipeline:
    def __init__(self, config: RaspRunConfig):
        self.config = config
        self.logger = setup_logger(config.full_output_dir, name="rasp")
        self.metrics_logger = MetricsLogger(config.full_output_dir)
        self.inputs = None
        self.result: DecompilationResult | None = None
        self.manifest: Dict[str, Any] | None = None

        self.logger.info("RASP Decompilation Pipeline initialized")

    def _setup(self) -> None:
        self.inputs = InputLoader(self.config).load()
        self.logger.info(
            "Loaded upstream artifacts for exp_name=%s (split_mlp=%s)",
            self.config.exp_name,
            self.inputs.split_mlp,
        )

    def _build_summary(self, result: DecompilationResult, manifest: Dict[str, Any]) -> Dict[str, Any]:
        assert self.inputs is not None
        coverage = manifest["coverage"]
        return {
            "exp_name": self.config.exp_name,
            "convert_to_primitives": self.config.convert_to_primitives,
            "split_mlp": self.inputs.split_mlp,
            "line_count": len(result.lines),
            "circled_matrix_count": len(result.circled_labels),
            "coverage": coverage,
            "pruning_metrics": self.inputs.pruning_metrics,
            "att_stats": self.inputs.att_stats,
            "var_mapping": result.var_mapping,
            "selector_to_config": {
                key: self._jsonable(value) for key, value in result.selector_to_config.items()
            },
        }

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): RaspPipeline._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RaspPipeline._jsonable(v) for v in obj]
        return obj

    def run(self) -> DecompilationResult:
        code_path = self.config.full_output_dir / "generated_code.txt"
        manifest_path = self.config.full_output_dir / "artifacts" / "manifest.json"

        if self.config.skip_convert and code_path.exists() and manifest_path.exists():
            self.logger.info(
                "skip_convert=True and outputs exist; loading cached decompilation"
            )
            with open(self.config.full_output_dir / "output.json") as f:
                cached = json.load(f)
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            return DecompilationResult(
                lines=[
                    line.split(". ", 1)[-1] if ". " in line else line
                    for line in code_path.read_text().splitlines()
                ],
                selector_to_config=cached.get("selector_to_config", {}),
                var_mapping=cached.get("var_mapping", {}),
            )

        self._setup()
        assert self.inputs is not None

        converter = DRASPConverter(
            config=self.inputs.pruning_config,
            split_mlp=self.inputs.split_mlp,
            converted_mlp=self.inputs.converted_mlp,
            interaction_map=self.inputs.interaction_map,
            convert_to_primitives=self.config.convert_to_primitives,
            show_logits_for_unconverted_mlp=self.config.show_logits_for_unconverted_mlp,
        )
        self.result = converter.convert()

        numbered_lines = [
            f"{idx + 1}. {line}" for idx, line in enumerate(self.result.lines)
        ]
        code_path.write_text("\n".join(numbered_lines))

        exp_root = Path(self.config.output_dir) / self.config.exp_name
        collector = ArtifactCollector(
            exp_root=exp_root,
            inputs=self.inputs,
            result=self.result,
            numbered_lines=numbered_lines,
            mlp_failure_threshold=self.config.mlp_failure_threshold,
            show_logits_for_unconverted_mlp=self.config.show_logits_for_unconverted_mlp,
        )
        self.manifest = collector.build()

        artifacts_dir = self.config.full_output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        summary = self._build_summary(self.result, self.manifest)
        with open(self.config.full_output_dir / "output.json", "w") as f:
            json.dump(summary, f, indent=2)

        att_cov = self.manifest["coverage"]["attention"]
        mlp_cov = self.manifest["coverage"]["mlp"]
        self.metrics_logger.log(
            task="rasp_complete",
            exp_name=self.config.exp_name,
            line_count=len(self.result.lines),
            circled_matrix_count=len(self.result.circled_labels),
            convert_to_primitives=self.config.convert_to_primitives,
            split_mlp=self.inputs.split_mlp,
            att_predefined=att_cov["predefined"],
            att_rounded=att_cov["rounded"],
            att_unconverted=att_cov["unconverted"],
            lm_predefined=self.manifest["coverage"]["lm_head"]["predefined"],
            lm_rounded=self.manifest["coverage"]["lm_head"]["rounded"],
            mlp_converted=mlp_cov["converted"],
            mlp_failed=mlp_cov["failed"],
            unexplained_mlp_count=len(self.manifest["unexplained_mlp"]),
        )

        self.logger.info("D-RASP program saved to %s", code_path)
        self.logger.info("Artifact manifest saved to %s", manifest_path)
        self.logger.info("RASP decompilation complete!")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])

        return self.result
