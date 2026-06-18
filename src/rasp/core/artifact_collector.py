from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from primitives_att.utilities.att_primitive_dataclasses import AbstractPrimitive
from primitives_mlp.utilities.logit_lens_cache import matching_cache_keys_for_path
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from rasp.core.line_emitters import is_replacement_matrix
from rasp.utilities.mlp_name_mapper import mlp_search_output_to_primitive_key
from rasp.utilities.rasp_dataclasses import CircledLabel, DecompilationResult, RaspInputs

_KIND_FROM_SUBDIR = {
    "primitives-matrices": "primitive_matrix",
    "original-matrices": "original_matrix",
    "primitives-example": "primitive_example",
    "original-example": "original_example",
}


class ArtifactCollector:
    def __init__(
        self,
        exp_root: Path,
        inputs: RaspInputs,
        result: DecompilationResult,
        numbered_lines: List[str],
        mlp_failure_threshold: float,
        show_logits_for_unconverted_mlp: bool,
    ):
        self.exp_root = exp_root
        self.inputs = inputs
        self.result = result
        self.numbered_lines = numbered_lines
        self.mlp_failure_threshold = mlp_failure_threshold
        self.show_logits_for_unconverted_mlp = show_logits_for_unconverted_mlp

    def build(self) -> Dict[str, Any]:
        return {
            "program": self._build_program(),
            "coverage": self._build_coverage(),
            "circled_matrices": self._build_circled_matrices(),
            "unexplained_mlp": self._build_unexplained_mlp(),
            "heatmap_index": self._build_heatmap_index(),
        }

    def _build_program(self) -> Dict[str, Any]:
        prediction_line = next(
            (line for line in self.numbered_lines if "prediction" in line.lower()),
            None,
        )
        return {
            "lines": self.numbered_lines,
            "prediction_line": prediction_line,
            "var_mapping": self.result.var_mapping,
            "selector_to_config": {
                key: self._jsonable(value)
                for key, value in self.result.selector_to_config.items()
            },
        }

    def _build_coverage(self) -> Dict[str, Any]:
        att_counts = self._count_attention_coverage()
        lm_counts = self._count_lm_head_coverage()
        mlp_counts = self._count_mlp_coverage()

        return {
            "attention": self._with_percentages(att_counts),
            "lm_head": self._with_percentages(lm_counts),
            "mlp": self._with_percentages(mlp_counts),
            "upstream": {
                "pruning": self.inputs.pruning_metrics,
                "att": self.inputs.att_stats,
            },
        }

    def _count_attention_coverage(self) -> Dict[str, int]:
        counts = {"predefined": 0, "rounded": 0, "unconverted": 0}
        interaction_map = self.inputs.interaction_map
        for layer, heads in interaction_map.items():
            if layer == "lm_head" or not isinstance(heads, dict):
                continue
            for head_interactions in heads.values():
                for abstract in head_interactions.values():
                    if not isinstance(abstract, AbstractPrimitive):
                        continue
                    counts[self._classify_abstract(abstract)] += 1
        return counts

    def _count_lm_head_coverage(self) -> Dict[str, int]:
        counts = {"predefined": 0, "rounded": 0, "unconverted": 0}
        lm_head = self.inputs.interaction_map.get("lm_head")
        if not isinstance(lm_head, dict):
            return counts
        for abstract in lm_head.values():
            if isinstance(abstract, AbstractPrimitive):
                counts[self._classify_abstract(abstract)] += 1
        return counts

    def _count_mlp_coverage(self) -> Dict[str, int]:
        all_paths = self._enumerate_mlp_paths()
        converted = 0
        failed = 0
        skipped = 0

        for path in all_paths:
            search_output = self.inputs.converted_mlp.get(path)
            if search_output is None:
                skipped += 1
                continue
            if self._is_mlp_converted(search_output):
                converted += 1
            else:
                failed += 1

        return {
            "converted": converted,
            "failed": failed,
            "skipped": skipped,
            "total": len(all_paths),
        }

    def _build_circled_matrices(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for label in self.result.circled_labels.values():
            heatmap_paths = self._resolve_circled_heatmap_paths(label)
            entry = asdict(label)
            entry["heatmap_paths"] = heatmap_paths
            entries.append(entry)
        entries.sort(key=lambda item: item.get("label", ""))
        return entries

    def _build_unexplained_mlp(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        mlp_io = self.inputs.mlp_input_output or {}

        for path in self._enumerate_mlp_paths():
            search_output = self.inputs.converted_mlp.get(path)
            if search_output is not None and self._is_mlp_converted(search_output):
                continue

            if search_output is None:
                status = "missing"
                primitive_name = None
                accuracy = None
            else:
                status = "failed"
                primitive_name = (
                    search_output.best_primitive.name
                    if search_output.best_primitive is not None
                    else None
                )
                accuracy = search_output.best_accuracy

            cache_keys = matching_cache_keys_for_path(path, mlp_io)
            entries.append(
                {
                    "path": path,
                    "status": status,
                    "primitive": primitive_name,
                    "accuracy": accuracy,
                    "logit_lens_cached": bool(cache_keys),
                    "has_logit_lens": bool(cache_keys),
                    "logit_lens_cache_keys": [self._jsonable(key) for key in cache_keys],
                }
            )

        return entries

    def _build_heatmap_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        heatmaps_root = self.exp_root / "att_primitives" / "heatmaps"
        if not heatmaps_root.exists():
            return index

        for png_path in sorted(heatmaps_root.rglob("*.png")):
            rel_path = str(png_path.relative_to(self.exp_root))
            activation_key = self._activation_key_from_heatmap_path(png_path, heatmaps_root)
            kind = self._kind_from_heatmap_path(png_path)
            index[activation_key] = {
                "relative_path": rel_path,
                "kind": kind,
                "layer": self._parse_layer_head(png_path, heatmaps_root)[0],
                "head": self._parse_layer_head(png_path, heatmaps_root)[1],
                "save_name": png_path.stem,
            }

        return index

    def _resolve_circled_heatmap_paths(self, label: CircledLabel) -> Dict[str, str]:
        heatmaps_root = self.exp_root / "att_primitives" / "heatmaps"
        if not heatmaps_root.exists():
            return {}

        if label.inp_path is not None:
            prefix = heatmaps_root / "lm_head"
            save_name = "bias" if label.inp_path == "vocab_bias" else label.inp_path
            candidates = [
                ("primitive_matrix", prefix / "primitives-matrices" / f"{save_name}.png"),
                ("primitive_example", prefix / "primitives-example" / f"{save_name}.png"),
                ("original_matrix", prefix / "original-matrices" / f"{save_name}.png"),
            ]
        else:
            prefix = heatmaps_root / f"{label.layer}-{label.head}"
            if label.q_path is None:
                save_name = f"bias-{label.k_path}"
            else:
                save_name = f"{label.q_path}-{label.k_path}"
            candidates = [
                ("primitive_matrix", prefix / "primitives-matrices" / f"{save_name}.png"),
                ("primitive_example", prefix / "primitives-example" / f"{save_name}.png"),
                ("original_matrix", prefix / "original-matrices" / f"{save_name}.png"),
            ]

        return {
            kind: str(path.relative_to(self.exp_root))
            for kind, path in candidates
            if path.exists()
        }

    def _enumerate_mlp_paths(self) -> List[str]:
        config = self.inputs.pruning_config
        paths: List[str] = []
        num_layers = len(config) - 1

        for layer_idx in range(num_layers):
            layer = config[layer_idx]
            mlp_inputs = layer.get("mlp", [])
            if not mlp_inputs:
                continue
            if self.inputs.split_mlp:
                for inp in mlp_inputs:
                    paths.append(f"mlp-{layer_idx}-{inp}")
            else:
                paths.append(f"mlp-{layer_idx}")

        return paths

    def _is_mlp_converted(self, search_output: PrimitiveSearchOutput) -> bool:
        primitive_key = mlp_search_output_to_primitive_key(search_output)
        if primitive_key in ("unknown", "noop", "no_op"):
            return primitive_key in ("noop", "no_op")
        return search_output.best_accuracy >= self.mlp_failure_threshold

    @staticmethod
    def _classify_abstract(abstract: AbstractPrimitive) -> str:
        if abstract.primitive is not None:
            return "predefined"
        if is_replacement_matrix(abstract):
            return "rounded"
        return "unconverted"

    @staticmethod
    def _with_percentages(counts: Dict[str, int]) -> Dict[str, Any]:
        total = counts.get("total")
        if total is None:
            total = sum(
                value
                for key, value in counts.items()
                if key not in {"total", "skipped"} and isinstance(value, int)
            )
        result = dict(counts)
        result["total"] = total
        for key, value in counts.items():
            if not key.startswith("pct_") and isinstance(value, int) and total > 0:
                result[f"pct_{key}"] = value / total
        return result

    @staticmethod
    def _kind_from_heatmap_path(png_path: Path) -> str:
        for part in png_path.parts:
            if part in _KIND_FROM_SUBDIR:
                return _KIND_FROM_SUBDIR[part]
        return "unknown"

    @staticmethod
    def _parse_layer_head(png_path: Path, heatmaps_root: Path) -> tuple[Optional[int], Optional[int]]:
        try:
            rel_parts = png_path.relative_to(heatmaps_root).parts
        except ValueError:
            return None, None

        if not rel_parts:
            return None, None
        if rel_parts[0] == "lm_head":
            return None, None

        match = re.fullmatch(r"(\d+)-(\d+)", rel_parts[0])
        if match is None:
            return None, None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _activation_key_from_heatmap_path(png_path: Path, heatmaps_root: Path) -> str:
        rel = png_path.relative_to(heatmaps_root)
        parts = rel.parts
        if not parts:
            return png_path.stem

        if parts[0] == "lm_head":
            prefix = "lm_head"
        elif re.fullmatch(r"\d+-\d+", parts[0]):
            prefix = parts[0]
        else:
            prefix = parts[0]

        subdir = parts[1] if len(parts) > 2 else ""
        save_name = png_path.stem
        if subdir:
            return f"{prefix}/{subdir}/{save_name}"
        return f"{prefix}/{save_name}"

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, (tuple, list)):
            return [ArtifactCollector._jsonable(value) for value in obj]
        if isinstance(obj, dict):
            return {
                str(key): ArtifactCollector._jsonable(value)
                for key, value in obj.items()
            }
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)
