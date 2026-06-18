from __future__ import annotations

import json
from pathlib import Path

import torch

from primitives_att.utilities.att_primitive_dataclasses import AttPrimitiveSearchOutput
from rasp.utilities.rasp_dataclasses import RaspInputs, RaspRunConfig
from rasp.utilities.rasp_utils import convert_keys_to_int


class InputLoader:
    def __init__(self, config: RaspRunConfig):
        self.config = config
        self.exp_root = Path(config.output_dir) / config.exp_name

    def _require(self, path: Path, label: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label} at {path}. Run upstream pipeline stages first."
            )
        return path

    def load(self) -> RaspInputs:
        pruning_dir = self.exp_root / "pruning" / "stage3"
        mlp_dir = self.exp_root / "mlp_primitives"
        att_dir = self.exp_root / "att_primitives"

        pruning_json = self._require(pruning_dir / "output.json", "pruning stage3 output")
        oa_vecs_path = self._require(pruning_dir / "oa_vecs.pt", "pruning oa_vecs")
        converted_mlp_path = self._require(mlp_dir / "converted_mlp.pt", "converted MLP")
        converted_att_path = self._require(att_dir / "converted_att.pt", "converted attention")

        with open(pruning_json) as f:
            pruning_output = json.load(f)

        key = "result_patching_config_global_iteration_2"
        if key not in pruning_output:
            raise KeyError(f"{key} not found in {pruning_json}")

        pruning_config = convert_keys_to_int(pruning_output[key])
        oa_vecs = torch.load(oa_vecs_path, map_location="cpu", weights_only=False)
        converted_mlp = torch.load(converted_mlp_path, map_location="cpu", weights_only=False)
        converted_att: AttPrimitiveSearchOutput = torch.load(
            converted_att_path, map_location="cpu", weights_only=False
        )

        mlp_io_path = mlp_dir / "mlp_input_output.pt"
        mlp_input_output = None
        if mlp_io_path.exists():
            mlp_input_output = torch.load(mlp_io_path, map_location="cpu", weights_only=False)

        pruning_metrics = {
            "acc_match": pruning_output.get("acc_match"),
            "acc_task": pruning_output.get("acc_task"),
            "kl_div": pruning_output.get("kl_div"),
            "task_loss": pruning_output.get("task_loss"),
        }

        return RaspInputs(
            pruning_config=pruning_config,
            pruning_metrics=pruning_metrics,
            split_mlp=hasattr(oa_vecs, "mlps"),
            converted_mlp=converted_mlp,
            interaction_map=converted_att.primitives,
            att_stats=converted_att.stats,
            mlp_input_output=mlp_input_output,
        )
