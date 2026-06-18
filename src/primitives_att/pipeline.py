"""
Pipeline for replacing attention and unembedding interactions with primitives.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

import primitives_att.primitives  # noqa: F401 — register primitives
from utilities.core import LossModule, int_key_hook
from primitives_att.core.PrimitiveExecutionEngine import PrimitiveExecutionEngine
from primitives_att.core.PrimitiveSearchEngine import PrimitiveSearchEngine
from primitives_att.utilities.att_primitive_dataclasses import (
    AttentionInteraction,
    AttPrimitiveSearchOutput,
    AttPrimitivesConfig,
    LogitsInteraction,
)
from primitives_att.utilities.registry import PrimitiveRegistry
from primitives_att.utilities.search_logging import count_converted, count_interactions, interaction_eval
from tasks.registry import get_task
from utilities.logger import setup_logger
from utilities.metrics_logger import MetricsLogger

from pruning.core.hooks import GPT2QKHooks
from pruning.core.mask_samplers import QKMaskSampler
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors


class AttPrimitivePipeline:
    def __init__(self, config: AttPrimitivesConfig):
        self.config = config
        self.candidate_primitives = PrimitiveRegistry.load_primitives_from_config(config)

        self.logger = setup_logger(config.full_output_dir, name="att_primitives")
        self.metrics_logger = MetricsLogger(config.full_output_dir)

        self.pruning_path: Path | None = None
        self.pruning_config: dict = {}
        self.model: GPT2LMHeadModel | None = None
        self.orig_model: GPT2LMHeadModel | None = None
        self.oa_vecs: OptimalQueryBiasVectors | None = None
        self.dataloader: DataLoader | None = None
        self.hooked_model: GPT2QKHooks | None = None
        self.tokenizer = None
        self.loss_module = None
        self.mask_sampler: QKMaskSampler | None = None
        self.converted_mlp: dict = {}
        self.converted_att: AttPrimitiveSearchOutput | None = None

        self.logger.info("Attention Primitive Pipeline initialized")

    def _setup(self) -> None:
        pruning_path = Path(f"src/out/{self.config.exp_name}/pruning/stage3/output.json")
        assert pruning_path.exists(), f"Missing pruning output at {pruning_path}"
        self.pruning_path = pruning_path.parent

        with open(pruning_path) as f:
            output_dict = json.load(f)
        assert "result_patching_config_global_iteration_2" in output_dict

        self.oa_vecs = torch.load(
            self.pruning_path / "oa_vecs.pt",
            map_location=self.config.torch_device,
            weights_only=False,
        )
        self.oa_vecs.requires_grad_(False)

        converted_mlp_path = (
            Path(self.config.output_dir)
            / self.config.exp_name
            / "mlp_primitives"
            / "converted_mlp.pt"
        )
        assert converted_mlp_path.exists(), (
            f"Missing converted MLP at {converted_mlp_path}. Run MLP primitive conversion first."
        )
        self.converted_mlp = torch.load(converted_mlp_path, weights_only=False)

        self.model = GPT2LMHeadModel.from_pretrained(self.config.model_path).to(
            self.config.torch_device
        )
        self.orig_model = GPT2LMHeadModel.from_pretrained(self.config.model_path).to(
            self.config.torch_device
        )
        self.model.eval()
        self.orig_model.eval()

        task = get_task(self.config.task_config.name, self.config.task_config)
        self.tokenizer, dataset = task.build()
        self.loss_module = LossModule.from_dataset(dataset["train"], self.tokenizer)
        collator = self.loss_module.collator()
        self.dataloader = DataLoader(
            dataset["train"],
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        with open(pruning_path) as f:
            self.pruning_config = json.load(f, object_hook=int_key_hook)[
                "result_patching_config_global_iteration_2"
            ]
        
        is_config_empty = True
        loaded_config = self.pruning_config
        current_config = {}
        for layer in range(len(self.model.transformer.h)):
            current_config[layer] = {}
            for tp in ["v", "k", "qk"]:
                current_config[layer][tp] = {}
                for head in range(self.model.transformer.h[layer].attn.num_heads):
                    current_config[layer][tp][head] = loaded_config[layer][tp][head]
                    if len(current_config[layer][tp][head]) > 0:
                        is_config_empty = False
                    if tp == "qk":
                        current_config[layer][tp][head] = list(map(tuple, current_config[layer][tp][head]))
                        if len(current_config[layer][tp][head]) > 0:
                            is_config_empty = False
            
            current_config[layer]["mlp"] = loaded_config[layer]["mlp"]
            if len(current_config[layer]["mlp"]) > 0:
                is_config_empty = False
        current_config["lm_head"] = loaded_config["lm_head"]
        if len(current_config["lm_head"]) > 0:
            is_config_empty = False
        if is_config_empty:
            self.logger.info(f"{self.pruning_path}: config is empty, skipping")
        
        self.pruning_config = current_config
        self.logger.info(self.pruning_config)

        self.hooked_model = GPT2QKHooks(
            model=self.model,
            config=self.pruning_config,
            mapping_to_param_idx=defaultdict(lambda: 0),
            split_mlp=hasattr(self.oa_vecs, "mlps"),
            logger=self.logger,
        )
        self.mask_sampler = QKMaskSampler(self.pruning_config).to(self.config.torch_device)

    def _build_interaction_map(self) -> Dict[Any, Any]:
        assert self.hooked_model is not None

        interaction_map: Dict[Any, Any] = {}
        num_layers = len(self.hooked_model.model.transformer.h)
        for layer in range(num_layers):
            interaction_map[layer] = {}
            num_heads = self.hooked_model.model.transformer.h[layer].attn.num_heads
            for head in range(num_heads):
                interaction_map[layer][head] = {
                    AttentionInteraction(
                        activation_name_to_keep_q=item[0],
                        activation_name_to_keep_k=item[1],
                    ): None
                    for item in self.pruning_config[layer]["qk"][head]
                }
                interaction_map[layer][head].update(
                    {
                        AttentionInteraction(activation_name_to_keep_k=k): None
                        for k in self.pruning_config[layer]["k"][head]
                    }
                )

        interaction_map["lm_head"] = {
            LogitsInteraction(activation_name_to_keep=item): None
            for item in self.pruning_config["lm_head"] + ["vocab_bias"]
        }
        return interaction_map

    @staticmethod
    def _serialize_primitives(interaction_map: Dict[Any, Any]) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {"layers": {}, "lm_head": []}
        for layer in sorted(k for k in interaction_map if isinstance(k, int)):
            serialized["layers"][str(layer)] = {}
            for head in interaction_map[layer]:
                qk_interactions = []
                k_interactions = []
                
                for interaction, primitive in interaction_map[layer][head].items():
                    item = asdict(interaction)
                    item["primitive"] = str(primitive) if primitive is not None else None
                    if primitive is not None:
                        item["scaling_factor"] = primitive.scaling_factor
                        if primitive.primitive is not None:
                            item["predefined_primitive"] = str(primitive.primitive)
                        if primitive.special_primitive is not None:
                            item["predefined_special_primitive"] = str(
                                primitive.special_primitive
                            )
                    if interaction.activation_name_to_keep_q is None:
                        k_interactions.append(item)
                    else:
                        qk_interactions.append(item)
                
                serialized["layers"][str(layer)][str(head)] = {
                    "qk_interactions": qk_interactions,
                    "k_interactions": k_interactions,
                }

        for interaction, primitive in interaction_map["lm_head"].items():
            item = asdict(interaction)
            item["primitive"] = str(primitive) if primitive is not None else None
            if primitive is not None:
                item["scaling_factor"] = primitive.scaling_factor
                if primitive.primitive is not None:
                    item["predefined_primitive"] = str(primitive.primitive)
                if primitive.special_primitive is not None:
                    item["predefined_special_primitive"] = str(primitive.special_primitive)
            serialized["lm_head"].append(item)

        return serialized

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): AttPrimitivePipeline._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AttPrimitivePipeline._to_jsonable(v) for v in obj]
        return obj

    def _save_heatmaps(self, execution_engine: PrimitiveExecutionEngine) -> None:
        assert self.hooked_model is not None
        assert self.converted_att is not None

        self.hooked_model.save_matrices_path = str(self.config.full_output_dir / "heatmaps")
        self.hooked_model.saved_matrices = set()
        self.hooked_model.save_matrices = True
        full_metrics = execution_engine.evaluate(
            self.converted_att.primitives,
            num_steps=self.config.num_test_steps,
            batch_size=self.config.batch_size,
        )
        self.hooked_model.save_matrices = False
        self.converted_att.stats.setdefault("acc", {})["after_full_batch"] = full_metrics.acc
        self.converted_att.stats.setdefault("kl", {})["after_full_batch"] = full_metrics.kl
        self.converted_att.stats.setdefault("acc_match", {})[
            "after_full_batch"
        ] = full_metrics.acc_match
        self.converted_att.stats.setdefault("task_loss", {})[
            "after_full_batch"
        ] = full_metrics.task_loss

    def run(self) -> Dict[Any, Any]:
        self._setup()
        assert self.hooked_model is not None
        assert self.orig_model is not None
        assert self.oa_vecs is not None
        assert self.dataloader is not None
        assert self.mask_sampler is not None
        assert self.tokenizer is not None
        assert self.loss_module is not None

        if self.config.skip_convert:
            self.converted_att = torch.load(
                self.config.full_output_dir / "converted_att.pt", weights_only=False
            )
            execution_engine = PrimitiveExecutionEngine(
                hooked_model=self.hooked_model,
                original_model=self.orig_model,
                dataloader=self.dataloader,
                mask_sampler=self.mask_sampler,
                oa_vecs=self.oa_vecs,
                tokenizer=self.tokenizer,
                converted_mlp=self.converted_mlp,
                loss_module=self.loss_module,
                logger=self.logger,
                metrics_logger=self.metrics_logger,
            )
            self._save_heatmaps(execution_engine)
            return self.converted_att.primitives

        interaction_map = self._build_interaction_map()

        execution_engine = PrimitiveExecutionEngine(
            hooked_model=self.hooked_model,
            original_model=self.orig_model,
            dataloader=self.dataloader,
            mask_sampler=self.mask_sampler,
            oa_vecs=self.oa_vecs,
            tokenizer=self.tokenizer,
            converted_mlp=self.converted_mlp,
            loss_module=self.loss_module,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
        )

        search_engine = PrimitiveSearchEngine(
            config=self.config,
            execution_engine=execution_engine,
            candidate_primitives=self.candidate_primitives,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
        )
        self.converted_att = search_engine.search(copy.deepcopy(interaction_map))

        self._save_heatmaps(execution_engine)

        torch.save(self.converted_att, self.config.full_output_dir / "converted_att.pt")

        output = {
            "primitives": self._serialize_primitives(self.converted_att.primitives),
            "config": self._to_jsonable(self.pruning_config),
            "accuracy": self.converted_att.stats["acc"],
            "kl": self.converted_att.stats["kl"],
            "acc_match": self.converted_att.stats["acc_match"],
            "task_loss": self.converted_att.stats["task_loss"],
            "eval": asdict(self.converted_att.eval),
        }
        with open(self.config.full_output_dir / "output.json", "w") as f:
            json.dump(output, f, indent=2)

        self.metrics_logger.log(
            task="att_pipeline_complete",
            acc_match_before=self.converted_att.stats["acc_match"]["before"],
            acc_match_after=self.converted_att.stats["acc_match"]["after"],
            acc_match_after_full_batch=self.converted_att.stats["acc_match"].get("after_full_batch"),
            converted_count=count_converted(self.converted_att.primitives),
            total_count=count_interactions(self.converted_att.primitives),
            fully_replaced=interaction_eval(self.converted_att.primitives).is_fully_replaced[0],
        )

        self.logger.info(f"Converted attention primitives saved to {self.config.full_output_dir}")
        self.logger.info("Attention Primitive Replacement complete!")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])

        return self.converted_att.primitives
