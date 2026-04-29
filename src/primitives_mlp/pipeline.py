import json
import re
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from pathlib import Path
from typing import Any, Dict

from utilities.logger import setup_logger
from utilities.metrics_logger import MetricsLogger

from data.CustomCollator import CustomCollator
from tasks.registry import get_task

from pruning.core.hooks import GPT2QKHooks
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from primitives_mlp.core.PrimitiveSearchEngine import PrimitiveSearchEngine
from primitives_mlp.core.MLPDataCollector import MLPDataCollector
from primitives_mlp.utilities.mlp_primitive_utils import get_primitives
from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPPrimitivesConfig, PrimitiveSearchOutput, MLPDataCollectorOutput
from utilities.core import int_key_hook

class MLPPrimitivePipeline:
    
    def __init__(self, config: MLPPrimitivesConfig):
        self.config: MLPPrimitivesConfig = config
        self.all_primitives = get_primitives(self.config)
        
        self.logger = setup_logger(config.full_output_dir, name='mlp_primitives')
        self.metrics_logger = MetricsLogger(config.full_output_dir)
        
        # Setup arguments
        self.pruning_path = ""
        self.model = None
        self.orig_model = None
        self.oa_vecs = None
        self.dataloader = None
        self.hooked_model = None
        self.tokenizer = None
        
        # Converted MLP
        self.converted_mlp = {}
        
        self.logger.info("MLP Primitive Pipeline initialized")
    
    def _setup(self):
        """Loads optimal model, OA vecs and dataloaders for the data collection"""
        
        # 1. Load the pruning output file
        pruning_path = Path(f"src/out/{self.config.exp_name}/pruning/stage3/output.json")
        assert pruning_path.exists()
        with open(pruning_path) as f:
            output_dict = json.load(f)
        assert "result_patching_config_global_iteration_2" in output_dict
        self.pruning_path = pruning_path.parent
        
        # 2. Get optimal ablation vectors from pruning output
        self.oa_vecs: OptimalQueryBiasVectors = torch.load(
            self.pruning_path / "oa_vecs.pt", 
            map_location=self.config.torch_device,
            weights_only=False
        )
        self.oa_vecs.requires_grad_(False)
        
        # 3. Load models
        self.model = GPT2LMHeadModel.from_pretrained(self.config.model_path).to(self.config.torch_device)
        self.orig_model = GPT2LMHeadModel.from_pretrained(self.config.model_path).to(self.config.torch_device)
        self.model.eval()
        self.orig_model.eval()
        
        # 4. Get dataloader
        task = get_task(self.config.task_config.name, self.config.task_config)
        self.tokenizer, dataset = task.build()
        collator = CustomCollator(self.tokenizer.pad_token_id)
        self.dataloader = DataLoader(dataset['train'], batch_size=self.config.batch_size, shuffle=False, collate_fn=collator)
        
        # 5. Load hooked model
        with open(pruning_path) as f:
            config = json.load(f, object_hook=int_key_hook)["result_patching_config_global_iteration_2"]
        self.logger.info(config)
        
        self.hooked_model = GPT2QKHooks(
            model=self.model,
            config=config,
            mapping_to_param_idx=defaultdict(lambda : 0),
            split_mlp=hasattr(self.oa_vecs, "mlps"),
            logger=self.logger
        )

    def run(self) -> Dict[Any, Any]:
        """
        Runs the entire MLP Primitive Replacement pipeline.
        """
        
        # If we are using an already converted model, return it
        if self.config.skip_convert:
            return torch.load(self.config.full_output_dir / "converted_mlp.pt")
        
        # 0. Run setup
        self._setup()
        
        num_layers = len(self.hooked_model.model.transformer.h)
        config = self.hooked_model.config
        curr_path = 0
        num_paths = sum([len(config[layer]['mlp']) for layer in range(num_layers)])
        for layer in range(num_layers):
            self.logger.info(f"---------- LAYER {layer + 1} ------------")
            
            # For every input to the current layer's MLP...
            for mlp_inp in config[layer]["mlp"]:
                # Identify the relevant path
                path = f"mlp-{layer}-{mlp_inp}"
                self.logger.info(f"Converting: {path}")
                
                # Keep track of primitive replacement progress
                self.metrics_logger.log(
                    task='primitive_replacement',
                    path_idx = curr_path + 1,
                    total_paths = num_paths
                )
                
                # 1. Collect data for primitive replacement (input, output pairs)
                data_collector = MLPDataCollector(
                    hooked_model=self.hooked_model,
                    converted_mlp=self.converted_mlp,
                    dataloader=self.dataloader,
                    oa_vecs=self.oa_vecs,
                    path=path,
                    metrics_logger=self.metrics_logger,
                    logger=self.logger
                )
                data_collector_output: MLPDataCollectorOutput = data_collector.collect(layer, mlp_inp)
                if data_collector_output.skip:
                    # Input dependency with unconverted MLP
                    continue
                
                # 2. Run primitive search
                search_engine = PrimitiveSearchEngine(
                    config=self.config,
                    hooked_model=self.hooked_model,
                    original_model=self.orig_model,
                    converted_mlp=self.converted_mlp,
                    oa_vecs=self.oa_vecs,
                    dataloader=self.dataloader,
                    all_primitives=self.all_primitives,
                    metrics_logger=self.metrics_logger,
                    logger=self.logger
                )
                search_output: PrimitiveSearchOutput = search_engine.search(path, layer, data_collector_output.mlp_inputs, data_collector_output.mlp_outputs)
                self.logger.info(f"Best Primitive: {search_output.best_primitive.name} | Accuracy: {search_output.best_accuracy:.2f}")
                
                failed = search_output.best_accuracy < self.config.failure_threshold
                self.metrics_logger.log(
                    task='primitive_search',
                    path=path,
                    failed=failed,
                    best_primitive=search_output.best_primitive.name,
                    best_primitive_accuracy=search_output.best_accuracy
                )
                
                if failed:
                    self.logger.info(f"Unable to convert MLP: Low Accuracy ({search_output.best_accuracy:.2f})")
                else:
                    self.converted_mlp[path] = search_output
                
                curr_path += 1
        
        # 3. Save the converted model
        torch.save(self.converted_mlp, self.config.full_output_dir / "converted_mlp.pt")
        self.logger.info(f"Converted MLP saved at {self.config.full_output_dir / 'converted_mlp.pt'}")
        
        #TODO: LogitLens + Visualization of converted MLP
        
        self.logger.info("Primitive MLP Replacement Complete!")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])
        
        return self.converted_mlp