import json
import re
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utilities.logger import setup_logger
from utilities.metrics_logger import MetricsLogger

from data.CustomCollator import CustomCollator
from tasks.registry import get_task

from pruning.core.hooks import GPT2QKHooks
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from primitives_mlp.core.LogitLens import LogitLens
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
        self.pruning_config = {}
        self.model = None
        self.orig_model = None
        self.oa_vecs = None
        self.dataloader = None
        self.hooked_model = None
        self.tokenizer = None
        self.single_input_mlps = True
        
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
        self.single_input_mlps = hasattr(self.oa_vecs, "mlps")
        
        # Filter primitives to only single/multi input depending on this
        filtered_primitives = []
        for prim in self.all_primitives:
            if self.single_input_mlps and prim.single_input:
                filtered_primitives.append(prim)
            elif not self.single_input_mlps and not prim.single_input:
                filtered_primitives.append(prim)
    
        self.all_primitives = filtered_primitives
        
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
        self.pruning_config = config
        self.logger.info(config)
        
        self.hooked_model = GPT2QKHooks(
            model=self.model,
            config=config,
            mapping_to_param_idx=defaultdict(lambda : 0),
            split_mlp=hasattr(self.oa_vecs, "mlps"),
            logger=self.logger
        )

    def _find_paths_to_inspect(self) -> Tuple[List[Any], List[Any]]:
        """Collects the paths that need inspecting because they couldn't be replaced by a primitive"""
        
        failed_mlps = []
        lens_qk_paths = []
        lens_unembed_paths = []
            
        # Find relevant paths
        for layer in range(len(self.pruning_config) - 1):
            for head in self.pruning_config[layer]['qk']:
                for q, k in self.pruning_config[layer]['qk'][head]:
                    if any(q.endswith(item) for item in failed_mlps) or any(k.endswith(item) for item in failed_mlps):
                        lens_qk_paths.append((q, k, layer, head))
            
            if self.single_input_mlps:
                for mlp_inp in self.pruning_config[layer]['mlp']:
                    node = f"mlp-{layer}-{mlp_inp}"
                    if node not in self.converted_mlp:
                        failed_mlps.append(node)
            else:
                # Only check if more than one input to the MLP
                if len(self.pruning_config[layer]["mlp"]) > 0:
                    node = f"mlp-{layer}"
                    if node not in self.converted_mlp:
                        failed_mlps.append(node)
        
        for lm_head_inp in self.pruning_config['lm_head']:
            if any(lm_head_inp.endswith(item) for item in failed_mlps):
                lens_unembed_paths.append(f"lm_head-{lm_head_inp}")
        
        self.logger.info(f"QK Paths to Inspect: {lens_qk_paths}")
        self.logger.info(f"Output Paths to Inspect: {lens_unembed_paths}")
        
        return lens_unembed_paths, lens_qk_paths

    def _check_dependency(self, layer) -> bool:
        """Utility to check if there is a dependency on unconverted MLPs for a multi-input MLP"""
        
        dependencies = []
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        for mlp_inp in self.hooked_model.config[layer]["mlp"]:
            if "mlp" in mlp_inp:
                inp_layer = int(re.findall(pattern, mlp_inp)[-1][4:])
                dependencies.append(self._check_dependency(inp_layer))
            else:
                dependencies.append(True)
        
        return all(dependencies)

    def _collect_and_search(
        self,
        layer: int,
        path: str,
        mlp_inp: str | None = None
    ) -> bool:
        """
        Performs data collection and primitive search for a single path

        Args:
            layer (int): the transformer layer the MLP sits in
            path (str): the name of the path we are attempting to convert
            mlp_inp (str | None): the MLP input being considered in the single-input MLP case. Defaults to None.
        """
        
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
            return True
        
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
            single_input_mlps=self.single_input_mlps,
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
        
        return False


    def run(self) -> Dict[Any, Any]:
        """
        Runs the entire MLP Primitive Replacement pipeline.
        """
        
        # 0. Run setup
        self._setup()
        
        # If we are using an already converted model, return it
        if self.config.skip_convert:
            self.converted_mlp = torch.load(self.config.full_output_dir / "converted_mlp.pt", weights_only=False)
        else: 
            num_layers = len(self.hooked_model.model.transformer.h)
            for layer in range(num_layers):
                self.logger.info(f"---------- LAYER {layer + 1} ------------")
                
                config = self.hooked_model.config
                curr_path = 0
                num_paths = sum([len(config[layer]['mlp']) for layer in range(num_layers)])
                
                if not self.single_input_mlps:
                    
                    # If all inputs to the MLP were pruned, ignore
                    if len(config[layer]["mlp"]) == 0:
                        continue
                    
                    # Only one path to keep track of
                    self.metrics_logger.log(
                        task='primitive_replacement',
                        path_idx = curr_path + 1,
                        total_paths = 1
                    )
                    
                    path = f"mlp-{layer}"
                    self.logger.info(f"Converting: {path}")
                    
                    # Check if unsatisfiable dependency exists
                    if not self._check_dependency(layer):
                        self.logger.warning("Unable to convert MLP: Dependency on Unconverted MLP")
                        continue
                    
                    self._collect_and_search(layer, path)
                
                else:
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
                        
                        # Collect and search
                        dependency = self._collect_and_search(layer, path, mlp_inp)
                        if dependency:
                            self.logger.warning(f"MLP in {path} could not be replaced due to an input dependency with an unconverted MLP")
                        
                        curr_path += 1
            
            # 3. Save the converted model
            torch.save(self.converted_mlp, self.config.full_output_dir / "converted_mlp.pt")
            self.logger.info(f"Converted MLP saved at {self.config.full_output_dir / 'converted_mlp.pt'}")
        
        # 4. Visualize MLPs that failed conversion using LogitLens
        lens = LogitLens(
            hooked_model=self.hooked_model,
            oa_vecs=self.oa_vecs,
            tokenizer=self.tokenizer,
            dataloader=self.dataloader,
            converted_mlp=self.converted_mlp,
            metrics_logger=self.metrics_logger,
            logger=self.logger
        )
        
        lens_unembed_paths, lens_qk_paths = self._find_paths_to_inspect()
        cached_data = {}
        
        # Inspect the output paths
        for unexplained_path in lens_unembed_paths:
            if self.single_input_mlps:
                inp_cache, out_cache = lens.inspect_mlp_logits(unexplained_path)
            else:
                inp_cache, out_cache = lens.inspect_multi_mlp_logits(unexplained_path)
            
            cached_data[unexplained_path] = (inp_cache, out_cache)

        # Inspect the QK paths
        for path_info in lens_qk_paths:
            if not self.single_input_mlps:
                #TODO: see if we can implement this once the full pipeline is working
                self.logger.warning("Multi-source MLP LogitLens in QK not yet implemented. Skipping...")
                break
            
            q_unexplained_path, k_unexplained_path, attn_layer_idx, attn_head_idx = path_info
            q_inp_cache, k_inp_cache, out_cache = lens.inspect_mlp_attn(
                query_path=q_unexplained_path,
                key_path=k_unexplained_path,
                attn_layer=attn_layer_idx,
                attn_head=attn_head_idx
            )
            cached_data[path_info] = (q_inp_cache, k_inp_cache, out_cache)
        
        # Save the inspected paths
        torch.save(cached_data, self.config.full_output_dir / "mlp_input_output.pt")
        self.logger.info(f"Input-Output MLP Inspection Results Saved at {self.config.full_output_dir / 'mlp_input_output.pt'}")
        
        self.logger.info("Primitive MLP Replacement Complete!")
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])
        
        return self.converted_mlp