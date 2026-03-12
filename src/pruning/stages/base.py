import logging

from abc import ABC, abstractmethod
from pathlib import Path
from transformers import GPT2LMHeadModel

from data.CustomCollator import CustomCollator
from pruning.tasks.registry import get_task
from pruning.utilities.pruning_dataclasses import PruningRunConfig
from pruning.utilities.pruning_utils import get_full_possible_config_for_pruning
from pruning.utilities.metrics_logger import MetricsLogger

class PruningStage(ABC):
    """
    Base abstract class defining a pruning stage and containing methods for stage-specific
    training, validation and graph transformation. Each pruning stage will be an child instance of
    this abstract class.
    """
    
    def __init__(
        self,
        config: PruningRunConfig,
        stage_idx: int,
        logger: logging.Logger,
        metrics_logger: MetricsLogger
    ): 
        self.config = config
        self.stage_idx = stage_idx
        self.logger = logger
        self.metrics_logger = metrics_logger
        
        # Initialize output dict and models
        self.output_config_path = config.full_output_dir / 'args.json'
        self.output_dict = {}
        self.model = GPT2LMHeadModel.from_pretrained(Path(config.model_path)).to(config.torch_device)
        self.original_model = GPT2LMHeadModel.from_pretrained(Path(config.model_path)).to(config.torch_device)
        self.model.eval()
        self.original_model.eval()
        
        # Initialize task tokenizers, datasets and collators
        self.task_config = config.task_config
        self.task = get_task(self.task_config.name, self.task_config)
        self.tokenizer, datasets = self.task.build()
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.collator = CustomCollator(self.tokenizer.pad_token_id)
        
        # Initialize model variables
        self.num_layers = len(self.model.transformer.h)
        self.num_heads_per_layer = {
            layer: self.model.transformer.h[layer].attn.num_heads for layer in range(self.num_layers)
        }
        self.model_config = get_full_possible_config_for_pruning(self.num_heads_per_layer)
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def val(self):
        pass
    
    @abstractmethod
    def transform_graph(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    def run(self):
        """
        Convenience method to run each of the steps associated with a pruning stage:
        
        1. Training
        2. Validation
        3. Graph Transformation
        4. Intermediate Model Saving
        """
        
        self.logger.info(f"1. Pruning Stage {self.stage_idx + 1} training...")
        self.train()
        
        self.logger.info(f"2. Pruning Stage {self.stage_idx + 1} validation...")
        self.val()
        
        self.logger.info(f"3. Pruning Stage {self.stage_idx + 1} graph transformation...")
        self.transform_graph()
        
        self.logger.info(f"4. Pruning Stage {self.stage_idx + 1} saving...")
        self.save()
        
        self.logger.info(f"Pruning Stage {self.stage_idx + 1} complete!\n")