from pathlib import Path

from pruning.utilities.logger import setup_logger
from pruning.utilities.metrics_logger import MetricsLogger
from pruning.stages.stage1 import CausalPruningStage1
from pruning.stages.stage2 import CausalPruningStage2
# from pruning.stages.stage3 import CausalPruningStage3

STAGE_REGISTRY = {
    "stage1": CausalPruningStage1,
    "stage2": CausalPruningStage2 
}

class CausalPruningPipeline:
    
    def __init__(self, config):
        self.config = config
        
        self.logger = setup_logger(config.full_output_dir)
        self.metrics = MetricsLogger(config.full_output_dir)
        
        self.logger.info("Pipeline initialized")
        
        self.stages = []
        self._initialize_stages()
    
    def _initialize_stages(self):
        """
        Initialize all pruning stages defined in the config
        """
        
        for stage_name in self.config.pruning_stages.keys():
            if stage_name not in STAGE_REGISTRY:
                raise ValueError(f"Unknown pruning stage: {stage_name}")
        
            stage_class = STAGE_REGISTRY[stage_name]
            
            stage = stage_class(
                config=self.config,
                stage_name=stage_name,
                logger=self.logger,
                metrics_logger=self.metrics
            )
            self.stages.append(stage)

        if len(self.stages) == 0:
            raise ValueError("No pruning stages enabled in config.")

        self.logger.info(f"{len(self.stages)} pruning stages initialized.\n")
    
    def run(self):
        """
        Runs the entire pruning pipeline.
        """
        
        for stage_idx, stage in enumerate(self.stages):
            self.logger.info("=" * 60)
            self.logger.info(f"Running Stage {stage_idx + 1}: {stage.__class__.__name__}")
            self.logger.info("=" * 59 + "=\n")
            
            stage.run()
        
        self.logger.info("Causal Pruning Complete")