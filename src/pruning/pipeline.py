from pathlib import Path

from pruning.stages.stage1 import CausalPruningStage1
# from pruning.stages.stage2 import CausalPruningStage2
# from pruning.stages.stage3 import CausalPruningStage3

STAGE_REGISTRY = {
    "stage1": CausalPruningStage1
}

class CausalPruningPipeline:
    
    def __init__(self, config):
        self.config = config
        self.stages = []
        self._initialize_stages()
    
    def _initialize_stages(self):
        """
        Initialize all pruning stages defined in the config
        """
        
        for stage_idx, stage_cfg in enumerate(self.config.pruning_stages):
            stage_name = stage_cfg.name
            
            if stage_name not in STAGE_REGISTRY:
                raise ValueError(f"Unknown pruning stage: {stage_name}")
        
            stage_class = STAGE_REGISTRY[stage_name]
            
            stage = stage_class(config=self.config, stage_idx=stage_idx)
            self.stages.append(stage)

        if len(self.stages) == 0:
            raise ValueError("No pruning stages enabled in config.")

        #TODO: change to logging
        print(f"{len(self.stages)} pruning stages initialized.")
    
    def run(self):
        """
        Runs the entire pruning pipeline.
        """
        
        #TODO: Logging
        
        for stage_idx, stage in enumerate(self.stages):
            print()
            print("=" * 60)
            print(f"Running Stage {stage_idx + 1}: {stage.__class__.__name__}")
            print("=" * 60)
            print()
            
            stage.run()
    
        #TODO: Logging