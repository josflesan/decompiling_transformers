import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utilities.core import TaskConfig

@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps: int = 1000
    batch_size: int = 32
    
@dataclass
class LambSearchConfig:
    probe_lambdas: Optional[List[float]] = None
    probe_num_steps: Optional[int] = None
    max_probes: int = 3
    enable_refinement: bool = True
    probe_scale_factor: float = 10.0


@dataclass
class StageConfig:
    train_batch_size: int
    test_batch_size: int
    num_repeat: int
    lamb: float = 1e-3
    num_steps: int = 1000
    linear_ln: Optional[bool] = False
    mini_batch_size: Optional[int] = 0
    split_mlp: Optional[bool] = False
    auto_lamb: bool = False
    lamb_search: Optional[LambSearchConfig] = None

@dataclass
class PruningRunConfig:
    seed: int
    device: str
    exp_name: str
    output_dir: str
    model_path: str
    task_config: TaskConfig
    pruning_stages: Dict[str, StageConfig]
    lr_sampler_for_pruning: float
    lr_ln_var_for_pruning: float
    lr_oa_for_pruning: float
    lr_mlp_for_pruning: Optional[float] = 0.001
    
    init_sample_param: Optional[float] = None
    baseline_loss: Optional[float] = None
    baseline_acc: Optional[float] = None
    relative_gap: float = 0.9
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name / 'pruning'
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True)
        
        # Create subfolders
        for i in range(3):
            stage_path = self.full_output_dir / f'stage{i + 1}'
            stage_path.mkdir(exist_ok=True)
            
        # Set up device
        self.torch_device = (
            torch.device("cuda") if self.device == 'cuda' and torch.cuda.is_available()
            else torch.device("mps") if self.device == 'mps' and torch.backends.mps.is_available()
            else torch.device("cpu")
        )