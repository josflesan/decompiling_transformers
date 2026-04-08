import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps: int = 1000
    batch_size: int = 32
    
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

@dataclass
class TaskConfig:
    name: str
    train_length_range: List[int]
    val_length_range: List[int]
    max_test_length: int

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
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name
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