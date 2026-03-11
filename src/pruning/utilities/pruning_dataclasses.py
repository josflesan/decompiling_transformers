import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps: int = 1000
    batch_size: int = 32
    
@dataclass
class StageConfig:
    name: str
    linear_ln: Optional[bool] = False
    lamb: float = 1e-3
    num_steps: int = 1000

@dataclass
class TaskConfig:
    name: str
    batch_size: int
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
    stage_config: StageConfig
    init_sample_param: Optional[float] = None
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True)
            
        # Set up device
        self.torch_device = (
            torch.device("cuda") if self.device == 'cuda' and torch.cuda.is_available()
            else torch.device("mps") if self.device == 'mps' and torch.backends.mps.is_available()
            else torch.device("cpu")
        )