import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from utilities.core import TaskConfig

if TYPE_CHECKING:
    from primitives_mlp.primitives.base import Primitive

# --- Config dataclasses
@dataclass
class PrimitiveSearchOutput:
    best_primitive: "Primitive"
    best_C: torch.Tensor
    best_accuracy: float

@dataclass
class MLPDataCollectorOutput:
    mlp_inputs: torch.Tensor
    mlp_outputs: torch.Tensor
    skip: bool

@dataclass
class PrimitiveConfig:
    type: str
    pow: Optional[float] = None
    center: Optional[float] = None
    threshold: Optional[float] = None

@dataclass
class MLPPrimitivesConfig:
    seed: int
    device: str
    exp_name: str
    batch_size: int
    model_path: str
    output_dir: str
    skip_convert: bool
    skip_vis: bool
    do_test: bool
    success_threshold: float
    failure_threshold: float
    task_config: TaskConfig
    mlp_primitives: List[PrimitiveConfig]
    
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    model_name: str = field(init=False)
    task_name: str = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name / 'mlp_primitives'
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True)
        
        # Save model name and task name
        self.model_name = self.model_path.split("/")[1].strip()
        self.task_name = self.model_name.split("-")[0].strip()
        
        # Set up device
        self.torch_device = (
            torch.device("cuda") if self.device == 'cuda' and torch.cuda.is_available()
            else torch.device("mps") if self.device == 'mps' and torch.backends.mps.is_available()
            else torch.device("cpu")
        )