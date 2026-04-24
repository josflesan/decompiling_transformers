import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from primitives_mlp.primitives.base import Primitive

# --- Config dataclasses
@dataclass
class PrimitiveSearchOutput:
    best_primitive: "Primitive"
    best_C: torch.Tensor
    best_accuracy: float

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
    exp_path: str
    series_path: str
    output_dir: str
    skip_convert: bool
    skip_vis: bool
    do_test: bool
    range_50: bool
    success_threshold: float
    failure_threshold: float
    mlp_primitives: List[PrimitiveConfig]
    
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name / 'mlp_primitives'
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True)
        
        # Set up device
        self.torch_device = (
            torch.device("cuda") if self.device == 'cuda' and torch.cuda.is_available()
            else torch.device("mps") if self.device == 'mps' and torch.backends.mps.is_available()
            else torch.device("cpu")
        )