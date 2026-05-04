import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional, TYPE_CHECKING

from utilities.core import TaskConfig

if TYPE_CHECKING:
    from primitives_att.primitives.base import Primitive

# --- Config dataclasses
# @dataclass
# class PrimitiveSearchOutput:
#     best_primitive: "Primitive"
#     best_C: torch.Tensor
#     best_accuracy: float

@dataclass
class PrimitiveConfig:
    type: str
    nth_diagonal: Optional[int] = None
    special_token: Optional[str] = None
    every_nth: Optional[int] = None

@dataclass
class PrimitiveInfo:
    """Metadata about a primitive"""
    name: str
    rasp_op: Optional[str] = None
    has_default_scalar: bool = False
    is_only_token: bool = False

@dataclass
class PrimitiveRegistration:
    cls: Type['Primitive']
    params: Dict[str, Any]

@dataclass
class AttPrimitivesConfig:
    seed: int
    device: str
    exp_name: str
    batch_size: int
    learning_rate: float
    model_path: str
    output_dir: str
    skip_convert: bool
    task_config: TaskConfig
    att_primitives_matrix: List[PrimitiveConfig]
    att_primitives_bias: List[PrimitiveConfig]
    unembedding_primitives_matrix: List[PrimitiveConfig]
    unembedding_primitives_bias: List[PrimitiveConfig]
    
    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    model_name: str = field(init=False)
    task_name: str = field(init=False)
    
    def __post_init__(self):
        # Set up full output directory
        self.full_output_dir = Path(self.output_dir) / self.exp_name / 'att_primitives'
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