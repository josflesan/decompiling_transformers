import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional, TYPE_CHECKING

from utilities.core import TaskConfig

if TYPE_CHECKING:
    from primitives_att.primitives.base import Primitive

# --- Interaction and result dataclasses

@dataclass(frozen=True)
class AttentionInteraction:
    activation_name_to_keep_k: str
    activation_name_to_keep_q: Optional[str] = None


@dataclass(frozen=True)
class LogitsInteraction:
    activation_name_to_keep: str


@dataclass
class AbstractPrimitive:
    name: str
    primitive: Optional["Primitive"] = None
    special_primitive: Optional["Primitive"] = None
    scaling_factor: Optional[float] = None
    replacement_matrix: Optional[Any] = None  # MatrixRounder in Phase 2


@dataclass
class PrimitiveEval:
    zero_parameters: List[int]
    total_parameters: List[int]
    is_fully_replaced: List[bool]


@dataclass
class EvalMetrics:
    acc: float
    kl: float
    acc_match: float
    task_loss: float


@dataclass
class AttPrimitiveSearchOutput:
    primitives: Dict[Any, Any]
    eval: PrimitiveEval
    stats: Dict[str, Dict[str, Optional[float]]]


# --- Config dataclasses

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
    cls: Type["Primitive"]
    params: Dict[str, Any]


@dataclass
class RoundConfig:
    training_lambda: float = 0.01
    num_steps: int = 2000
    lr: float = 0.0001
    two_stages: bool = True
    train_scalar: bool = False
    round_loss_coef: float = 1.0
    scalar_loss_coef: float = 1.0
    to_zero_loss_coef: float = 1.0
    average_over_pixels: bool = True
    to_zero_loss_penalty: str = "l1"
    round_loss_penalty: str = "l1"


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
    search_method: str = "greedy_search"
    threshold_on_acc: float = 0.95
    only_default_scalars: bool = True
    num_test_steps: int = 10
    eval_batch_size: int = 120
    failure_threshold: float = 0.9
    round: Optional[RoundConfig] = None

    full_output_dir: Path = field(init=False)
    torch_device: torch.device = field(init=False)
    model_name: str = field(init=False)
    task_name: str = field(init=False)

    def __post_init__(self):
        self.full_output_dir = Path(self.output_dir) / self.exp_name / "att_primitives"
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True)

        self.model_name = self.model_path.split("/")[1].strip()
        self.task_name = self.model_name.split("-")[0].strip()

        self.torch_device = (
            torch.device("cuda")
            if self.device == "cuda" and torch.cuda.is_available()
            else torch.device("mps")
            if self.device == "mps" and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
