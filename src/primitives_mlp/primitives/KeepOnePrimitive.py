import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.registry import register
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType

@register("keepone")
class KeepOnePrimitive(Primitive):
    """
    The keep one primitive preserves a single input from a multi-input MLP
    """
    
    def __init__(self, type: PrimitiveType, name: str, keep_n: int, single_input: bool = False):
        super().__init__(type, name, single_input)
        self.keep_n = keep_n
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = x[self.keepn]
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([1, input_dim[1]])