import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.registry import register
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType

@register("erase")
class ErasePrimitive(Primitive):
    """
    The erase primitive "deletes" the input by converting it to all 1s
    """
    
    def __init__(self, type: PrimitiveType, name: str, single_input: bool = True):
        super().__init__(type, name, single_input)
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        if self.single_input:
            out = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        else:
            out = torch.ones(x[0].size(0), 1, device=x.device, dtype=x.dtype)
        
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], 2])