import torch
from typing import List

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.primitives.registry import register

@register("equal")
class EqualPrimitive(Primitive):
    """
    The equal primitive is asking if all values are nearly identical
    """
    
    def __init__(self, name: str, indices: List[int] = [], single_input: bool = True):
        super().__init__(name, single_input)
        self.indices = indices
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:, self.indices]
        out = ((out.max(dim=1)[0] - out.min(dim=1)[0]) < 0.01).float()
        out = torch.stack([out, 1 - out], dim=1)
        
        return out
    
    def set_indices(self, indices):
        self.indices = indices
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], 2])