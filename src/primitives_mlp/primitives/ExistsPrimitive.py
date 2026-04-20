import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.primitives.registry import register

@register("exists")
class ExistsPrimitive(Primitive):
    """
    The exists primitive is asking if a component is non-negligible using a threshold
    """
    
    def __init__(self, name: str, idx: int, single_input: bool = True):
        super().__init__(name, single_input)
        self.idx = idx
        self.threshold = 0.1
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = (x[:, self.idx] > self.threshold).float()
        out = torch.stack([out, 1 - out], dim=1)
        
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], 2])