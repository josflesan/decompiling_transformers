import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.registry import register
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType

@register("forall")
class ForallPrimitive(Primitive):
    """
    The forall primitive is asking if ALL components are non-negligible using a threshold
    """
    
    def __init__(self, type: PrimitiveType, name: str, threshold: float, single_input: bool = True):
        super().__init__(type, name, single_input)
        self.threshold = threshold
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = (x > self.threshold).float()
        out = torch.cat([out, 1 - out.sum(dim=1, keepdim=True)], dim=1)
        
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], input_dim[1] + 1])