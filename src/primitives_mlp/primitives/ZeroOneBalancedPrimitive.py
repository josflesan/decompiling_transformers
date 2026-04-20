import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.primitives.registry import register

@register("zeroone")
class ZeroOneBalancedPrimitive(Primitive):
    """
    The 01-balanced primitive takes a 2D binary vector and determines the majority
    bit in the representation. Intuitively, it is usually computing some kind of
    conditional such as "is 1 > 0".
    
    In essence, the primitive is trying to compute a soft version of
    
    out = sign(x_1 - x_0)
    
    where x_1 is the number of 1s and x_0 is the number of 0s
    """
    
    def __init__(self, name: str, pow: float, center: float, single_input: bool = True):
        super().__init__(name, single_input)
        
        self.pow = pow
        self.center = center  # represents the value at which the system can be considered to be "balanced"
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:, 1] - x[:, 0] - self.center
        out = torch.stack([out, -out], dim=1).clamp(min=0).pow(self.pow)
        out = torch.cat([out, 1 - out.sum(dim=1, keepdim=True)], dim=1)
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], input_dim[0]])