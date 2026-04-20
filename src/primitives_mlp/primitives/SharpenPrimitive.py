import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.primitives.registry import register

@register("sharpen")
class SharpenPrimitive(Primitive):
    """
    The sharpen primitive makes the distribution more peaked, essentially
    amplifying the largest components of the input. This can be seen as a
    form of "self selection".
    
    The applied function is (x_i^n) / sum_i(x_i^n)
    """
    
    def __init__(self, name: str, pow: int, single_input: bool = True):
        super().__init__(name, single_input)
        self.pow = pow
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = x.pow(self.pow)
        out /= out.sum(dim=-1, keepdim=True)
        
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return input_dim