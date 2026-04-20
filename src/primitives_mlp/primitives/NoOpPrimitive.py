import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.primitives.registry import register

@register("noop")
class NoOpPrimitive(Primitive):
    """
    The noop primitive is the trivial identity mapping. The idea is that
    many MLPs will just be pass-throughs for their input variable
    """
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return input_dim