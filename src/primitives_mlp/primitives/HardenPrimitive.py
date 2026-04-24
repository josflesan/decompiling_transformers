import torch
import torch.nn.functional as F

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.registry import register

@register("harden")
class HardenPrimitive(Primitive):
    """
    The harden primitive coverts a vector into a discrete choice. Intuitively, it can
    be seen as an MLP which is picking a token/making a decision
    """
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x.argmax(dim=-1), num_classes=x.size(-1)).float()
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return input_dim