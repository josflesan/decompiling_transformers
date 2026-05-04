import torch

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.registry import register
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType

@register("combine")
class CombinePrimitive(Primitive):
    f"""
    The combine primitive collapses all of the inputs into the MLP into a single input,
    consisting of the minimum value found in all the n-tuples representing each possible
    combination of input values. For an MLP with two inputs, it thus computes...
    
    out_ij = min(a_i, b_j)
    
    The output logits are further normalized. The intuition is that this represents a logical
    AND of the inputs, and every resulting output value can be interpreted as the strength
    of interaction between two inputs.
    """
    
    def __init__(self, type: PrimitiveType, name: str, single_input: bool = False):
        super().__init__(type, name, single_input)
    
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        out = x[0]
        
        for item in x[1:]:
            # If the dimensions of the output matrix pass a threshold, break early
            if out.size(1) * item.size(1) > 10_000:
                out = torch.ones(out.size(0), 10_001, device=out.device)
                break
            
            out = torch.min(out.unsqueeze(-1), item.unsqueeze(1)).flatten(start_dim=1)
        
        out = out / out.sum(dim=-1, keepdim=True)
        return out
    
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        return torch.Size([input_dim[0], 2])