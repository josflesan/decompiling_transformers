import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("everyN")
class EveryNPrimitive(Primitive):
    """
    The Every-N Primitive attends to every nth position in the token sequence. It only applies to
    the Attention matrix primitives.
    """
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape,
        every_nth: int = 2,
    ):
        super().__init__(domain, shape)
        self.every_nth = every_nth
        self.info = PrimitiveInfo(
            name=f"every{self.every_nth}",
            rasp_op=f"k%{self.every_nth}==q%{self.every_nth}==0",
            has_default_scalar=True,
            is_only_token=False,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        if shape_left is None:
            raise RuntimeError("The Every-N Primitive cannot be constructed for 1D (bias) shapes")

        matrix = torch.zeros(shape_left, shape_right)
        d = min(shape_left, shape_right)
        for x in range(d // self.every_nth + d % self.every_nth != 0):
            matrix[torch.arange(x * self.every_nth, d), torch.arange(d - x * self.every_nth)] = 1.
        
        return matrix
    
    def __str__(self):
        return f"Every {self.every_nth} | {self.domain} | {self.shape}"
