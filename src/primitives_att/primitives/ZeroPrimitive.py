import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("zero")
class ZeroPrimitive(Primitive):
    """
    The Zero Primitive returns a matrix which ignores all of the tokens (i.e. no effect on the logits or the project
    operation). It is thus similar to the no-op primitive in the MLP primitive replacement logic.
    """
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape
    ):
        super().__init__(domain, shape)
        self.info = PrimitiveInfo(
            name="zero",
            rasp_op="uniform selection",
            has_default_scalar=True,
            is_only_token=False,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        if shape_left is None:
            matrix = torch.zeros(shape_right)
            return matrix

        matrix = torch.zeros(shape_left, shape_right)
        return matrix

    def __str__(self):
        return f"Zero | {self.domain} | {self.shape}"
