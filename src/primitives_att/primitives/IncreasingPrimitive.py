import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("increasing")
class IncreasingPrimitive(Primitive):
    """
    The increasing primitive returns a matrix/bias vector that gradually increases from the first token
    position up until the last token position. This is only relevant to the Attention layers
    """
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape
    ):
        super().__init__(domain, shape)
        self.info = PrimitiveInfo(
            name="increasing",
            rasp_op="k is last",
            has_default_scalar=True,
            is_only_token=False,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        if shape_left is None:
            matrix = (torch.arange(shape_right, 0, -1)) / shape_right
            return matrix

        matrix = (torch.arange(shape_right, 0, -1)).unsqueeze(0).expand(shape_left, shape_right) / shape_right
        return matrix
    
    def __str__(self):
        return f"Increasing | {self.domain} | {self.shape}"
