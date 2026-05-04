import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("decreasing")
class DecreasingPrimitive(Primitive):
    """
    The decreasing primitive returns a matrix/bias vector that gradually decreases from the first token
    position up until the last token position. This is only relevant to the Attention layers
    """
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape
    ):
        super().__init__(domain, shape)
        self.info = PrimitiveInfo(
            name="decreasing",
            rasp_op="k is first",
            has_default_scalar=True,
            is_only_token=False,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        if shape_left is None:
            matrix = (torch.arange(shape_right) + 1) / shape_right
            return matrix

        matrix = (torch.arange(shape_right) + 1).unsqueeze(0).expand(shape_left, shape_right) / shape_right
        return matrix

    def __str__(self):
        return f"Decreasing Primitive | {self.domain} | {self.shape}"
