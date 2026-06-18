import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("toSpecial")
class ToSpecialPrimitive(Primitive):
    """
    The To-Special Primitive replaces either the Attention or Unembedding layer with a simple matrix which
    attends to/projects only one of the three special symbols (BOS, EOS, SEP). To be specific, we consider
    primitives:
    
    - ToBOS: for Attention Matrix, Attention Bias, Unembedding Matrix and Unembedding Bias
    - ToEOS: for Attention Matrix, Unembedding Matrix and Unembedding Bias
    - ToSEP: for Attention Matrix and Attention Bias
    """
    
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape,
        special_token: str = 'bos'
    ):
        super().__init__(domain, shape)
        self.special_token_str = special_token
        self.info = PrimitiveInfo(
            name=f"to{special_token.upper()}",
            rasp_op=f"k=={special_token.upper()}" if self.domain == PrimitiveDomain.ATTENTION else f"out=={special_token.upper()}",
            has_default_scalar=True,
            is_only_token=True,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        # Find special token identity
        special_token = None
        match self.special_token_str:
            case "bos":
                special_token = tokenizer.bos_token_id
            case "eos":
                special_token = tokenizer.eos_token_id
            case "sep":
                special_token = tokenizer.sep_token_id
            case _:
                raise RuntimeError(f"The token {self.special_token_str} is not a special token! Pick one from BOS, EOS, SEP.")
        
        if shape_left is None:
            matrix = torch.zeros(shape_right)
            matrix[special_token] = 1
            return matrix

        matrix = torch.zeros(shape_left, shape_right)
        matrix[:, special_token] = 1.
        return matrix

    def __str__(self):
        return f"To {self.special_token_str.upper()} | {self.domain} | {self.shape}"