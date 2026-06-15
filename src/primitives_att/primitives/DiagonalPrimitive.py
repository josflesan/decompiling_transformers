import torch
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape, PrimitiveRegistry

@PrimitiveRegistry.register("diagonal")
class DiagonalPrimitive(Primitive):
    """
    The Diagonal primitive returns a matrix with maximal values along the diagonal, k - 1 or k - 2 diagonals
    representing attention/projection patterns which 'reward' the token at the current position, the token
    immediately before the current token, or the token two positions behind the current token.
    """
    
    def __init__(
        self,
        domain: PrimitiveDomain,
        shape: PrimitiveShape,
        nth_diagonal: int = 1
    ):
        super().__init__(domain, shape)
        self.config_nth_diagonal = nth_diagonal
        self.nth_diagonal = nth_diagonal - 1
        
        # Determine D-RASP operation shorthand
        rasp_op = "k==q" if self.domain == PrimitiveDomain.ATTENTION else "inp==out"
        if self.domain == PrimitiveDomain.ATTENTION and self.nth_diagonal > 0:
            # If 2nd/3rd diagonal, adjust the D-RASP operation
            rasp_op = f"k==q-{self.nth_diagonal}"
        
        self.info = PrimitiveInfo(
            name="diag",
            rasp_op=rasp_op,
            has_default_scalar=True,
            is_only_token=False,
        )
    
    def construct(self, shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        if shape_left is None:
            raise RuntimeError("Diagonal primitive only defined for matrices!")
        
        matrix = torch.zeros(shape_left, shape_right)
        d = min(shape_left, shape_right)
        start_idx = 0
        if self.nth_diagonal == 1:
            start_idx = 1
        elif self.nth_diagonal == 2:
            start_idx = 3
        
        matrix[torch.arange(start_idx, d), torch.arange(d - start_idx)] = 1.
        return matrix

    def __str__(self):
        return f"Diagonal ({self.config_nth_diagonal}) | {self.domain} | {self.shape}"