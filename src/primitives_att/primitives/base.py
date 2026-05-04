import torch
from abc import ABC, abstractmethod
from typing import Optional

from data.CustomTokenizer import CustomTokenizer
from primitives_att.utilities.registry import PrimitiveDomain, PrimitiveShape
from primitives_att.utilities.att_primitive_dataclasses import PrimitiveInfo

class Primitive(ABC):
    
    def __init__(self, domain: PrimitiveDomain, shape: PrimitiveShape):
        self.domain = domain
        self.shape = shape
    
    @abstractmethod
    def construct(shape_left: Optional[int], shape_right: int, tokenizer: CustomTokenizer) -> torch.Tensor:
        """Construct the primitive matrix"""
        pass
