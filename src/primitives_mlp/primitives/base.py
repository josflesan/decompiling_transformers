import torch
from abc import ABC, abstractmethod

from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType

class Primitive(ABC):
    """Base primitive class implementing Template Method pattern to handle input tensor pre- and post-processing."""
    
    def __init__(self, type: PrimitiveType, name: str, single_input: bool = True):
        self.type = type
        self.name = name
        self.single_input = single_input
    
    def _preprocessing(self, x: torch.Tensor):
        """Ensure the input tensor has two dimensions before transform"""
        
        if self.single_input:
            # If single-input MLP, ensure its input is a 2D tensor
            
            if x.dim() == 1:
                # If only one dimension, add a batch dimension
                return x.unsqueeze(0)
            elif x.dim() == 2:
                return x
            elif x.dim() == 3:
                # If three dimensions...
                return x.flatten(end_dim=1)

        else:
            # If multi-input MLP, we need to ensure that each input is a 2D tensor
            if x[0].dim() == 1:
                return [item.unsqueeze(0) for item in x]
            elif x[0].dim() == 2:
                return x
            elif x[0].dim() == 3:
                return [item.flatten(end_dim=1) for item in x]

        raise RuntimeError("The input dimension is not supported")
    
    @abstractmethod
    def _apply_primitive(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _postprocessing(self, x: torch.Tensor, original_shape: torch.Size):
        """Map the transformed input back to its original shape"""
        return x.view(*original_shape, x.size(-1))
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess the vector (map to 2 dimensions)
        original_shape = x.size()[:-1] if self.single_input else x[0].size()[:-1]
        out = self._preprocessing(x)
        
        # Process vector according to the primitive logic
        out = self._apply_primitive(out)
        
        # Postprocess the vector (map back to original shape)
        out = self._postprocessing(out, original_shape)
        
        return out
    
    @abstractmethod
    def output_dim(self, input_dim: torch.Size) -> torch.Size:
        pass
    
    def __str__(self) -> str:
        return f"MLP Primitive: {self.name} | Single-Input: {self.single_input}"