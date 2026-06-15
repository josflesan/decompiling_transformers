from dataclasses import dataclass
from typing import Optional

@dataclass
class CircuitNode:
    name: str  # TransformerLens hook name
    layer_idx: Optional[int] = None
    head_idx: Optional[int] = None
    neuron_idx: Optional[int] = None
    
    def __repr__(self):
        if self.head_idx is not None:
            return f"H{self.layer_idx}.{self.head_idx}"
        if self.neuron_idx is not None:
            return f"L{self.layer_idx}.N{self.neuron_idx}"
        return f"L{self.layer_idx}.{self.name.split('.')[-1]}"