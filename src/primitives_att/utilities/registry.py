from typing import Callable, Type, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from primitives_att.primitives.base import Primitive
from primitives_att.utilities.att_primitive_dataclasses import AttPrimitivesConfig, PrimitiveRegistration

class PrimitiveDomain(Enum):
    """Which layer the primitive applies to"""
    ATTENTION = "ATTENTION"
    UNEMBEDDING = "UNEMBEDDING"

class PrimitiveShape(Enum):
    """Whether the primitive refers to a bias vector (1D) or matrix (2D)"""
    BIAS = "BIAS"
    MATRIX = "MATRIX"

class PrimitiveRegistry:
    """Singleton registry for all primitives."""

    _registry: Dict[str, Type['Primitive']] = {}
    
    @classmethod
    def register(cls, primitive_name: str):
        """Decorator to register a primitive class."""
        def decorator(primitive_cls: Type['Primitive']) -> Type['Primitive']:
            
            # Populate name mapping
            cls._registry[primitive_name] = primitive_cls
            
            return primitive_cls
        return decorator
    
    @classmethod
    def load_primitives_from_config(cls, config: AttPrimitivesConfig) -> Dict[Tuple[PrimitiveDomain, PrimitiveShape], List['Primitive']]:
        """Load primitives based on config, filtering by what's specified in the config."""
        result = {}
        
        config_map = [
            (config.att_primitives_matrix, PrimitiveDomain.ATTENTION, PrimitiveShape.MATRIX),
            (config.att_primitives_bias, PrimitiveDomain.ATTENTION, PrimitiveShape.BIAS),
            (config.unembedding_primitives_matrix, PrimitiveDomain.UNEMBEDDING, PrimitiveShape.MATRIX),
            (config.unembedding_primitives_bias, PrimitiveDomain.UNEMBEDDING, PrimitiveShape.BIAS)
        ]
        
        for prim_list, domain, shape in config_map:
            primitives = []
            for prim_cfg in prim_list:
                prim_cls = cls._registry.get(prim_cfg.type)
                if prim_cls is None:
                    raise ValueError(f"Unknown primitive type: {prim_cfg.type}")
            
                # Extract params from config
                params = {k: v for k, v in prim_cfg.__dict__.items() if k != 'type' and v is not None}
                primitives.append(prim_cls(domain=domain, shape=shape, **params))

            result[(domain, shape)] = primitives
        
        return result
