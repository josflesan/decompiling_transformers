'''
Orchestrator that runs the greedy search. Should have the following

- search_for_mlp(mlp_node): iterates through library, uses RegressionSolver
and maintains the state of the model for accuracy validation

- Contains specific threshold logic (0.92 for identity, 0.9 for failure)
'''

import logging
import torch
from typing import List

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType, build_primitive

class PrimitiveSearchEngine:
    
    def __init__(
        self,
        logger: logging.Logger,
        all_primitives: List[Primitive],
        success_threshold: float = 0.92,
        failure_threshold: float = 0.9
    ):
        self.logger = logger
        self.all_primitives = all_primitives
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
    
    def _find_optimalC(
        self,
        primitive: Primitive,
        mlp_inputs: torch.Tensor,
        mlp_outputs: torch.Tensor
    ) -> torch.Tensor | None:
        # Compute the primitive outputs
        Y = primitive.apply(mlp_inputs)
        
        try:
            C = torch.linalg.pinv(Y) @ mlp_outputs
        except RuntimeError as e:
            # SVD failure: algorithm couldn't converge because input matrix ill-conditioned or has too many repeat singular values
            self.logger.warning(f"Primitive {primitive.name} failed to compute the inverse")
            return None
            
        
        # Check that there are no nans in the output
        if C.isnan().any():
            self.logger.warning(f"Primitive {primitive.name} failed to compute the inverse")
            return None
        
        return C
    
    def search(
        self,
        path: str,
        mlp_inputs: torch.Tensor,
        mlp_outputs: torch.Tensor
    ) -> PrimitiveSearchOutput:
        """This function should be the one to return the optimal primitive if it exists"""
        
        with torch.no_grad():
            best_acc = 0
            best_primitive = None
            best_C = None            
            
            # Filter the primitive set
            filtered_primitives = []
            for primitive in self.all_primitives:
                
                match(primitive.type):
                    
                    case PrimitiveType.EXISTS:
                        # Add exists primitive to each position
                        for i in range(mlp_inputs.size(-1)):
                            filtered_primitives.append(build_primitive(PrimitiveType.EXISTS, idx=i))
                    
                    case PrimitiveType.EQUALS:
                        # for 0, 1 enough, but should add much more possibilities for a,b,c,d...
                        if mlp_inputs.size(-1) >= 6:
                            filtered_primitives.append(build_primitive(PrimitiveType.EQUALS, indices=list(range(mlp_inputs.size(-1) - 4))))
                    
                    case PrimitiveType.ZEROONE:
                        #TODO: explain this
                        if mlp_inputs.size(-1) == 6 and "attn_output" in path:
                            filtered_primitives.append(build_primitive(PrimitiveType.ZEROONE, pow=primitive.pow, center=0))

                    case _:
                        filtered_primitives.append(primitive)
            
            # Test each of the primitives
            
        
        return PrimitiveSearchOutput(
        best_primitive=best_primitive,
            best_C=best_C,
            best_accuracy=best_acc,
        )