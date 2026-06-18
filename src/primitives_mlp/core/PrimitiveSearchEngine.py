import logging
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple

from primitives_mlp.primitives.base import Primitive
from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPPrimitivesConfig, PrimitiveSearchOutput
from primitives_mlp.utilities.mlp_primitive_utils import PrimitiveType, build_primitive
from primitives_mlp.utilities.activation_tracing import trace_mlp, trace_mlp_multi
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from utilities.core import LossModule
from utilities.metrics_logger import MetricsLogger

class PrimitiveSearchEngine:
    
    def __init__(
        self,
        config: MLPPrimitivesConfig,
        hooked_model: GPT2LMHeadModel,
        original_model: GPT2LMHeadModel,
        converted_mlp: Dict[str, PrimitiveSearchOutput],
        oa_vecs: OptimalQueryBiasVectors,
        dataloader: DataLoader,
        loss_module: LossModule,
        all_primitives: List[Primitive],
        metrics_logger: MetricsLogger,
        single_input_mlps: bool,
        logger: logging.Logger
    ):
        self.config = config
        self.metrics_logger = metrics_logger
        self.logger = logger
        
        self.hooked_model = hooked_model
        self.orig_model = original_model
        self.converted_mlp = converted_mlp
        self.oa_vecs = oa_vecs
        self.single_input_mlps = single_input_mlps
        
        self.dataloader = dataloader
        self.loss_module = loss_module
        self.all_primitives = all_primitives
    
    def _find_optimalC(
        self,
        primitive: Primitive,
        mlp_inputs: torch.Tensor,
        mlp_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        # Compute the primitive outputs
        Y = primitive.apply(mlp_inputs)
        
        if Y.size(1) > 10_000:
            self.logger.info(f"Y dim ({Y.size(1)}) is too large, computing C would take a long time. Skipping...")
            return None
        
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
        
        return Y, C
    
    def _evaluate_primitive(
        self,
        path: str,
        layer: int,
        primitive: Primitive,
        C: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute the task and match accuracy for the selected primitive"""
        total_num = 0
        match_num = 0
        correct_num = 0
        
        for i, batch in enumerate(self.dataloader):
            batch = {k: v.to(self.config.torch_device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            # Get target and predicted logits
            target_logits = self.orig_model(**batch).logits
            self.hooked_model(
                masks=torch.ones((1, 1), device=self.config.torch_device),
                oa_vecs=self.oa_vecs,
                **batch
            )
            
            # Trace the activations to find the dependent inputs
            # Then transform the input-dependent inputs by the primitive instead of the MLP
            if self.single_input_mlps:
                input_dependent = trace_mlp(self.hooked_model, self.converted_mlp, path, batch['input_ids'], batch['position_ids'])
            else:
                input_dependent = trace_mlp_multi(self.hooked_model, self.converted_mlp, path, batch['input_ids'], batch['position_ids'])
            Y = primitive.apply(input_dependent)
            
            recon_mlp_out = Y @ C.unsqueeze(0)
            
            def temp_hook(module, input, output):
                self.hooked_model.activations[path] = recon_mlp_out
            
            handle = self.hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
            logits = self.hooked_model(masks=torch.ones((1, 1), device=self.config.torch_device), oa_vecs=self.oa_vecs, **batch).logits
            handle.remove()
            
            batch_size = batch["input_ids"].size(0)
            acc_task, acc_match = self.loss_module.batch_accuracy(
                logits, labels, target_logits, batch["input_ids"]
            )
            match_num += acc_match * batch_size
            correct_num += acc_task * batch_size
            total_num += batch_size
            
            if total_num > 2_000:
                break
        
        acc_match = match_num / total_num
        acc_task = correct_num / total_num
        
        return acc_match, acc_task
    
    def search(
        self,
        path: str,
        layer: int,
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
                        # Only test for this primitive if attention head involved in the path
                        # Only test for this primitive if the second dimension of the inputs is 6 (TODO: why 6?)
                        if mlp_inputs.size(-1) == 6 and "attn_output" in path:
                            filtered_primitives.append(build_primitive(PrimitiveType.ZEROONE, pow=primitive.pow, center=0))
                    
                    case PrimitiveType.KEEPONE:
                        # Test keeping any of the inputs
                        for i in range(len(mlp_inputs)):
                            filtered_primitives.append(build_primitive(PrimitiveType.KEEPONE, keep_n=i))
                    
                    case PrimitiveType.COMBINE:
                        # Do not test the combine primitive if the MLP only has a single input
                        if len(mlp_inputs) == 1:
                            pass

                    case _:
                        filtered_primitives.append(primitive)
            
            # Test each of the primitives
            for idx, primitive in enumerate(filtered_primitives):
                if (primitive.type == PrimitiveType.EXISTS or primitive.type == PrimitiveType.EQUALS) and best_acc >= 0.9:
                    break
                
                optimal_results = self._find_optimalC(primitive, mlp_inputs, mlp_outputs)
                if optimal_results:
                    Y, optC = optimal_results
                    reconstruction_error = (Y @ optC - mlp_outputs).pow(2).mean()
                    FVU = (reconstruction_error / mlp_outputs.var(dim=0).mean()).item()
                else:
                    continue
                
                if FVU < 0.6:
                    # Compute the match and task accuracy
                    acc_match, acc_task = self._evaluate_primitive(path, layer, primitive, optC)
                else:
                    acc_match, acc_task = 0, 0
                
                self.logger.info(f"Primitive: {primitive.name} | FVU: {FVU:.4f} | Acc (match): {acc_match:.3f} | Acc (task): {acc_task:.3f}")
                self.metrics_logger.log(
                    task='primitive_search',
                    path=path,
                    current_primitive=idx+1,
                    total_primitives=len(filtered_primitives),
                    primitive=primitive.name,
                    fvu=FVU,
                    acc_match=acc_match,
                    acc_task=acc_task
                )
                
                adjusted_acc = acc_match if primitive.type != PrimitiveType.NOOP else acc_match + 0.01
                
                if adjusted_acc > best_acc:
                    best_acc = adjusted_acc
                    best_primitive = primitive
                    best_C = optC
                
                if primitive.type == PrimitiveType.NOOP and acc_match > self.config.success_threshold:
                    break
        
        return PrimitiveSearchOutput(
            best_primitive=best_primitive,
            best_C=best_C,
            best_accuracy=best_acc,
        )