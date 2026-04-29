'''
This code should act as a hook manager, responsible for
instrumenting the pruned transformer with hooks.

- Should identify paths and extract activations specifically for those trajectories
- Handles filtering of zero-gradient samples to ensure collected data is relevant
'''

import re
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from typing import Dict

from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPDataCollectorOutput, PrimitiveSearchOutput
from primitives_mlp.utilities.activation_tracing import trace_mlp
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors
from utilities.metrics_logger import MetricsLogger

class MLPDataCollector:
    
    def __init__(
        self,
        hooked_model: GPT2LMHeadModel,
        converted_mlp: Dict[str, PrimitiveSearchOutput],
        path: str,
        dataloader: DataLoader,
        oa_vecs: OptimalQueryBiasVectors,
        metrics_logger: MetricsLogger,
        logger: logging.Logger
    ):
        self.hooked_model = hooked_model
        self.converted_mlp = converted_mlp
        self.dataloader = dataloader
        self.oa_vecs = oa_vecs
        self.path = path
        self.metrics_logger = metrics_logger
        self.logger = logger
    
    def collect(self, layer, mlp_inp) -> MLPDataCollectorOutput:
        
        # Get relevant information about the model
        d_model = self.hooked_model.model.config.hidden_size
        
        # Detect all input MLPs and make sure input MLPs have been converted
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        split_nodes = re.findall(pattern, mlp_inp)
        if not all(not node.startswith("mlp") or ("-".join(split_nodes[i:]) in self.converted_mlp) for i, node in enumerate(split_nodes)):
            self.logger.warning("Unable to convert MLP: Dependency on unconverted MLP")
            return MLPDataCollectorOutput(
                mlp_inputs=torch.Tensor([]),
                mlp_outputs=torch.Tensor([]),
                skip=True
            )
        
        mlp_inputs = []
        mlp_outputs = []
        for i, batch in enumerate(self.dataloader):
            batch = {k: v.to(self.hooked_model.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            bz, seq_len = batch["input_ids"].size()
            
            test_tensor = torch.randn(bz, seq_len, d_model, device=self.hooked_model.device, requires_grad=True)
            test_tensor_grad = None
            
            # Hook to save gradient on input tensor
            def capture_grad(grad):
                nonlocal test_tensor_grad
                test_tensor_grad = grad
            
            handle_tensor = test_tensor.register_hook(capture_grad)
            
            mlp_out = None
            def temp_hook(module, input, output):
                nonlocal mlp_out
                mlp_out = self.hooked_model.activations[self.path]
                self.hooked_model.activations[self.path] = test_tensor
            
            handle_mlp = self.hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
            
            # Compute predicted logits and loss
            logits = self.hooked_model(
                masks=torch.ones((1, 1), device=self.hooked_model.device),
                oa_vecs=self.oa_vecs,
                **batch
            ).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten())
            loss.backward()
            
            #TODO: what is this for? - where are we training these masks? Are they the ablated ones?
            masks = test_tensor_grad.abs().sum(dim=-1) > 1e-5
            
            # Remove hooks
            handle_tensor.remove()
            handle_mlp.remove()
            
            # Trace the MLP activations back to determine relevant inputs
            input_dependent = trace_mlp(
                self.hooked_model,
                self.converted_mlp,
                self.path,
                batch["input_ids"],
                batch["position_ids"]
            )
            mlp_inputs.append(input_dependent[masks])
            mlp_outputs.append(mlp_out[masks])
            
            # Log data collection progress
            num_collected = sum(item.size(0) for item in mlp_inputs)
            self.metrics_logger.log(
                task='mlp_data_collection',
                path=self.path,
                collected=num_collected,
                total=max(20_000, num_collected)
            )
            
            # Exit once we have collected enough pairs
            if num_collected > 20_000:
                break    
        
        # Convert lists into tensors
        mlp_inputs = torch.cat(mlp_inputs, dim=0)
        mlp_outputs = torch.cat(mlp_outputs, dim=0)
        assert torch.allclose(mlp_inputs.sum(dim=1), torch.ones(mlp_inputs.size(0), device=self.hooked_model.device), atol=1e-3)
        
        return MLPDataCollectorOutput(
            mlp_inputs=mlp_inputs,
            mlp_outputs=mlp_outputs,
            skip=False
        )