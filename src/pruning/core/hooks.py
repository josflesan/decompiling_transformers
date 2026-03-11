import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from itertools import product

from pruning.core.OptimalAblationVectors import OptimalAblationVectors

class GPT2ComponentHooks:
    """
    Model that exposes ablation hooks for causal pruning stage 1 (components). Note this
    assumes a GPT2 architecture
    
    Model Hooks:
    
        1. Embedding Layer Hook (nn.Embedding) - hook for both word token embeddings and position embeddings
        2. LM Head Hook - hook on the final projection (unembedding) layer
        3. Final Layer Normalization Hook - hook on the final LayerNorm (after last transformer block)
        4. Transformer Layer Hooks:

            a. LN1 Hook - LayerNorm before self-attention
            b. LN2 Hook - LayerNorm before MLP
            c. Attention Input Projection Hook - linear layers for query, key and value computations
            d. Attention Output Projection Hook - linear projection after attention aggregation
            e. MLP Hook - FFNN in each transformer block
    """
    
    def __init__(
        self,
        model,
        config,
        mapping_to_param_idx,
        logger=None
    ):
        self.model = model
        
        # Disable model training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        self.full_config = deepcopy(config)
        self.mapping_to_param_idx = mapping_to_param_idx
        self.logger = logger
        self.device = self.model.device
        self.linearLN = self.full_config.stage_config.linear_ln
        self.hooks =[]
        self.activations = {}
        self.ln_1 = {}
        self.ln_2 = {}
        
        # Set mask sampler
        #TODO: change this to the appropriate mask sampler
        self.mask_sampler = None
        
        # Define model hooks
        self.hooks.append(self.model.transformer.wte.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wte", layer=None
        )))
        self.hooks.append(self.model.transformer.wpe.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wpe", layer=None
        )))
        self.hooks.append(self.model.lm_head.register_forward_hook(
            self.lm_head_hook
        ))
        self.ln_f = self.model.transformer.ln_f
        for layer in range(len(self.model.transformer.h)):
            self.ln_1[layer] = self.model.transformer.h[layer].ln_1
            self.ln_2[layer] = self.model.transformer.h[layer].ln_2
            self.hooks.append(self.model.transformer.h[layer].attn.c_attn.register_forward_hook(partial(
                self.c_attn_hook, layer=layer
            )))
            self.hooks.append(self.model.transformer.h[layer].attn.c_proj.register_forward_hook(partial(
                self.save_activation_hook, activation_type="attn_output", layer=layer
            )))
            self.hooks.append(self.model.transformer.h[layer].mlp.register_forward_hook(partial(
                self.mlp_hook, layer=layer
            )))
        
        self.masks = None
        self.oa_vecs = None
        
    def _linear_layer_norm(
        self,
        module: nn.LayerNorm,
        input: torch.Tensor,
        scalar: float,
        bias=True
    ):
        out = (input - input.mean(dim=-1, keepdim=True)) / (scalar + module.eps).sqrt() * module.weight.view(1, 1, -1)
        if bias:
            out = out + module.bias.view(1, 1, -1)
        
        return out
        
    def save_activation_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        activation_type: str,
        layer: int
    ):
        """
        Passive hook that captures and saves the activations in the self.activations dictionary. These activations
        are needed by other components in the transformer to compute direct paths.

        Args:
            module (nn.Module): PyTorch module of the component we are hooking
            input (torch.Tensor): input to the component
            output (torch.Tensor): output of the component
            activation_type (str): type of activation (WTE, WPE or Attention Output Projection)
            layer (int): transformer layer

        Raises:
            NotImplementedError: if an unknown activation is passed, we raise this error
        """
        
        if activation_type in ["wte", "wpe"]:
            self.activations[activation_type] = self.model.transformer.drop(output.detach())
        elif activation_type == "attn_output":
            input = input[0]
            bz, seq_len = input.shape[:-1]
            d_model = module.weight.size(-1)
            num_heads, head_dim = self.model.transformer.h[layer].attn.num_heads, self.model.transformer.h[layer].attn.head_dim
            attention_by_head = input.view(bz * seq_len, num_heads, head_dim)
            attention_by_head_output = attention_by_head.transpose(0, 1) @ module.weight.view(num_heads, head_dim, d_model) # num_head, bz*seq_len, d_model
            attention_by_head_output = self.mode.transformer.h[layer].attn.resid_dropout(attention_by_head_output)
            attention_by_head_output = attention_by_head_output.view(num_heads, bz, seq_len, d_model).unbind(dim=0)
            
            for head, attn in enumerate(attention_by_head_output):
                self.activations[f"{activation_type}-{layer}-{head}"] = attn
            
        else:
            raise NotImplementedError()
    
    def c_attn_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        layer: int
    ):
        
        num_heads = self.model.transformer.h[layer].attn.num_heads
        head_dim = self.model.transformer.h[layer].attn.head_dim
        d_model = input[0].size(-1)
        
        input_activations = []
        for i, (activation, head) in enumerate(product(["q", "k", "v"], range(num_heads))):
            summed_activation = torch.zeros_like(input[0])
            for activation_name in self.full_config[layer][activation][head]:
                
                # 1. Select the learned coefficient/mask for this input
                coef = self.masks[:, self.mapping_to_param_idx[(layer, activation, head, activation_name)]]
                
                # 2. Mask the real activation signal using the coefficient
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                
                # 3. Select the relevant optimal ablation vector and compute the second term
                # if coef == 1 -> we keep the edge (first_term)
                # if coef == 0 -> we prune the edge (second_term)
                # if 0 < coef < 1 -> we keep some degree of both
                #
                # Note: we only train the optimal ablation vector if the edge is being pruned (coef ~ 0)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_activation += first_term + second_term.unsqueeze(1)
            
            # 4. Add the optimal ablation vector for the generic bias learned
            oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, activation, head)]]
            summed_activation += oa.view(1, 1, -1)
            
            if self.linearLN:
                ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, activation, head)]].exp()
                input_activations.append(self._linear_layer_norm(self.ln_1[layer], summed_activation, ln_var))
            else:
                input_activations.append(self.ln_1[layer](summed_activation))
            
        input_activations = torch.stack(input_activations)
        
        # Compute the new output activations
        output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
            module.weight.view(d_model, num_heads * 3, head_dim).transpose(0, 1)
        output_activations = output_activations.transpose(0, 1).contiguous().view(*input[0].size()[:2], num_heads * head_dim * 3)
        output_activations += module.bias.view(1, 1, -1)
        
        return output_activations
    
    def mlp_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        layer: int
    ):
        """
        Hook for the MLP component of the transformer layer. We loop over every allowed input path to the MLP
        and fetch previous activations, multiplying these by the mask and adding the optimal ablation
        vector. The new manually computed input_activation becomes the input to this module and the output based
        on this input is saved to the activations dictionary.
        
        The hook also optionally uses the gamma to linearize LayerNorm if needed and applies that linear 
        transformation instead of the original LN.

        Args:
            module (nn.Module): the PyTorch module for the MLP being hooked
            input (torch.Tensor): the original input tensor for the MLP
            output (torch.Tensor): the original output tensor for the MLP
            layer (int): the transformer layer

        Returns:
            torch.Tensor: the updated output of the component after multiplying paths with mask and adding ablation vecs
        """
        
        summed_activation = torch.zeros_like(input[0])
        for activation_name in self.full_config[layer]["mlp"]:
            coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]
            
            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_activation += first_term + second_term.unsqueeze(1)
        
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, "mlp")]]
        summed_activation += oa.view(1, 1, -1)
        
        if self.linearLN:
            ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, "mlp")]].exp()
            input_activation = self._linear_layer_norm(self.ln_2[layer], summed_activation, ln_var)
        else:
            input_activation = self.ln_2[layer](summed_activation)
            
        output = module.forward(input_activation)
        self.activations[f"mlp-{layer}"] = output  # dropout included
        return output
            

    def lm_head_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook for the unembedding layer. We again loop over every allowed input path to the LM head,
        scale the activations by their masks and add the optimal ablation vectors. The manually computed
        input_activation becomes the input to this module and the output based on this input is returned.
        
        The hook also optionally uses the gamma to linearize the final LayerNorm if needed and applies the
        linear transformation instead of the original LN.

        Args:
            module (nn.Module): _description_
            input (torch.Tensor): _description_
            output (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        
        summed_activation = torch.zeros_like(input[0])
        for activation_name in self.full_config["lm_head"]:
            coef = self.masks[:, self.mapping_to_param_idx[("lm_head", activation_name)]]
            
            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            
            summed_activation += first_term + second_term.unsqueeze(1)
            
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]].exp()
        summed_activation += oa.view(1, 1, -1)
        
        if self.linearLN:
            ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_LN_idx[("lm_head",)]].exp()
            input_activation = self._linear_layer_norm(self.ln_f, summed_activation, ln_var)
        else:
            input_activation = self.ln_f(summed_activation)
            
        output = module.forward(input_activation)
        return output
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
    
    def __call__(
        self,
        *args,
        masks=None,
        oa_vecs: OptimalAblationVectors = None,
        **kwargs
    ):
        if masks is not None:
            self.masks = masks
            self.oa_vecs = oa_vecs
            
        else:
            assert hasattr(self, "mask_sampler") and self.mask_sampler is not None
            assert self.oa_vecs is not None
            if args:
                bz = args[0].size(0)
            else:
                bz = kwargs['input_ids'].size(0)
            
            self.masks = self.mask_sampler.sample_binary_masks(bz)
            
        return self.model(*args, **kwargs)