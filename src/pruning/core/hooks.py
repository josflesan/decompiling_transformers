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
        logger=None,
        linearLN=True
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
        self.linearLN = linearLN
        self.hooks =[]
        self.activations = {}
        self.ln_1 = {}
        self.ln_2 = {}
        
        # Set mask sampler
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
            attention_by_head_output = self.model.transformer.h[layer].attn.resid_dropout(attention_by_head_output)
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
                coef = self.masks[:, self.mapping_to_param_idx[(layer, activation, head, activation_name)]].to(self.device)
                
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
                input_activations.append(self._linear_layer_norm(
                    module=self.ln_1[layer],
                    input=summed_activation,
                    scalar=ln_var
                ))
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
            coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]].to(self.device)
            
            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_activation += (first_term + second_term.unsqueeze(1))
        
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
            module (nn.Module): transformer module associated with the hook
            input (torch.Tensor): the original input tensor
            output (torch.Tensor): the original output tensor

        Returns:
            torch.Tensor: the adjusted output tensor
        """
        
        summed_activation = torch.zeros_like(input[0])
        for activation_name in self.full_config["lm_head"]:
            coef = self.masks[:, self.mapping_to_param_idx[("lm_head", activation_name)]].to(self.device)
            
            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            
            summed_activation += (first_term + second_term.unsqueeze(1))
            
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]]
        summed_activation += oa.view(1, 1, -1)
        
        if self.linearLN:
            ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[("lm_head",)]].exp()
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

class GPT2FullPathHooks:
    """
    The goal of these hooks is to return a valid set of activations representing individual input contributions to the output of
    each attention head. We achieve this by masking values and computing repeated forward passes to determine the activation vectors
    of each individual input and attention head combination. The model also linearly decomposes the MLP after self-attention in the
    architecture to keep the influence of each individual input on the overall transformer output clear. In other words, we assume
    
    MLP(x1 + x2) ≈ MLP(x1) + MLP(x2)
    
    This lets us understand the set of contributions by each input to the overall output of the model. In turn, we thus prune these
    paths such as to eliminate irrelevant such circuits/paths.
    """
    
    def __init__(
        self,
        model,
        config,
        mapping_to_param_idx,
        split_mlps=True,
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
        self.split_mlps = split_mlps
        
        self.hooks = []
        self.activations = {}
        self.ln_1 = {}
        self.ln_2 = {}
        self.ln_f = self.model.transformer.ln_f
        
        
        # Set mask sampler
        self.mask_sampler = None
        
        # Define model hooks for token_emb, pos_emb and unembedding layer
        self.hooks.append(self.model.transformer.wte.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wte", layer=None
        )))
        self.hooks.append(self.model.transformer.wpe.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wpe", layer=None
        )))
        self.hooks.append(self.model.lm_head.register_forward_hook(
            self.lm_head_hook
        ))
        
        for layer in range(self.model.transformer.h):
            self.ln_1[layer] = self.model.transformer.h[layer].ln_1
            self.ln_2[layer] = self.model.transformer.h[layer].ln_2
            
            # Pre-attention hook, c_attn hook, c_proj hook and mlp_hook
            self.hooks.append(self.model.transformer.h[layer].attn.register_forward_hook(partial(
                self.attn_pre_hook, layer=layer
            ), with_kwargs=False))
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
        bias=False
    ):
        """
        Linear Layer Norm variant proposed by the authors. Scalar represents the learned gamma constants
        from stage 1 pruning and module represents the original LayerNorm module, which we take the weights from.

        Args:
            module (nn.LayerNorm): the original model's LayerNorm component
            input (torch.Tensor): the input activation to the linear layer norm component
            scalar (float): the scalar learned during KL-div minimization in pruning stage 1
            bias (bool, optional): dictates whether the original module's bias is included in the calculation. Defaults to False.

        Returns:
            torch.Tensor: result of the linearized LayerNorm component
        """
        out = (input - input.mean(dim=-1, keepdim=True)) / (scalar + module.eps).sqrt() * module.weight.view(1, 1, -1)
        if bias:
            out = out + module.bias.view(1, 1, -1)
        
        return out
    
    def save_activation_hook(self, module, input, output, activation_type, layer):
        """
        Passive activation hook to capture original module's activations. In the case of token and position embeddings, this simply captures 
        the output in the activations dictionary.
        
        In the case of attention outputs (after computing per-(head, input) outputs), we save the activations in the path corresponding
        to whichever value input we are currently keeping for this head. Remember that each input is isolated and multiple forward inferences
        are ran in the pre-hook in order to compute several attention outputs for each (head, input) combination.

        Args:
            module (nn.Module): the original transformer component we are capturing activations of
            input (torch.Tensor): the original input to the transformer component
            output (torch.Tensor): the original output of the transformer component
            activation_type (str): the type of activation being saved
            layer (int): the layer in which this component sits
        """
        
        if activation_type in ["wte", "wpe"]:
            self.activations[activation_type] = output.detach()
            
        elif activation_type == "attn_output":
            if not hasattr(self, "activation_name_to_keep") or self.activation_name_to_keep is None:
                return None
            
            input = input[0]
            bz, seq_len = input.shape[:-1]
            d_model = module.weight.size(-1)
            num_heads, head_dim = self.model.transformer.h[layer].attn.num_heads, self.model.transformer.h[layer].attn.head_dim
            attention_by_head = input.view(bz * seq_len, num_heads, head_dim)
            attention_by_head_output = attention_by_head.transpose(0, 1) @ module.weight.view(num_heads, head_dim, d_model)  # (num_head, bz*seq_len, d_model)
            attention_by_head_output = attention_by_head_output.view(num_heads, bz, seq_len, d_model).unbind(dim=0)  # (num_heads, bz, seq_len, d_model)
            
            # Distribute the final attention_by_head_output across the different value input-dependent paths
            for head, attn in enumerate(attention_by_head_output):
                if self.activation_name_to_keep[head] is not None:
                    self.activations[f"{activation_type}-{layer}-{head}-{self.activation_name_to_keep[head]}"] = attn
        
        else:
            raise NotImplementedError(f"Activation Save Hook not defined for {activation_type}")
    
    def attn_pre_hook(self, module, input, layer):
        """
        This hook fires before attention weights and outputs are computed. In other words, we receive the input to the
        whole attention block. This hook is tasked with masking each input value and executing multiple forward passes
        to construct the per-(head, input) paths for the attention layer. In particular, it first computes a list of
        activation inputs to keep for each head and then runs the forward method on these inputs only. Note that we
        use the global attribute activation_name_to_keep as the list of inputs to be kept by each head during a pass.
        

        Args:
            module (nn.Module): global attention module of the transformer receiving input from previous layer/wte+wpe
            input (torch.Tensor): original input of the attention module
            layer (torch.Tensor): layer in which the attention module sits
        """
        self.current_layer = layer
        
        # For each pass, compute which inputs should be kept for each head. Each pass will compute the attention output
        # for multiple single-input heads at the same time
        activation_names_to_keep_per_head = [
            [
                self.config[layer]["v"][head][i]
                if i < len(self.config[layer]["v"][head])
                else None
                for head in range(self.model.transformer.h[layer].attn.num_heads)
            ]
            for i in range(max([len(self.config[layer]["v"][head])
                                for head in range(self.model.transformer.h[layer].attn.num_heads)]))
        ]
        
        # Run forward passes for each input to each head
        for activation_name_to_keep in activation_names_to_keep_per_head:
            self.activation_name_to_keep = activation_name_to_keep
            module.forward(*input)
        
        # Reset attention input masking
        self.activation_name_to_keep = None
    
    def c_attn_hook(self, module, input, output, layer):
        """
        Attention computation hook. This hook is responsible for applying learned masks and optimal ablation vectors
        to each of the paths computed. Importantly, Query and Key computation proceeds, however Value outputs are computed
        according to the active inputs dictated by self.activation_name_to_keep. In other words, we learn optimal ablation
        vectors (and masks) for each of these new input-dependent paths for each head.

        Args:
            module (nn.Module): the transformer attention module
            input (torch.Tensor): the original input tensor to the module
            output (torch.Tensor): the original output tensor of the module
            layer (int): the layer in which the attention module sits

        Returns:
            torch.Tensor: masked and ablated output activation
        """
        
        if not hasattr(self, "activation_name_to_keep") or self.activation_name_to_keep is None:
            return None
        assert self.current_layer == layer
        
        num_heads = self.model.transformer.h[layer].attn.num_heads
        head_dim = self.model.transformer.h[layer].attn.head_dim
        d_model = input[0].size(-1)
        
        # Mask and ablate query and key paths as before
        input_activations = []
        for activation in ["q", "k"]:
            for head in range(num_heads):
                summed_activation = torch.zeros_like(input[0])
                
                # Accumulate masked and ablated contributions from each sending vertex of the head
                for activation_name in self.config[layer][activation][head]:
                    coef = self.masks[:, self.mapping_to_param_idx[(layer, activation, head, activation_name)]]
                    
                    first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                    oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                    second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                        oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                    summed_activation = summed_activation + first_term + second_term.unsqueeze(1)
                
                # Apply learned output optimal ablation to compensate for missing biases
                oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, activation, head)]]
                summed_activation = summed_activation + oa.view(1, 1, -1)
                
                # Apply linear LayerNorm transformation for this head learned in stage 1
                ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, activation, head)]].exp()
                input_activations.append(self._linear_layer_norm(self.ln_1[layer], summed_activation, ln_var))

        # Compute attention outputs by isolating active value inputs only 
        for head in range(num_heads):
            if hasattr(self, "activation_name_to_keep") and self.activation_name_to_keep is not None and self.activation_name_to_keep[head] is not None:
                activation_name = self.activation_name_to_keep[head]
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "v", head, activation_name)]]
                
                # Compute masked and ablated contribution of this value input
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_activation = first_term + second_term.unsqueeze(1)
                
                # Apply learned linear LayerNorm transformation
                ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, "v", head, activation_name)]].exp()
                # Avoid adding bias multiple times, leave this to output OA vec
                input_activation = self._linear_layer_norm(self.ln_1[layer], summed_activation, ln_var, bias=False)
                
            else:
                input_activation = torch.zeros_like(input[0])
            
            input_activations.append(input_activation)
        
        # Collect input activations for each head and compute final attention output
        # Importantly, be careful not to double count the bias contribution to value inputs by zeroing those out
        input_activations = torch.stack(input_activations)
        output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
            module.weight.view(d_model, num_heads * 3, head_dim).transpose(0, 1)
        output_activations = output_activations.transpose(0, 1).contiguous().view(*input[0].size()[:2], num_heads * head_dim * 3)
        bias = module.bias.clone()
        bias[num_heads * head_dim * 2:] = 0  # Avoid adding bias multiple times to the input Value positions
        output_activations += bias.view(1, 1, -1)
        
        return output_activations

    def mlp_hook(self, module, input, output, layer):
        """
        Hook for the MLP component of the transformer layer. If not using split MLPs, we compute the output
        activation of this module as the original MLP block, otherwise we route the accumulated input
        activations to the corresponding single-input MLP.
        
        Input activation accumulation happens as in GPT2ComponentHooks. Namely, we consider each sending
        vertex of the MLP, mask them and add the learned ablation vector if needed (i.e. do not train
        the ablation vector if the edge is to be kept). As before, we also add another global ablation
        vector representing the global bias of previous layers (e.g. linear layers in activation heads).
        We also apply the linearized LayerNorm using the coefficients learned in stage 1.

        Args:
            module (nn.Module): the PyTorch module for the MLP being hooked
            input (torch.Tensor): the original input tensor for the MLP
            output (torch.Tensor): the original output tensor for the MLP
            layer (int): the transformer layer

        Returns:
            Union[torch.Tensor, None]: the updated output of this component after masking + ablations or None if
            we are using the split MLPs (since the original MLP will no longer exist)
        """
        
        if len(self.config[layer]["mlp"]) == 0:
            return None
        
        if not self.split_mlps:
            # For each MLP receiving vertex, total new activations using ablation vectors
            # where needed
            summed_activation = torch.zeros_like(input[0])
            for activation_name in self.config[layer]["mlp"]:
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]
                
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_activation = summed_activation + first_term + second_term.unsqueeze(1)

            # Add output ablation vector to account for bias
            oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, "mlp")]]
            summed_activation = summed_activation + oa.view(1, 1, -1)
            
            # Transform resulting input activation using linear LayerNorm
            ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, "mlp")]].exp()
            input_activation = self._linear_layer_norm(self.ln_2[layer], summed_activation, ln_var)
            
            output = module.forward(input_activation)
            self.activations[f"mlp-{layer}"] = output
            return output
        
        else:
            # For each MLP receiving vertex, total masked activations, using OA vecs where needed
            for activation_name in self.config[layer]["mlp"]:
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]
                
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(1) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_activation = first_term + second_term.unsqueeze(1)
                
                # Compute LN transformation using linear variant
                ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[(layer, "mlp", activation_name)]].exp()
                input_activation = self._linear_layer_norm(self.ln_2[layer], summed_activation, ln_var)
                
                # Output will be produced by the relevant split MLP
                output = self.oa_vecs.mlps[f"{layer} {activation_name}"](input_activation)
                self.activations[f"mlp-{layer}-{activation_name}"] = output
            
            return None
    
    def lm_head_hook(self, module, input, output):
        """
        Hook for the LM Head (unembedding) component of the model. As with other hooks, this computes
        an accumulated input activation from all of the sending vertices of this component (making sure
        to mask and add learned ablation vectors as needed). The resulting input activation is then added
        with a global output ablation vector to account for skipped biases. Finally, we apply the learned 
        linear LayerNorm transform.

        Args:
            module (nn.Module): the unembedding layer module of the GPT2 transformer
            input (torch.Tensor): the original input tensor for this module
            output (torch.Tensor): the original output tensor for this module

        Returns:
            torch.Tensor: the adjusted output activation after applying masking and ablations
        """
        summed_activation = torch.zeros_like(input[0])
        
        # Accumulate input activation from set of all sending vertices
        for activation_name in self.config["lm_head"]:
            coef = self.masks[:, self.mapping_to_param_idx[("lm_head", activation_name)]]
            
            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_activation = summed_activation + first_term + second_term.unsqueeze(1)
        
        # Add global bias for the receiving vertices
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]]
        summed_activation = summed_activation + oa.view(1, 1, -1)
        
        # Apply linear layer norm transformation
        ln_var = self.oa_vecs.ln_var[self.oa_vecs.to_ln_idx[("lm_head",)]].exp()
        input_activation = self._linear_layer_norm(self.ln_f, summed_activation, ln_var)
        
        output = module.forward(input_activation)
        return output
    
    def remove_hooks(self):
        """Simple utility to remove all hooks from the model."""
        for h in self.hooks:
            h.remove()
    
    def __call__(
        self,
        *args,
        masks=None,
        oa_vecs: OptimalAblationVectors=None,
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
                bz = kwargs["input_ids"].size(0)
            
            self.masks = self.mask_sampler.sample_binary_masks(bz)
        
        return self.model(*args, **kwargs)