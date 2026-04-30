'''
This file collects a series of utility functions to extract parameters relevant to the LogitLens procedure
'''

import re
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from typing import Dict, Tuple

from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from pruning.core.hooks import GPT2QKHooks
from pruning.core.OptimalAblationVectors import OptimalQueryBiasVectors


def get_ln_matrix_for_node(
    model: GPT2LMHeadModel,
    oa_vecs: OptimalQueryBiasVectors,
    layer_idx: int | None,
    qkv_type: str,
    head_idx: int | None,
    activation_name: str | None=None
) -> torch.Tensor:
    """
    Returns the (linear) LayerNorm matrix for a specific component type. In particular, we distinguish
    between the LN for Attention Head Values, Attention Head MLPs, Unembedding Layers and Attention
    Head Queries/Keys

    Args:
        model (GPT2LMHeadModel): the original (unpruned) model
        oa_vecs (OptimalQueryBiasVectors): the OptimalAblation vectors learned during pruning
        layer_idx (int | None): the layer we are interested in
        qkv_type (str): either query or key if inspecting attention head Q/K matrix
        head_idx (int | None): the ID of the head we are interested in (if node is attention head)
        activation_name (str | None, optional): the activation name of the component
    """
    
    match qkv_type:
        
        case "v":
            assert activation_name is not None
            ln_var = oa_vecs.ln_var.data[oa_vecs.to_ln_idx[(layer_idx, "v", head_idx, activation_name)]].exp()
            denom = (ln_var + model.transformer.h[layer_idx].ln_1.eps).sqrt()
            gamma = model.transformer.h[layer_idx].ln_1.weight.data
        
        case "mlp":
            assert activation_name is not None
            ln_var = oa_vecs.ln_var.data[oa_vecs.to_ln_idx[(layer_idx, "mlp", activation_name)]].exp()
            denom = (ln_var + model.transformer.h[layer_idx].ln_2.eps).sqrt()
            gamma = model.transformer.h[layer_idx].ln_2.weight.data
        
        case "lm_head":
            ln_var = oa_vecs.ln_var.data[oa_vecs.to_ln_idx[("lm_head",)]].exp()
            denom = (ln_var + model.transformer.ln_f.eps).sqrt()
            gamma = model.transformer.ln_f.weight.data
        
        case _:
            ln_var = oa_vecs.ln_var.data[oa_vecs.to_ln_idx[(layer_idx, qkv_type, head_idx)]].exp()
            denom = (ln_var + model.transformer.h[layer_idx].ln_1.eps).sqrt()
            gamma = model.transformer.h[layer_idx].ln_1.weight.data
    
    d_model = model.config.hidden_size
    mean_op = torch.eye(d_model) - torch.ones(d_model, d_model) / d_model
    
    W_ln = mean_op.to(gamma.device) @ torch.diag(gamma) / denom.to(gamma.device)
    return W_ln  # Already transposed

def get_qk_for_head(
    model: GPT2LMHeadModel,
    layer_idx: int,
    qkv_type: str,
    head_idx: int
) -> torch.Tensor:
    """
    Returns either the Query or Key weight matrix for a given head

    Args:
        model (GPT2LMHeadModel): the original (unpruned) model
        layer_idx (int): the transformer layer of interest
        qkv_type (str): either q or k depending on desired output
        head_idx (int): the attention head of interest
    """
    attn_layer = model.transformer.h[layer_idx].attn
    w_matrix = attn_layer.c_attn.weight.data
    k_offset = attn_layer.embed_dim
    head_dim = attn_layer.head_dim
    
    W_q = w_matrix[:, head_idx * head_dim : (head_idx + 1) * head_dim].clone()
    W_k = w_matrix[:, k_offset + head_idx * head_dim : k_offset + (head_idx + 1) * head_dim].clone()
    
    if qkv_type == "q":
        return W_q
    elif qkv_type == "k":
        return W_k

    raise RuntimeError(f"The QKV Type passed is not valid: {qkv_type}")
    

def get_ov_for_head(
    model: GPT2LMHeadModel,
    layer_idx: int,
    head_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the output and value weight matrices for a given attention head

    Args:
        model (GPT2LMHeadModel): the original (unpruned) model
        layer_idx (int): the transformer layer of interest
        head_idx (int): the attention head of interest
    """
    attn_layer = model.transformer.h[layer_idx].attn
    w_matrix = attn_layer.c_attn.weight.data
    v_offset = attn_layer.embed_dim * 2
    head_dim = attn_layer.head_dim
    
    W_v = w_matrix[:, v_offset + head_idx * head_dim : v_offset + (head_idx + 1) * head_dim].clone()  # d_model, d_head
    W_o = attn_layer.c_proj.weight.data[head_idx * head_dim : (head_idx + 1) * head_dim].clone()  # d_head, d_model
    
    return W_v, W_o  # Already transposed

def get_attn_weights_for_head(
    hooked_model: GPT2QKHooks,
    layer_idx: int,
    head_idx: int
) -> torch.Tensor:
    """
    Returns the attention weights for a given head. Note that by attention weights
    we mean the softmax weights computed by the head

    Args:
        hooked_model (GPT2QKHooks): the model with hooks, as it already has access to this information from pruning
        layer_idx (int): the transformer layer of interest
        head_idx (int): the attention head of interest
    """
    
    return hooked_model.activations[f'attn_weights-{layer_idx}'][:, head_idx]

def get_input_weights_symbolic(
    hooked_model: GPT2QKHooks,
    oa_vecs: OptimalQueryBiasVectors,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns both the interpretable inputs and the absorbed matrix weights for a
    path ending in a fully symbolic MLP (i.e. one that was explainable). This allows us to collect
    the relevant inputs and outputs for this MLP in situations where we are trying to assess the
    interpretable effect of an unexplained MLP on a select operator where one of the paths (query
    or keys) could be the only one to contain the unexplained MLP. 

    Args:
        hooked_model (GPT2QKHooks): the model with pruning hooks
        oa_vecs (OptimalQueryBiasVectors): the optimal ablation vectors learned during pruning
        converted_mlp (Dict[str, PrimitiveSearchOutput]): the dictionary of MLP primitive replacements
        path (str): the path of interest
        input_ids (torch.Tensor): the token embedding input IDs for this batch
        position_ids (torch.Tensor): the position embedding IDs for this batch

    Raises:
        RuntimeError: if a node in the path is not recognized

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the activation inputs to the MLP and the absorbed weights applied to the original
    """
    
    model = hooked_model.model
    d_model = model.config.hidden_size
    
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path)
    split_nodes = split_nodes[::-1].copy()
    dep_prod = None
    indep_prod = None
    
    # Trace the input from the beginning up until the symbolic MLP 
    for idx, node in enumerate(split_nodes):
        
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A = get_attn_weights_for_head(hooked_model, layer, head).squeeze(0)
            dep_prod = A @ dep_prod
            
            past_path = "-".join(split_nodes[idx - 1::-1])
            W_ln = get_ln_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
            W_v, W_o = get_ov_for_head(model, layer, head)
            indep_prod = indep_prod @ W_ln @ W_v @ W_o
        
        elif node == "wte":
            dep_prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            indep_prod = model.transformer.wte.weight.data
        
        elif node == "wpe":
            dep_prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
            indep_prod = model.transformer.wpe.weight.data
        
        elif node.startswith("mlp"):
            layer = int(node.split("-")[1])
            node_path = "-".join(split_nodes[idx::-1])
            
            if node_path in converted_mlp:
                search_results = converted_mlp[node_path]
                dep_prod = search_results.best_primitive.apply(dep_prod)
                indep_prod = search_results.best_C
            else:
                dep_prod = hooked_model.activations[node_path].squeeze(0)
                indep_prod = torch.eye(d_model).to(model.device)
        
        else:
            raise RuntimeError(f"Node not recognized: {node}")
    
    return dep_prod, indep_prod