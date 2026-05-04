import torch
import re
import torch.nn.functional as F
from typing import Dict

from primitives_mlp.utilities.mlp_primitive_dataclasses import PrimitiveSearchOutput
from pruning.core.hooks import GPT2ComponentHooks

@torch.no_grad()
def trace_mlp(
    hooked_model: GPT2ComponentHooks,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor
) -> torch.Tensor:
    """
    This function applies all of the transformations from the beginning of
    the computation path up until the last component in the path string. By doing so
    we can collect the representative inputs to the specific component we are analysing.

    Args:
        hooked_model (GPT2ComponentHooks): The hooked model.
        converted_mlp (dict): The converted MLP.
        path (str): The path of the MLP.
        input_ids (torch.Tensor): The input ids.
        position_ids (torch.Tensor): The position ids.
    
    Returns:
        torch.Tensor: The input activations that are relevant to the MLP.
    """
    model = hooked_model.model
    
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path)
    split_nodes = split_nodes[len(split_nodes) - 1:0:-1]  # Reverse because paths are reversed (input dependencies)
    prod = None
    
    # For each layer from the innermost, to the outermost...
    for i, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A = hooked_model.activations[f"attn_weights-{layer}"][:, head]
            prod = A @ prod
        elif node == "wte":
            prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
        elif node == "wpe":
            if not ((position_ids >= 0).all() and (position_ids < model.config.max_position_embeddings).all()):
                raise ValueError(
                    f"Position IDs out of bounds for one-hot encoding: "
                    f"min={position_ids.min().item()}, max={position_ids.max().item()}, "
                    f"allowed=[0, {model.config.max_position_embeddings - 1}]"
                )
            prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
        elif node.startswith("mlp"):
            search_results = converted_mlp["-".join(split_nodes[i::-1])]
            prod = search_results.best_primitive.apply(prod)
        else:
            raise RuntimeError(f"Node not recognized: {node}")
    
    assert prod.dim() == 3
    return prod

@torch.no_grad()
def trace_mlp_multi(
    hooked_model: GPT2ComponentHooks,
    converted_mlp: Dict[str, PrimitiveSearchOutput],
    path: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor
) -> torch.Tensor:
    """
    This function applies the components in a path from its beginning to its end, which can be used to recover
    the inputs to a specific MLP module we are trying to analyze. Unlike the regular trace_mlp function, this function
    contains recursive logic in order to deal with multi-input MLPs that were converted to primitives. 
    
    Tracing through unconverted multi-input MLPs is not supported by this function.

    Args:
        hooked_model (GPT2ComponentHooks): the transformer model with the pruning hooks
        converted_mlp (Dict[str, PrimitiveSearchOutput]): the dictionary containing the primitive search replacements found for different MLPs
        path (str): the string representation of the current path
        input_ids (torch.Tensor): the token embeddings
        position_ids (_type_): the positional embeddings

    Returns:
        torch.Tensor: result from tracing the path from the beginning of the computation graph up until its end
    """
    
    model = hooked_model.model
    config = hooked_model.config
    
    # If the path ends in an MLP, call the function recursively
    # If that MLP was converted to a primitive, apply the primitive instead of the full MLP
    if path.startswith("mlp"):
        prods = []
        for mlp_inp in config[int(path[4:])]["mlp"]:
            prods.append(trace_mlp_multi(hooked_model, converted_mlp, mlp_inp, input_ids, position_ids))
        
        if path in converted_mlp:
            search_results = converted_mlp[path]
            prod = search_results.best_primitive.apply(prod)
            return prod
        else:
            return prods
    
    else:
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        split_nodes = re.findall(pattern, path)
        split_nodes = split_nodes[len(split_nodes) - 1::-1]  # Reverse and exclude the last element in the path
        prod = None
        
        for node in split_nodes:
            if node.startswith("attn_output"):
                _, layer, head = node.split("-")
                layer, head = int(layer), int(head)
                A = hooked_model.activations[f'attn_weights-{layer}'][:, head]
                prod = A @ prod
            elif node == 'wte':
                prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            elif node == 'wpe':
                prod = F.one_hot(position_ids, num_classes=model.config.max_position_embedding).float()
            elif node.startswith("mlp"):
                prod = trace_mlp_multi(hooked_model, converted_mlp, node, input_ids, position_ids)
                if isinstance(prod, list):
                    raise NotImplementedError("Cannot trace through an unconverted MLP")
            else:
                raise RuntimeError(f"Node not recognised: {node}")
        
        assert prod.dim() == 3
        return prod
