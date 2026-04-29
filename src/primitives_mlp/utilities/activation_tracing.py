import torch
import re
import torch.nn.functional as F

@torch.no_grad()
def trace_mlp(
    hooked_model,
    converted_mlp,
    path,
    input_ids,
    position_ids
) -> torch.Tensor:
    """
    This function traces the MLP back to the input activations.
    It is used to determine the input activations that are relevant to the MLP.

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
    split_nodes = split_nodes[len(split_nodes) - 1:0:-1]  # Reverse because we trace backwards 
    prod = None
    
    # For each layer from the outermost, to the innermost...
    for i, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A = hooked_model.activations[f"attn_weights-{layer}"][:, head]
            prod = A @ prod
        elif node == "wte":
            prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
        elif node == "wpe":
            prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
        elif node.startswith("mlp"):
            search_results = converted_mlp["-".join(split_nodes[i::-1])]
            prod = search_results.best_primitive.apply(prod)
        else:
            raise RuntimeError(f"Node not recognized: {node}")
    
    assert prod.dim() == 3
    return prod

@torch.no_grad()
def trace_mlp_multi():
    pass