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
def trace_mlp_multi():
    pass