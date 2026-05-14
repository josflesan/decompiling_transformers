import torch
import random
import numpy as np
import os
import einops
import yaml

from dacite import from_dict
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, utils

from utilities.core import TaskConfig

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def residual_stack_to_logit(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    target_logit_directions: Float[Tensor, "batch d_model"],
    target_bias: Float[Tensor, "batch"] | None = None,
) -> Float[Tensor, "..."]:
    """
    Projects residual stream components onto target logit directions.
    
    Returns the average contribution to the target logits
    """
    
    bz = residual_stack.size(-2)
    
    # Apply final LayerNorm scaling
    scaled_residual = cache.apply_ln_to_stack(
        residual_stack,
        layer=-1,
        pos_slice=-1
    )
    
    # Project onto target token unembedding directions
    avg_logits = einops.einsum(
        scaled_residual,
        target_logit_directions,
        "... batch d_model, batch d_model -> ..."
    ) / bz
    
    # Optional bias term
    # This is typically omitted in attribution analysis since bias isn't attributable
    if target_bias is not None:
        avg_logits = avg_logits + target_bias.mean()
    
    return avg_logits

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    """
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()