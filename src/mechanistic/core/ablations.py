import torch
import numpy as np
from torch import Tensor
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple

from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from mechanistic.utilities.mechinterp_dataclasses import CircuitNode
from mechanistic.utilities.metrics import get_logit_diff
from data.utils import CleanCorruptData

# ----------------------------------------------------------------------------
# Hook Helpers
# ----------------------------------------------------------------------------

AblationMode = Literal["zero", "mean"]

def _ablate_activation(
    activation: Float[Tensor, "..."],
    hook: HookPoint,
    node: CircuitNode,
    mode: AblationMode,
    mean_cache: Optional[ActivationCache] = None
) -> Float[Tensor, "..."]:
    """
    Ablates a single node (head, neuron or full hook) in place.
    """
    
    if mode == "mean":
        assert mean_cache is not None, "Mean cache is required for mean ablation"
        replacement = mean_cache[hook.name]
    else:
        replacement = torch.zeros_like(activation)
    
    if node.head_idx is not None:
        # Activation head: shape [..., n_heads, head_dim]
        activation[:, :, node.head_idx] = (
            replacement[:, :, node.head_idx] if mode == "mean"
            else torch.zeros_like(activation[:, :, node.head_idx])
        )
    
    elif node.neuron_idx is not None:
        activation[:, :, node.neuron_idx] = (
            replacement[:, :, node.neuron_idx] if mode == "mean"
            else torch.zeros_like(activation[:, :, node.neuron_idx])
        )
    
    else:
        # Full hook (e.g. MLP out, full residual stream, etc.)
        activation[...] = replacement[...] if mode == "mean" else torch.zeros_like(activation)
    
    return activation

# ----------------------------------------------------------------------------
# Mean Cache Computation
# ----------------------------------------------------------------------------

def compute_mean_cache(
    model: HookedTransformer,
    tokens: Float[Tensor, "batch seq"],
    position_ids: Float[Tensor, "batch seq"],
    hook_names: List[str],
) -> ActivationCache:
    """
    Runs the model on a corpus and returns a cache of mean activations
    (averaged over the batch dimension) for the requested hook names.
    """
    
    names_filter = lambda name: name in set(hook_names)
    
    kwargs = dict(names_filter=names_filter, return_type=None)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(model.cfg.device)
    
    _, cache = model.run_with_cache(
        tokens.to(model.cfg.device),
        **kwargs
    )
    
    mean_values = {k: v.mean(dim=0, keepdim=True) for k, v in cache.items()}
    return ActivationCache(mean_values, model)

# ----------------------------------------------------------------------------
# Ablation Functions
# ----------------------------------------------------------------------------

def ablate_node(
    model: HookedTransformer,
    node: CircuitNode,
    clean_corrupt_data: CleanCorruptData,
    metric: Callable,
    mode: AblationMode = "mean",
    mean_cache: Optional[ActivationCache] = None,
) -> float:
    """
    Ablates a single CircuitNode and returns the metric value
    """
    
    model.reset_hooks()
    
    # Unpack the clean and corrupted tokens and positions
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    clean_logits = model(clean_tokens.to(model.cfg.device), position_ids=clean_pos.to(model.cfg.device))
    
    hook_fn = partial(
        _ablate_activation,
        node=node,
        mode=mode,
        mean_cache=mean_cache,
    )
    
    run_kwargs = dict(
        fwd_hooks=[(lambda name: name == node.name, hook_fn)],
        position_ids=clean_pos.to(model.cfg.device),
        return_type="logits",
    )
    
    logits = model.run_with_hooks(clean_tokens.to(model.cfg.device), **run_kwargs)
    return metric(logits, clean_logits, answer_tokens)

def ablate_circuit(
    model: HookedTransformer,
    circuit: Dict[str, CircuitNode],
    clean_corrupt_data: CleanCorruptData,
    metric: Callable,
    mode: AblationMode = "mean",
    mean_cache: Optional[ActivationCache] = None,
) -> float:
    """
    Ablates all nodes in a circuit simultaneously and returns the metric value.
    Useful for measuring the combined necessity of a set of components.
    """
    model.reset_hooks()
    
    # Unpack the clean and corrupted tokens and positions
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    clean_logits = model(clean_tokens.to(model.cfg.device), position_ids=clean_pos.to(model.cfg.device))
    
    # Group nodes by hook name so one hook fn handles multiple heads per layer
    hook_name_to_nodes: Dict[str, List[CircuitNode]] = {}
    for node in circuit.values():
        hook_name_to_nodes.setdefault(node.name, []).append(node)
    
    hook_names = set(hook_name_to_nodes.keys())
    
    def _multi_node_ablate(activation, hook, nodes, mode, mean_cache):
        for node in nodes:
            activation = _ablate_activation(activation, hook, node, mode, mean_cache)
        return activation

    fwd_hooks = [
        (
            lambda name, hn=hook_name: name == hn,
            partial(_multi_node_ablate, nodes=nodes, mode=mode, mean_cache=mean_cache)
        )
        for hook_name, nodes in hook_name_to_nodes.items()
    ]
    
    run_kwargs = dict(fwd_hooks=fwd_hooks, position_ids=clean_pos.to(model.cfg.device), return_type="logits")
    logits = model.run_with_hooks(clean_tokens.to(model.cfg.device), **run_kwargs)
    
    return metric(logits, clean_logits, answer_tokens)

# ----------------------------------------------------------------------------
# Ablation Sweep
# ----------------------------------------------------------------------------

def ablation_sweep(
    model: HookedTransformer,
    nodes: Dict[str, CircuitNode],
    clean_corrupt_data: CleanCorruptData,
    metric: Callable,
    mode: AblationMode = "mean",
    mean_tokens: Optional[Float[Tensor, "batch seq"]] = None,
    mean_position_ids: Optional[Float[Tensor, "batch seq"]] = None,
) -> Dict[str, float]:
    """
    Runs a per-node ablation sweep, returning a dictionary of node_key->scores

    Args:
        model (HookedTransformer): the TransformerLens hooked transformer
        nodes (Dict[str, CircuitNode]): a dictionary of node_key->CircuitNode
        tokens (Float[Tensor, &quot;batch seq&quot;]): the input tokens
        position_ids (Optional[Float[Tensor, &quot;batch seq&quot;]]): the input position ids
        answer_tokens (Int[Tensor, &quot;batch 2&quot;]): the answer tokens
        metric (Callable): the metric to use
        mode (AblationMode, optional): the ablation mode. Defaults to "mean".
        mean_tokens (Optional[Float[Tensor, &quot;batch seq&quot;]], optional): the mean tokens to use. Defaults to None.
        mean_position_ids (Optional[Float[Tensor, &quot;batch seq&quot;]], optional): the mean position ids to use. Defaults to None.
    """
    
    # Unpack
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    
    mean_cache = None
    if mode == "mean":
        corpus = mean_tokens if mean_tokens is not None else clean_tokens
        corpus_pos = mean_position_ids if mean_position_ids is not None else clean_pos
        hook_names = list({n.name for n in nodes.values()})
        mean_cache = compute_mean_cache(model, corpus, corpus_pos, hook_names)
    
    results = {}
    for node_key, node in nodes.items():
        score = ablate_node(
            model=model,
            node=node,
            clean_corrupt_data=clean_corrupt_data,
            metric=metric,
            mode=mode,
            mean_cache=mean_cache,
        )
        score_val = score.item() if hasattr(score, "item") else float(score)
        results[node_key] = score_val
        print(f"    {node_key:45s} -> {score_val:.4f}")
    
    return results
    