from torch import Tensor
from functools import partial
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
from typing import Callable, Dict
from functools import partial

from mechanistic.utilities.mechinterp_dataclasses import CircuitNode
from mechanistic.utilities.metrics import get_logit_diff
from data.utils import CleanCorruptData

def patch_from_cache(
    original_act: Float[Tensor, "..."],
    hook: HookPoint,
    patch_cache: ActivationCache,
    node_map: Dict[str, CircuitNode]
):
    """
    IOI-style causal path patching:
    corrupted baseline + selective clean interventions
    """
    
    hook_name = hook.name
    patched_activation = patch_cache[hook_name]
    
    # Patching
    for node in node_map.values():
        
        if hook.layer() != node.layer_idx:
            continue
        
        if node.head_idx is not None:
            original_act[:, :, node.head_idx] = patched_activation[:, :, node.head_idx]
        elif node.neuron_idx is not None:
            original_act[:, node.neuron_idx] = patched_activation[:, node.neuron_idx]
        else:
            original_act[...] = patched_activation[...]

    return original_act

def patch_or_freeze(
    original_act: Float[Tensor, "..."],
    hook: HookPoint,
    patch_cache: ActivationCache,
    freeze_cache: ActivationCache,
    sender_node: CircuitNode
):
    """
    Patch or freeze the original activation based on the node type
    """
    
    original_act[...] = freeze_cache[hook.name][...]
    
    # Only patch the specific sender node to corrupted
    if hook.name == sender_node.name and sender_node.layer_idx == hook.layer():
        
        if sender_node.head_idx is not None:
            original_act[:, :, sender_node.head_idx] = patch_cache[hook.name][:, :, sender_node.head_idx]
        elif sender_node.neuron_idx is not None:
            original_act[:, sender_node.neuron_idx] = patch_cache[hook.name][:, sender_node.neuron_idx]
        else:
            original_act[...] = patch_cache[hook.name][...]
    
    return original_act


def path_patch(
    model: HookedTransformer,
    sender_nodes: Dict[str, CircuitNode],
    receiver_nodes: Dict[str, CircuitNode],
    clean_corrupt_data: CleanCorruptData,
    metric: Callable,
):
    """
    This function performs path patching on a model, using the sender and receiver nodes to patch the model.

    Args:
        model (HookedTransformer): the TransformerLens hooked transformer
        sender_nodes (Dict[str, CircuitNode]): the set of sender nodes to isolate
        receiver_nodes (Dict[str, CircuitNode]): the set of receiver nodes to patch
        clean_corrupt_data (CleanCorruptData): the clean and corrupted tokens and positions
        metric (Callable): the patching metric to use
    """
    
    sender_hook_names = {sn.name for sn in sender_nodes.values()}
    sender_node_filter = lambda name: name in sender_hook_names
    receiver_hook_names = {rn.name for rn in receiver_nodes.values()}
    receiver_node_filter = lambda name: name in receiver_hook_names
    
    # Unpack the clean and corrupted tokens and positions
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    corrupted_tokens = clean_corrupt_data.corrupted_tokens
    corrupted_pos = clean_corrupt_data.corrupted_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    
    # 0. Run models on clean and corrupted tokens to get logits
    clean_logits = model(
        clean_tokens.to(model.cfg.device),
        position_ids=clean_pos.to(model.cfg.device)
    )
    corrupted_logits = model(
        corrupted_tokens.to(model.cfg.device),
        position_ids=corrupted_pos.to(model.cfg.device)
    )
    
    # 1. Get model caches
    model.reset_hooks()
    
    _, clean_cache = model.run_with_cache(
        clean_tokens.to(model.cfg.device),
        position_ids=clean_pos.to(model.cfg.device),
        names_filter=sender_node_filter,
        return_type=None
    )
    
    _, corrupted_cache = model.run_with_cache(
        corrupted_tokens.to(model.cfg.device),
        position_ids=corrupted_pos.to(model.cfg.device),
        names_filter=sender_node_filter,
        return_type=None
    )
    
    # Iterate over sender interventions
    results = {}
    for sender_name, sender_node in sender_nodes.items():
        model.reset_hooks()
                
        # 2. Run with clean tokens, patching ONLY the sender node to its corrupted value
        # This isolates the sender's contribution: everything else stays clean, so the
        # cached activations downstream of the sender reflect only the sender's changed
        # signal propagating forward
        model.add_hook(sender_node_filter, partial(
            patch_or_freeze,
            patch_cache=corrupted_cache,
            freeze_cache=clean_cache,
            sender_node=sender_node
        ))
        _, patched_cache = model.run_with_cache(
            clean_tokens.to(model.cfg.device),
            position_ids=clean_pos.to(model.cfg.device),
            names_filter=receiver_node_filter,
            return_type=None
        )
        
        # Run step 3 for each receiver node
        results[sender_name] = {}
        corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)
        clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
        for receiver_name, receiver_node in receiver_nodes.items():
            model.reset_hooks()
            
            # 3. Run clean tokens, patching each receiver with the activation from
            # patched_cache. This injects only the sender's altered signal into the receiver,
            # while the rest of the corrupted run is unaffected. Thus, we isolate the 
            # sender->receiver edge
            patched_logits = model.run_with_hooks(
                clean_tokens.to(model.cfg.device),
                position_ids=clean_pos.to(model.cfg.device),
                fwd_hooks=[(lambda name: name == receiver_node.name, partial(
                    patch_from_cache,
                    patch_cache=patched_cache,
                    node_map={receiver_name: receiver_node}
                ))],
                return_type="logits"
            )
        
            results[sender_name][receiver_name] = metric(
                logits=patched_logits,
                answer_tokens=answer_tokens,
                corrupted_logit_diff=corrupted_logit_diff,
                clean_logit_diff=clean_logit_diff
            )
        
    return results