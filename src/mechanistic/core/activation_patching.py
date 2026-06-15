from functools import partial
from typing import Callable
from transformer_lens import HookedTransformer, patching

from mechanistic.utilities.metrics import get_logit_diff
from mechanistic.utilities.mechinterp_viz import imshow
from data.utils import CleanCorruptData

def residual_stream_patching(
    model: HookedTransformer,
    clean_corrupt_data: CleanCorruptData,
    metric: Callable
):
    """
    This function performs activation patching on the residual stream of a model, using the metric to compute the logit difference.
    
    Args:
        model (HookedTransformer): the TransformerLens hooked transformer
        clean_corrupt_data (CleanCorruptData): the clean and corrupted tokens and positions
        metric (Callable): the patching metric to use
    """
    
    # Unpack the clean and corrupted tokens and positions
    corrupted_tokens = clean_corrupt_data.corrupted_tokens
    corrupted_pos = clean_corrupt_data.corrupted_pos
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    
    
    # Run models on clean and corrupted tokens to get logits
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens.to(model.cfg.device),
        position_ids=clean_pos.to(model.cfg.device)
    )
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens.to(model.cfg.device),
        position_ids=corrupted_pos.to(model.cfg.device)
    )
    
    # Get the logit diffs
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    
    # Compute the activation patching
    act_patch_block_every = patching.get_act_patch_block_every(
        model, corrupted_tokens.to(model.cfg.device), clean_cache, partial(
            metric, answer_tokens=answer_tokens, corrupted_logit_diff=corrupted_logit_diff,
            clean_logit_diff=clean_logit_diff
        )
    )
    
    labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
    fig = imshow(
        act_patch_block_every,
        x=labels,
        labels={"x": "Sequence Position", "y": "Layer"},
        title="Logit Difference From Patched Residual Stream",
        facet_col=0,
        width=1200,
        return_fig=True,
    )
    
    return fig

def attention_head_patching(
    model: HookedTransformer,
    clean_corrupt_data: CleanCorruptData,
    metric: Callable
):
    """
    This function performs activation patching on the attention heads of a model, using the metric to compute the logit difference.
    """
    
    # Unpack the clean and corrupted tokens and positions
    corrupted_tokens = clean_corrupt_data.corrupted_tokens
    corrupted_pos = clean_corrupt_data.corrupted_pos
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    
    # Run models on clean and corrupted tokens to get logits
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens.to(model.cfg.device),
        position_ids=clean_pos.to(model.cfg.device)
    )
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens.to(model.cfg.device),
        position_ids=corrupted_pos.to(model.cfg.device)
    )
    
    # Get the logit diffs
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    
    act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens.to(model.cfg.device), clean_cache, partial(
            metric, answer_tokens=answer_tokens, corrupted_logit_diff=corrupted_logit_diff,
            clean_logit_diff=clean_logit_diff
        )
    )
    
    fig = imshow(
        act_patch_attn_head_all_pos_every,
        facet_col=0,
        facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
        title="Activation Patching Per Head (All Pos)",
        labels={"x": "Head", "y": "Layer"},
        width=1200,
        return_fig=True,
    )
    
    return fig