import einops
from pathlib import Path
from torch import Tensor
from jaxtyping import Float
from transformer_lens.model_bridge import TransformerBridge

from mechanistic.utilities.misc import get_output_path
from mechanistic.utilities.mechinterp_utils import residual_stack_to_logit, topk_of_Nd_tensor
from mechanistic.utilities.mechinterp_viz import line, imshow

SUBFOLDER = 'attribution'

def residual_stream_attribution(
    model: TransformerBridge,
    tokens: Float[Tensor, "..."],
    position_ids: Float[Tensor, "..."],
    answer_tokens: Float[Tensor, "..."],
    exp_name: str,
    device: str='mps',
    save_html: bool = True,
    artifact_subdir: str | None = None,
):
    """Simple visualization of accumulated residual stream logit attribution"""
    
    # Compute logits and cache
    _, cache = model.run_with_cache(
        tokens.to(device),
        position_ids=position_ids.to(device)
    )
    
    # Determine logit directions for target and accumulate residual contributions
    #TODO: we might want to make the parameters of accumulated_resid editable
    target_logit_directions = model.W_U[:, answer_tokens].T
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1,
        incl_mid=True,
        pos_slice=-1,
        return_labels=True,
    )

    logit_lens_logits: Float[Tensor, "component"] = residual_stack_to_logit(
        accumulated_residual,
        cache,
        target_logit_directions,
    )

    fig = line(
        logit_lens_logits,
        hovermode="x unified",
        title="Logit Contribution From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Contribution"},
        xaxis_tickvals=labels,
        height=260,
        return_fig=True,
    )
    fig.update_layout(
        autosize=True,
        margin=dict(l=48, r=28, t=52, b=44),
    )
    if save_html:
        output_path = get_output_path(exp_name, SUBFOLDER, artifact_subdir)
        fig.write_html(
            output_path / "residual_stream_attribution.html",
            config={"responsive": True},
        )
    return fig

def layerwise_attribution(
    model: TransformerBridge,
    tokens: Float[Tensor, "..."],
    position_ids: Float[Tensor, "..."],
    answer_tokens: Float[Tensor, "..."],
    exp_name: str,
    device: str='mps',
    save_html: bool = True,
    artifact_subdir: str | None = None,
):
    """More fine-grained residual logit attribution by component"""
    
    # Compute logits and cache
    _, cache = model.run_with_cache(
        tokens.to(device),
        position_ids=position_ids.to(device)
    )
    
    target_logit_directions = model.W_U[:, answer_tokens].T
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logits = residual_stack_to_logit(per_layer_residual, cache, target_logit_directions,)

    fig = line(
        per_layer_logits,
        hovermode="x unified",
        title="Logit Contribution From Each Layer",
        labels={"x": "Layer", "y": "Logit Contribution"},
        xaxis_tickvals=labels,
        height=260,
        return_fig=True,
    )
    fig.update_layout(
        autosize=True,
        margin=dict(l=48, r=28, t=52, b=44),
    )
    if save_html:
        output_path = get_output_path(exp_name, SUBFOLDER, artifact_subdir)
        fig.write_html(
            output_path / "residual_stream_component_attribution.html",
            config={"responsive": True},
        )
    return fig

def attention_head_attribution(
    model: TransformerBridge,
    tokens: Float[Tensor, "..."],
    position_ids: Float[Tensor, "..."],
    answer_tokens: Float[Tensor, "..."],
    exp_name: str,
    device: str='mps',
    save_html: bool = True,
    artifact_subdir: str | None = None,
):
    
    # Run model with cache
    _, cache = model.run_with_cache(
        tokens.to(device),
        position_ids=position_ids.to(device)
    )
    
    # Determine logit directions for target
    target_logit_directions = model.W_U[:, answer_tokens].T
    
    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(
        per_head_residual, "(layer head) ... -> layer head ...", layer=model.cfg.n_layers
    )
    per_head_logit_diffs = residual_stack_to_logit(per_head_residual, cache, target_logit_directions)

    fig = imshow(
        per_head_logit_diffs,
        labels={"x": "Head", "y": "Layer"},
        title="Logit Contribution From Each Head",
        height=260,
        return_fig=True,
    )
    fig.update_layout(
        autosize=True,
        margin=dict(l=48, r=56, t=52, b=44),
    )
    if save_html:
        output_path = get_output_path(exp_name, SUBFOLDER, artifact_subdir)
        fig.write_html(
            output_path / "attention_head_attribution.html",
            config={"responsive": True},
        )
    return fig