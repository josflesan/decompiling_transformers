from __future__ import annotations

import torch
import circuitsvis as cv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from torch import Tensor
from jaxtyping import Float
from functools import partial
from typing import Callable, Dict, Optional, Tuple

from transformer_lens import HookedTransformer, patching
from transformer_lens.hook_points import HookPoint

from data.utils import CleanCorruptData
from data.CustomTokenizer import CustomTokenizer
from mechanistic.core.activation_patching import attention_head_patching
from mechanistic.utilities.metrics import get_logit_diff
from mechanistic.utilities.mechinterp_utils import get_clean_cache, mean_attn_pattern, topk_of_Nd_tensor

# ----------------------------------------------------------------------------
# 1. Visualize top-k heads by activation patching score
# ----------------------------------------------------------------------------

def investigate_topk_attention_heads(
    model: HookedTransformer,
    tokenizer: CustomTokenizer,
    clean_corrupt_data: CleanCorruptData,
    metric: Callable,
    k: int = 4,
    view: int = 3
):
    # Unpack the clean and corrupted tokens and positions
    clean_tokens = clean_corrupt_data.clean_tokens
    clean_pos = clean_corrupt_data.clean_pos
    answer_tokens = clean_corrupt_data.answer_tokens
    
    corrupted_logits, _ = model.run_with_cache(
        clean_corrupt_data.corrupted_tokens.to(model.cfg.device),
        position_ids=clean_corrupt_data.corrupted_pos.to(model.cfg.device)
    )
    clean_logits, cache = model.run_with_cache(
        clean_tokens.to(model.cfg.device),
        position_ids=clean_pos.to(model.cfg.device)
    )
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    
    # Get the heads with largest value patching
    act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
        model, clean_corrupt_data.corrupted_tokens.to(model.cfg.device), cache, partial(
            metric, answer_tokens=answer_tokens, corrupted_logit_diff=corrupted_logit_diff,
            clean_logit_diff=clean_logit_diff
        )
    )
    top_heads = topk_of_Nd_tensor(act_patch_attn_head_all_pos_every[view], k=k)

    # Get all their attention patterns
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack([
        mean_attn_pattern(cache, layer, head)
        for layer, head in top_heads
    ])

    # Display results
    safe_tokens = tokenizer.convert_ids_to_tokens(clean_tokens[0])
    safe_tokens[0] = "BOS"
    safe_tokens[3] = "SEP"
    
    fig = cv.attention.attention_patterns(
        attention = attn_patterns_for_important_heads,
        tokens = safe_tokens,
        attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
    )
    
    return fig

# ----------------------------------------------------------------------------
# 2. Full-model attention pattern grid
# ----------------------------------------------------------------------------

def plot_all_attention_patterns(
    model: HookedTransformer,
    tokenizer: CustomTokenizer,
    clean_corrupt_data: CleanCorruptData,
) -> go.Figure:
    cache = get_clean_cache(model, clean_corrupt_data)
    clean_tokens = clean_corrupt_data.clean_tokens
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = clean_tokens.shape[1]
    
    safe_tokens = tokenizer.convert_ids_to_tokens(clean_tokens[0])
    safe_tokens[0] = "BOS"
    safe_tokens[3] = "SEP"
    positions = list(range(seq_len))

    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[
            f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for layer in range(n_layers):
        for head in range(n_heads):
            pattern = mean_attn_pattern(cache, layer, head).cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=pattern,
                    x=positions,
                    y=positions,
                    colorscale="Blues",
                    showscale=(layer == 0 and head == n_heads - 1),
                    zmin=0,
                    zmax=1,
                ),
                row=layer + 1,
                col=head + 1,
            )
            if layer == n_layers - 1:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=positions,
                    ticktext=safe_tokens,
                    row=layer + 1,
                    col=head + 1,
                )
            if head == 0:
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=positions,
                    ticktext=safe_tokens,
                    row=layer + 1,
                    col=head + 1,
                )

    cell_px = 72
    fig.update_layout(
        title="Mean Attention Patterns - All Heads",
        template="plotly_white",
        height=cell_px * n_layers + 80,
        width=cell_px * n_heads + 100,
        autosize=False,
        margin=dict(l=16, r=16, t=48, b=16),
    )
    fig.for_each_xaxis(lambda ax: ax.update(matches=None))
    fig.for_each_yaxis(lambda ax: ax.update(matches=None, autorange="reversed"))

    return fig


# ----------------------------------------------------------------------------
# 3. OV Circuit Analysis
# ----------------------------------------------------------------------------

def inspect_ov_circuit(
    model: HookedTransformer,
    layer: int,
    head: int
):
    """
    This function inspects the OV circuit for a given layer and head.
    
    Args:
        model (HookedTransformer): the TransformerLens hooked transformer
        layer (int): the layer to inspect
        head (int): the head to inspect
    """
    # Get the W_V and W_O matrices
    W_V = model.W_V[layer, head]
    W_O = model.W_O[layer, head]
    W_OV = W_V @ W_O
    
    # Project onto the token embedding space
    W_E = model.W_E
    OV_circuit = W_E @ W_OV @ W_E.T
    
    # Display the OV circuit
    fig = px.imshow(
        OV_circuit.detach().cpu().numpy(),
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        title=f"OV Circuit - L{layer}H{head}",
        labels=dict(x="Output Token", y="Input Token", color="Logit"),
        aspect="auto",
    )
    
    return fig

def ov_copying_score(
    model: HookedTransformer,
    layer: int,
    head: int,
    top_k: int = 10
) -> float:
    """
    Measures how much a head copies via its OV circuit.
    Returns the fraction of the top-k predicted tokens (per input token)
    that are the same as the input token. Score in [0, 1]
    """
    
    W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
    OV = (model.W_E @ W_OV @ model.W_E.T).detach()
    
    # For each input token, get topk output tokens
    topk_outputs = OV.topk(top_k, dim=-1).indices  # [vocab, top_k]
    vocab_size = OV.shape[0]
    input_tokens = torch.arange(vocab_size, device=OV.device).unsqueeze(1)
    matches = (topk_outputs == input_tokens).any(dim=-1).float()
    return matches.mean().item()

def inspect_ov_eigenspectrum(
    model: HookedTransformer,
    layer: int,
    head: int,
    top_k: int = 20
) -> go.Figure:
    """
    Plots the top-k singular values of W_OV. Large dominant singular values
    indicate the head is acting as a low-rank copying/routing mechanism.
    """
    
    W_OV = (model.W_V[layer, head] @ model.W_O[layer, head]).detach().cpu()
    _, S, _ = torch.svd(W_OV)
    S = S[:top_k].numpy()
    
    fig = px.bar(
        x=list(range(top_k)),
        y=S,
        labels=dict(x="Singular value rank", y="Magnitude"),
        title=f"OV Eigenspectrum - L{layer}H{head}",
    )
    
    return fig

# ----------------------------------------------------------------------------
# 4. QK Circuit Analysis
# ----------------------------------------------------------------------------

def inspect_qk_circuit(
    model: HookedTransformer,
    layer: int,
    head: int
) -> go.Figure:
    """
    Plots the QK circuit (W_E @ W_Q @ W_K^T @ W_E^T) in token space.
    High values at (i, j) means token i attends to token j.
    """
    
    W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T
    QK_circuit = (model.W_E @ W_QK @ model.W_E.T).detach().cpu().numpy()
    
    fig = px.imshow(
        QK_circuit,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        title=f"QK Circuit - L{layer}H{head}",
        labels=dict(x="Key Token", y="Query Token", color="Attention Logit"),
        aspect="auto",
    )
    
    return fig

# ----------------------------------------------------------------------------
# 5. Positional Attention Heatmap
# ----------------------------------------------------------------------------

def plot_positional_attention(
    model: HookedTransformer,
    clean_corrupt_data: CleanCorruptData,
    layer: int,
    head: int,
) -> go.Figure:
    """
    Shows the mean attention weight from each query position to each key
    position, averaged over the batch. Useful for spotting previous
    token, induction or diagonal heads without looking at token identity
    """
    
    cache = get_clean_cache(model, clean_corrupt_data)
    pattern = mean_attn_pattern(cache, layer, head).cpu().numpy()
    seq_len = pattern.shape[0]
    
    fig = px.imshow(
        pattern,
        x=list(range(seq_len)),
        y=list(range(seq_len)),
        color_continuous_scale="Blues",
        title=f"Positional Attention - L{layer}H{head}",
        labels=dict(x="Key Position", y="Query Position", color="Attention"),
    )
    
    return fig

# ----------------------------------------------------------------------------
# 6. Automatic Head-Type Detection
# ----------------------------------------------------------------------------

@torch.no_grad()
def detect_head_type(
    model: HookedTransformer,
    clean_corrupt_data: CleanCorruptData,
    layer: int,
    head: int,
    induction_offset: int = 1,
    threshold_attn: float = 0.5,
    threshold_copy: float = 0.3,
) -> Dict[str, bool | float]:
    """
    Heuristically classifies a head into common mechanistic archetypes.
    
    Returns a dict with boolean flags and supporting scores:
        - previous_token  : attends primarily to the immediately preceding token
        - bos_attention   : attends primarily to the BOS token (position 0)
        - diagonal        : attends primarily to the current token (identity)
        - induction       : attends to the token after the previous occurrence
                            of the current token (detected via shifted diagonal)
        - copying         : OV circuit has high copying score
        - copying_score   : raw copying score in [0, 1]
        - prev_token_score: mean attention weight on position (q-1)
        - bos_score       : mean attention weight on position 0
        - diagonal_score  : mean attention weight on position q (self)
    """
    
    cache = get_clean_cache(model, clean_corrupt_data)
    pattern = mean_attn_pattern(cache, layer, head).cpu()
    seq_len = pattern.shape[0]
    
    # -- Previous-token Score: Mean of Subdiagonal ---
    prev_attn = torch.diagonal(pattern, offset=-1).mean().item()
    
    # -- BOS Score: Mean Attention to Position 0 --
    bos_attn = pattern[:, 0].mean().item()
    
    # -- Diagonal (self-attention) Score --
    diag_attn = torch.diagonal(pattern, offset=0).mean().item()
    
    # -- Induction Score: Look for attention to (q - seq//2) offset --
    # Classic induction heads attend to the token after the previous
    # occurrence; in a repeated sequence of length L//2 this appears as
    # a diagonal at offset -(L//2 - 1)
    half = seq_len // 2
    if half > 1:
        induction_attn = torch.diagonal(pattern, offset=-(half - 1)).mean().item()
    else:
        induction_attn = 0.0
    
    # -- OV Copying Score --
    copy_score = ov_copying_score(model, layer, head)
    
    return {
        "previous_token":   prev_attn > threshold_attn,
        "bos_attention":    bos_attn > threshold_attn,
        "diagonal":         diag_attn > threshold_attn,
        "induction":        induction_attn > threshold_attn,
        "copying":          copy_score > threshold_copy,
        "prev_token_score": prev_attn,
        "bos_score":        bos_attn,
        "diagonal_score":   diag_attn,
        "induction_score":  induction_attn,
        "copying_score":    copy_score,
    }

def classify_all_heads(
    model: HookedTransformer,
    clean_corrupt_data: CleanCorruptData,
    threshold_attn: float = 0.5,
    threshold_copy: float = 0.3,
) -> Dict[str, Dict[str, bool | float]]:
    """
    Runs detect_head_type for every head in the model and returns a nested
    dict keyed by (layer, head)
    
    Also prints a summary table to stdout
    """
    
    results = {}
    header = f"{'Head':<10} {'Prev':>6} {'BOS':>6} {'Diag':>6} {'Induct':>8} {'Copy':>6}  Types"
    print(header)
    print("-" * len(header))
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            info = detect_head_type(model, clean_corrupt_data, layer, head, threshold_attn, threshold_copy)
            results[layer, head] = info
            
            types = [
                t for t in ("previous_token", "bos_attention", "diagonal", "induction", "copying")
                if info[t]
            ]
            type_str = ", ".join(types) if types else "-"
            print(
                f"L{layer}H{head:<6}"
                f"  {info['prev_token_score']:>5.2f}"
                f"  {info['bos_score']:>5.2f}"
                f"  {info['diagonal_score']:>5.2f}"
                f"  {info['induction_score']:>7.2f}"
                f"  {info['copying_score']:>5.2f}"
                f"  {type_str}"
            )
    
    return results

def plot_head_classification_heatmap(
    model: HookedTransformer,
    classification_results: Dict[Tuple[int, int], Dict],
) -> go.Figure:
    """
    Plots a heatmap grid of (layer x head) with one subplot per head type,
    showing the raw score for each archetype. Useful for spotting patterns
    across the model at a glance.
    """
    
    archetypes = [
        ("prev_token_score", "Previous Token"),
        ("bos_score", "BOS Attention"),
        ("diagonal_score", "Diagonal"),
        ("induction_score", "Induction"),
        ("copying_score", "Copying (OV)"),
    ]
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    fig = make_subplots(
        rows=1,
        cols=len(archetypes),
        subplot_titles=[
            label for _, label in archetypes
        ],
    )
    
    for col_idx, (score_key, label) in enumerate(archetypes):
        mat = torch.zeros(n_layers, n_heads)
        for layer in range(n_layers):
            for head in range(n_heads):
                mat[layer, head] = classification_results[layer, head][score_key]
        
        fig.add_trace(
            go.Heatmap(
                z=mat.numpy(),
                x=[f"H{h}" for h in range(n_heads)],
                y=[f"L{l}" for l in range(n_layers)],
                colorscale="Blues",
                showscale=(col_idx == len(archetypes) - 1),
                zmin=0,
                zmax=1
            ),
            row=1,
            col=col_idx + 1,
        )
    
    fig.update_layout(
        title="Head Type Scores - All Heads",
        height=220,
        width=None,
        autosize=True,
        margin=dict(l=16, r=16, t=48, b=16),
    )
    
    return fig
    