"""Render LogitLens heatmaps from cached mlp_input_output.pt entries."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

VAL_TO_BE_NON_ZERO = 0.2
TOP_SAMPLES_TO_KEEP = 40
MAX_LOGIT_OPTIONS = 60
MAX_TICK_LABELS = 24


@st.cache_resource(ttl=2)
def load_mlp_heatmap_cache(mlp_file: str):
    file_path = Path(mlp_file)
    if not file_path.exists():
        return None

    try:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _select_display_columns(sample_rows: np.ndarray) -> tuple[list[int | None], list[str]]:
    num_cols = sample_rows.shape[1]
    non_zero_cols: list[int] = []

    if num_cols > 15:
        for col_idx in range(num_cols):
            if np.max(np.abs(sample_rows[:, col_idx])) > VAL_TO_BE_NON_ZERO:
                non_zero_cols.append(col_idx)
    else:
        for col_idx in range(num_cols):
            if np.any(np.abs(sample_rows[:, col_idx]) > 1e-6):
                non_zero_cols.append(col_idx)

    if not non_zero_cols:
        return [], []

    blocks: list[tuple[int, int]] = []
    start = non_zero_cols[0]
    end = non_zero_cols[0]
    for col in non_zero_cols[1:]:
        if col == end + 1:
            end = col
        else:
            blocks.append((start, end))
            start = col
            end = col
    blocks.append((start, end))

    display_cols: list[int | None] = []
    display_labels: list[str] = []
    for block_idx, (b_start, b_end) in enumerate(blocks):
        if block_idx > 0:
            display_cols.append(None)
            display_labels.append("...")
        for col_idx in range(b_start, b_end + 1):
            display_cols.append(col_idx)
            display_labels.append(str(col_idx))

    return display_cols, display_labels


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _compute_tick_positions_and_labels(display_labels: list[str], max_labels: int = MAX_TICK_LABELS):
    total = len(display_labels)
    if total <= max_labels:
        return list(range(total)), display_labels

    step = max(1, int(np.ceil(total / max_labels)))
    positions = list(range(0, total, step))
    if positions[-1] != total - 1:
        positions.append(total - 1)
    labels = [display_labels[idx] for idx in positions]
    return positions, labels


def _compute_image_width(total_cols: int, base: int = 560, per_col: int = 12, max_width: int = 1200) -> int:
    return min(max_width, base + max(0, total_cols - 16) * per_col)


def build_unembed_heatmap_image(inp_cache, out_cache, token_id: int):
    all_samples = []
    num_bins = len(inp_cache)
    for bin_idx in range(num_bins):
        bin_inp = torch.as_tensor(inp_cache[bin_idx]).cpu()
        bin_out = torch.as_tensor(out_cache[bin_idx]).cpu()
        if bin_inp.numel() == 0 or bin_out.numel() == 0:
            continue

        sample_count = bin_inp.shape[0]
        for sample_idx in range(sample_count):
            logit_val = float(bin_out[sample_idx, token_id, token_id].item())
            inp_row = bin_inp[sample_idx, token_id, :].numpy()
            all_samples.append((logit_val, inp_row))

    if not all_samples:
        return None

    all_samples.sort(key=lambda x: x[0], reverse=True)
    selected = all_samples[:TOP_SAMPLES_TO_KEEP]
    sample_rows = np.stack([row for _, row in selected], axis=0)
    output_data = np.array([val for val, _ in selected], dtype=float)

    display_cols, display_labels = _select_display_columns(sample_rows)
    if not display_cols:
        return None

    heatmap_data = np.full((sample_rows.shape[0], len(display_cols)), np.nan, dtype=float)
    for sample_idx in range(sample_rows.shape[0]):
        for disp_idx, col_idx in enumerate(display_cols):
            if col_idx is not None:
                heatmap_data[sample_idx, disp_idx] = sample_rows[sample_idx, col_idx]

    fig, (input_ax, output_ax) = plt.subplots(
        1,
        2,
        figsize=(max(6.2, len(display_cols) * 0.30 + 1.4), sample_rows.shape[0] * 0.16 + 0.9),
        gridspec_kw={"width_ratios": [max(len(display_cols), 1), 1], "wspace": 0.03},
    )

    im1 = input_ax.imshow(heatmap_data, cmap="Blues", aspect="auto", vmin=0)
    tick_pos, tick_labels = _compute_tick_positions_and_labels(display_labels)
    input_ax.set_xticks(tick_pos)
    input_ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    input_ax.set_yticks([])
    input_ax.set_title("Input", fontsize=9)
    fig.colorbar(im1, ax=input_ax, location="left", fraction=0.04, pad=0.02)

    out_2d = output_data.reshape(-1, 1)
    im2 = output_ax.imshow(out_2d, cmap="RdGy_r", aspect="auto")
    output_ax.set_xticks([0])
    output_ax.set_xticklabels([f"logit[{token_id}]"], fontsize=7)
    output_ax.set_yticks([])
    output_ax.set_title("Output", fontsize=9)
    fig.colorbar(im2, ax=output_ax, fraction=0.25, pad=0.02)
    fig.tight_layout(pad=0.25)

    image = _fig_to_png_bytes(fig)
    plt.close(fig)
    return image, _compute_image_width(len(display_cols), base=560, per_col=14)


def get_ranked_logit_options(out_cache, max_options: int = MAX_LOGIT_OPTIONS):
    score_accumulator = None
    for bin_item in out_cache:
        bin_out = torch.as_tensor(bin_item).cpu()
        if bin_out.numel() == 0 or bin_out.ndim != 3:
            continue

        diag_vals = bin_out.diagonal(dim1=1, dim2=2).abs()
        token_scores = diag_vals.max(dim=0)[0]

        if score_accumulator is None:
            score_accumulator = token_scores
        else:
            score_accumulator = torch.maximum(score_accumulator, token_scores)

    if score_accumulator is None or score_accumulator.numel() == 0:
        return [], {}

    ranked = torch.argsort(score_accumulator, descending=True).tolist()
    ranked = ranked[: min(max_options, len(ranked))]
    score_map = {token_id: float(score_accumulator[token_id].item()) for token_id in ranked}
    return ranked, score_map


def build_select_heatmap_image(q_inp_cache, k_inp_cache, out_cache, cluster_idx: int = 0):
    all_samples = []
    num_bins = len(q_inp_cache)
    for bin_idx in range(num_bins):
        bin_q = torch.as_tensor(q_inp_cache[bin_idx]).cpu()
        bin_k = torch.as_tensor(k_inp_cache[bin_idx]).cpu()
        bin_out = torch.as_tensor(out_cache[bin_idx]).cpu()
        if bin_q.numel() == 0 or bin_k.numel() == 0 or bin_out.numel() == 0:
            continue
        if bin_q.ndim != 3 or bin_k.ndim != 3 or bin_out.ndim != 2:
            continue
        if cluster_idx >= bin_q.shape[1] or cluster_idx >= bin_k.shape[1] or cluster_idx >= bin_out.shape[1]:
            continue

        sample_count = bin_q.shape[0]
        for sample_idx in range(sample_count):
            logit_val = float(bin_out[sample_idx, cluster_idx].item())
            q_row = bin_q[sample_idx, cluster_idx, :].numpy()
            k_row = bin_k[sample_idx, cluster_idx, :].numpy()
            all_samples.append((logit_val, q_row, k_row))

    if not all_samples:
        return None

    all_samples.sort(key=lambda x: x[0], reverse=True)
    selected = all_samples[:TOP_SAMPLES_TO_KEEP]
    q_rows = np.stack([row for _, row, _ in selected], axis=0)
    k_rows = np.stack([row for _, _, row in selected], axis=0)
    output_data = np.array([val for val, _, _ in selected], dtype=float)

    q_cols, q_labels = _select_display_columns(q_rows)
    k_cols, k_labels = _select_display_columns(k_rows)
    if not q_cols or not k_cols:
        return None

    q_heatmap = np.full((q_rows.shape[0], len(q_cols)), np.nan, dtype=float)
    k_heatmap = np.full((k_rows.shape[0], len(k_cols)), np.nan, dtype=float)
    for sample_idx in range(q_rows.shape[0]):
        for disp_idx, col_idx in enumerate(q_cols):
            if col_idx is not None:
                q_heatmap[sample_idx, disp_idx] = q_rows[sample_idx, col_idx]
        for disp_idx, col_idx in enumerate(k_cols):
            if col_idx is not None:
                k_heatmap[sample_idx, disp_idx] = k_rows[sample_idx, col_idx]

    fig, (q_ax, k_ax, out_ax) = plt.subplots(
        1,
        3,
        figsize=(max(8.4, (len(q_cols) + len(k_cols)) * 0.26 + 1.8), q_rows.shape[0] * 0.16 + 1.0),
        gridspec_kw={"width_ratios": [max(len(q_cols), 1), max(len(k_cols), 1), 1], "wspace": 0.03},
    )

    im_q = q_ax.imshow(q_heatmap, cmap="Blues", aspect="auto", vmin=0)
    q_tick_pos, q_tick_labels = _compute_tick_positions_and_labels(q_labels)
    q_ax.set_xticks(q_tick_pos)
    q_ax.set_xticklabels(q_tick_labels, rotation=45, ha="right", fontsize=7)
    q_ax.set_yticks([])
    q_ax.set_title("Query Input", fontsize=9)
    fig.colorbar(im_q, ax=q_ax, location="left", fraction=0.04, pad=0.02)

    im_k = k_ax.imshow(k_heatmap, cmap="Blues", aspect="auto", vmin=0)
    k_tick_pos, k_tick_labels = _compute_tick_positions_and_labels(k_labels)
    k_ax.set_xticks(k_tick_pos)
    k_ax.set_xticklabels(k_tick_labels, rotation=45, ha="right", fontsize=7)
    k_ax.set_yticks([])
    k_ax.set_title("Key Input", fontsize=9)
    fig.colorbar(im_k, ax=k_ax, fraction=0.04, pad=0.02)

    out_2d = output_data.reshape(-1, 1)
    im_out = out_ax.imshow(out_2d, cmap="RdGy_r", aspect="auto")
    out_ax.set_xticks([0])
    out_ax.set_xticklabels(["attn_logit"], fontsize=7)
    out_ax.set_yticks([])
    out_ax.set_title("Output", fontsize=9)
    fig.colorbar(im_out, ax=out_ax, fraction=0.25, pad=0.02)
    fig.tight_layout(pad=0.25)

    image = _fig_to_png_bytes(fig)
    plt.close(fig)
    return image, _compute_image_width(len(q_cols) + len(k_cols), base=760, per_col=10)


def _matching_unembed_paths(path: str, mlp_input_output: dict) -> list[str]:
    return [
        key
        for key in mlp_input_output.keys()
        if isinstance(key, str) and key.endswith(path)
    ]


def _matching_select_paths(path: str, mlp_input_output: dict) -> list[tuple]:
    return [
        key
        for key in mlp_input_output.keys()
        if isinstance(key, tuple)
        and len(key) == 4
        and isinstance(key[0], str)
        and isinstance(key[1], str)
        and (key[0].endswith(path) or key[1].endswith(path))
    ]


def render_path_heatmaps(path: str, mlp_input_output: dict, key_prefix: str = ""):
    matching_unembed_paths = _matching_unembed_paths(path, mlp_input_output)
    matching_select_paths = _matching_select_paths(path, mlp_input_output)
    if not matching_unembed_paths and not matching_select_paths:
        st.caption("No unexplained MLP heatmap found for this path yet.")
        return

    for lens_path in sorted(matching_unembed_paths):
        payload = mlp_input_output.get(lens_path)
        if payload is None or len(payload) != 2:
            continue
        inp_cache, out_cache = payload
        if not inp_cache or not out_cache:
            continue

        ranked_token_ids, token_score_map = get_ranked_logit_options(out_cache)
        if not ranked_token_ids:
            continue

        st.markdown(f"**Unexplained MLP path:** `{lens_path}`")
        selected_token_id = st.selectbox(
            "Output logit to focus on",
            options=ranked_token_ids,
            index=0,
            format_func=lambda tok: f"logit[{tok}] (importance {token_score_map.get(tok, 0.0):.4f})",
            key=f"{key_prefix}logit_selector_{path}_{lens_path}",
        )

        heatmap_result = build_unembed_heatmap_image(inp_cache, out_cache, int(selected_token_id))
        if heatmap_result is None:
            continue
        image, image_width = heatmap_result
        st.image(
            image,
            caption=f"Input (left) and output (right) heatmap for logit[{selected_token_id}]",
            width=image_width,
        )

    for select_path in sorted(matching_select_paths):
        payload = mlp_input_output.get(select_path)
        if payload is None or len(payload) != 3:
            continue
        q_inp_cache, k_inp_cache, out_cache = payload
        if not q_inp_cache or not k_inp_cache or not out_cache:
            continue

        heatmap_result = build_select_heatmap_image(q_inp_cache, k_inp_cache, out_cache, cluster_idx=0)
        if heatmap_result is None:
            continue
        image, image_width = heatmap_result

        q_path, k_path, attn_layer, attn_head = select_path
        st.markdown(
            f"**Select operator path:** `L{attn_layer}H{attn_head}`  "
            f"(query: `{q_path}`, key: `{k_path}`)"
        )
        st.image(
            image,
            caption="Query input (left), key input (middle), output (right)",
            width=image_width,
        )
