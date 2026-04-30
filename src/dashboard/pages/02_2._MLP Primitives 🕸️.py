import psutil
import math
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics

st.set_page_config(layout="wide")
st_autorefresh(interval=2000)

st.title("MLP Primitive Replacement Dashboard")

VAL_TO_BE_NON_ZERO = 0.2
TOP_TOKENS_TO_PLOT = 1
TOP_SAMPLES_TO_KEEP = 40

#TODO: add column labels so the plots are more useful
#TODO: make the heatmaps interactive so users can select the kth most important vector if needed

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

    # Input heatmap (left)
    im1 = input_ax.imshow(heatmap_data, cmap="Blues", aspect="auto", vmin=0)
    input_ax.set_xticks(range(len(display_labels)))
    input_ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)
    input_ax.set_yticks([])
    input_ax.set_title("Input", fontsize=9)
    fig.colorbar(im1, ax=input_ax, location="left", fraction=0.04, pad=0.02)

    # Output heatmap (right)
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
    return image


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
    q_ax.set_xticks(range(len(q_labels)))
    q_ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=7)
    q_ax.set_yticks([])
    q_ax.set_title("Query Input", fontsize=9)
    fig.colorbar(im_q, ax=q_ax, location="left", fraction=0.04, pad=0.02)

    im_k = k_ax.imshow(k_heatmap, cmap="Blues", aspect="auto", vmin=0)
    k_ax.set_xticks(range(len(k_labels)))
    k_ax.set_xticklabels(k_labels, rotation=45, ha="right", fontsize=7)
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
    return image


def render_path_heatmaps(path: str, mlp_input_output: dict):
    matching_unembed_paths = [
        key for key in mlp_input_output.keys()
        if isinstance(key, str) and key.endswith(path)
    ]
    matching_select_paths = [
        key for key in mlp_input_output.keys()
        if isinstance(key, tuple)
        and len(key) == 4
        and isinstance(key[0], str)
        and isinstance(key[1], str)
        and (key[0].endswith(path) or key[1].endswith(path))
    ]
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

        out_tensor = torch.as_tensor(out_cache[0]).cpu()
        if out_tensor.numel() == 0:
            continue

        # Prefer the most impacted output tokens to mimic author plots.
        token_scores = out_tensor[:, :, :].abs().max(dim=0)[0].max(dim=1)[0]
        token_ids = torch.topk(
            token_scores,
            k=min(TOP_TOKENS_TO_PLOT, token_scores.numel())
        ).indices.tolist()

        st.markdown(f"**Unexplained MLP path:** `{lens_path}`")
        for token_id in token_ids:
            image = build_unembed_heatmap_image(inp_cache, out_cache, token_id)
            if image is None:
                continue
            st.image(
                image,
                caption=f"Input (left) and output (right) heatmap for token {token_id}",
                width=560
            )

    for select_path in sorted(matching_select_paths):
        payload = mlp_input_output.get(select_path)
        if payload is None or len(payload) != 3:
            continue
        q_inp_cache, k_inp_cache, out_cache = payload
        if not q_inp_cache or not k_inp_cache or not out_cache:
            continue

        image = build_select_heatmap_image(q_inp_cache, k_inp_cache, out_cache, cluster_idx=0)
        if image is None:
            continue

        q_path, k_path, attn_layer, attn_head = select_path
        st.markdown(
            f"**Select operator path:** `L{attn_layer}H{attn_head}`  "
            f"(query: `{q_path}`, key: `{k_path}`)"
        )
        st.image(
            image,
            caption="Query input (left), key input (middle), output (right)",
            width=760,
        )

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No runs found.")
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name
mlp_file = run_dir / "mlp_primitives" / "mlp_input_output.pt"

# Prefer per-task metrics file, fallback to run-root metrics.
metrics_file = run_dir / "mlp_primitives" / "metrics.jsonl"
if not metrics_file.exists():
    metrics_file = run_dir / "metrics.jsonl"

st.sidebar.markdown("### System")
col1, col2 = st.sidebar.columns(2)
col1.metric("CPU %", psutil.cpu_percent())
col2.metric("RAM %", psutil.virtual_memory().percent)

if not metrics_file.exists():
    st.warning(f"No metrics file found at `{run_dir}`.")
    st.stop()

df = load_metrics(metrics_file)
if df.empty:
    st.info("No primitive replacement metrics logged yet.")
    st.stop()

records = df.to_dict(orient="records")

paths = {}
path_order = []
global_path_idx = 0
global_total_paths = 0

for rec in records:
    task = rec.get("task")

    if task == "primitive_replacement":
        global_path_idx = int(rec.get("path_idx", global_path_idx) or 0)
        global_total_paths = int(rec.get("total_paths", global_total_paths) or 0)
        continue

    path = rec.get("path")
    if not path:
        continue

    if path not in paths:
        paths[path] = {
            "collected": 0,
            "collection_total": 0,
            "current_primitive": 0,
            "total_primitives": 0,
            "current_primitive_name": None,
            "current_acc_task": None,
            "replacement_done": False,
            "replacement_failed": False,
            "best_primitive": None,
            "best_accuracy": None,
        }
        path_order.append(path)

    info = paths[path]

    if task == "mlp_data_collection":
        info["collected"] = int(rec.get("collected", info["collected"]) or 0)
        info["collection_total"] = int(rec.get("total", info["collection_total"]) or 0)
    elif task == "primitive_search":
        if "current_primitive" in rec and not math.isnan(rec['current_primitive']):
            info["current_primitive"] = int(rec.get("current_primitive", info["current_primitive"]) or 0)
            info["total_primitives"] = int(rec.get("total_primitives", info["total_primitives"]) or 0)
            info["current_primitive_name"] = rec.get("primitive", info["current_primitive_name"])
            info["current_acc_task"] = rec.get("acc_task", info["current_acc_task"])
        if "failed" in rec and not math.isnan(rec['failed']):
            info["replacement_done"] = True
            info["replacement_failed"] = bool(rec.get("failed"))
            info["best_primitive"] = rec.get("best_primitive", info["best_primitive"])
            info["best_accuracy"] = rec.get("best_primitive_accuracy", info["best_accuracy"])

if global_total_paths > 0:
    global_progress = min(global_path_idx / global_total_paths, 1.0)
else:
    done_count = sum(1 for p in paths.values() if p["replacement_done"])
    global_total_paths = max(len(paths), 1)
    global_progress = min(done_count / global_total_paths, 1.0)
    global_path_idx = done_count

st.subheader("Global Progress")
st.progress(global_progress, text=f"Primitive replacement: {global_path_idx}/{global_total_paths} paths")

completed = sum(1 for p in paths.values() if p["replacement_done"] and not p["replacement_failed"])
failed = sum(1 for p in paths.values() if p["replacement_done"] and p["replacement_failed"])
in_progress = max(len(paths) - completed - failed, 0)
stat1, stat2, stat3 = st.columns(3)
stat1.metric("Succeeded", completed)
stat2.metric("Failed", failed)
stat3.metric("In Progress", in_progress)

st.subheader("Per-Path Progress")

for idx, path in enumerate(path_order, start=1):
    info = paths[path]
    collection_total = max(info["collection_total"], 1)
    primitive_total = max(info["total_primitives"], 1)
    collection_progress = min(info["collected"] / collection_total, 1.0)
    primitive_progress = min(info["current_primitive"] / primitive_total, 1.0)

    if info["replacement_done"] and not info["replacement_failed"]:
        status_icon = "✅"
        status_text = "Converted"
    elif info["replacement_done"] and info["replacement_failed"]:
        status_icon = "❌"
        status_text = "Failed"
    else:
        status_icon = "⏳"
        status_text = "Processing"

    with st.expander(f"{status_icon} [{status_text}] Path {idx}: {path}", expanded=False):
        st.markdown("**Data Collection**")
        st.progress(
            collection_progress,
            text=f"{info['collected']}/{info['collection_total']} examples collected"
            if info["collection_total"] > 0
            else "Waiting for data collection to start",
        )

        st.markdown("**Primitive Testing**")
        st.progress(
            primitive_progress,
            text=f"{info['current_primitive']}/{info['total_primitives']} primitives tested"
            if info["total_primitives"] > 0
            else "Waiting for primitive testing to start",
        )

        st.markdown("**Replacement Status**")
        best_primitive = info["best_primitive"] or "N/A"
        best_accuracy = info["best_accuracy"]
        best_accuracy_pct = f"{100 * best_accuracy:.2f}%" if best_accuracy is not None else "N/A"

        if info["replacement_done"] and not info["replacement_failed"]:
            st.success(
                f"Primitive replacement succeeded. Best primitive: `{best_primitive}` "
                f"with accuracy: **{best_accuracy_pct}**."
            )
        elif info["replacement_done"] and info["replacement_failed"]:
            st.warning(
                f"Primitive replacement failed. Best candidate: `{best_primitive}` "
                f"with accuracy: **{best_accuracy_pct}**."
            )
        else:
            current_name = info["current_primitive_name"] or "N/A"
            current_acc = info["current_acc_task"]
            current_acc_pct = f"{100 * current_acc:.2f}%" if current_acc is not None else "N/A"
            st.info(
                f"Primitive replacement in progress. Current primitive: `{current_name}` "
                f"(task accuracy: **{current_acc_pct}**)."
            )

        should_show_heatmap = (
            (not info["replacement_done"]) or info["replacement_failed"]
        )
        if should_show_heatmap:
            st.markdown("**Unexplained MLP Heatmap**")
            with st.spinner("Preparing heatmap..."):
                mlp_input_output = load_mlp_heatmap_cache(str(mlp_file))

            if mlp_input_output is None:
                st.info("Heatmaps are not ready yet. Waiting for `mlp_input_output.pt`...")
            else:
                render_path_heatmaps(path, mlp_input_output)