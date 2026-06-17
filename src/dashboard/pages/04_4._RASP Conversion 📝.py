"""Dashboard for inspecting D-RASP decompilation results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics
from utils.mlp_heatmap_rendering import load_mlp_heatmap_cache, render_path_heatmaps

st.set_page_config(layout="wide")
st.title("RASP Decompilation")

st.markdown(
    """
The final decompilation stage combines pruning, MLP primitive replacement, and attention primitive
replacement into a symbolic D-RASP program. Unexplained selector matrices appear as circled labels
(`\\circled{a}`, `\\circled{b}`, …) and are linked to saved heatmaps from the attention primitive stage.
"""
)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _pct(value: float | None) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{100 * float(value):.1f}%"


def _discover_rasp_runs() -> list[str]:
    runs = []
    if not PRUNING_EXPERIMENT_DIR.exists():
        return runs
    for run_dir in sorted(PRUNING_EXPERIMENT_DIR.iterdir()):
        if run_dir.is_dir() and (run_dir / "rasp" / "generated_code.txt").exists():
            runs.append(run_dir.name)
    return runs


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=5)
def _load_manifest(manifest_path: str) -> dict | None:
    return _load_json(Path(manifest_path))


runs = _discover_rasp_runs()
if not runs:
    st.warning(
        "No RASP outputs found. Run `python src/run_rasp.py --config src/rasp/configs/rasp_test.yaml` "
        "after pruning, MLP, and attention primitive stages."
    )
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name
rasp_dir = run_dir / "rasp"
manifest_path = rasp_dir / "artifacts" / "manifest.json"
output_path = rasp_dir / "output.json"
code_path = rasp_dir / "generated_code.txt"

manifest = _load_manifest(str(manifest_path))
output = _load_json(output_path)

if manifest is None and output is None:
    st.error(f"No RASP summary found under `{rasp_dir}`.")
    st.stop()

coverage = (manifest or output or {}).get("coverage", {})
program_lines = []
if manifest and manifest.get("program", {}).get("lines"):
    program_lines = manifest["program"]["lines"]
elif code_path.exists():
    program_lines = code_path.read_text().splitlines()

metrics_df = load_metrics(rasp_dir / "metrics.jsonl")

st.sidebar.markdown("### Run Info")
if output:
    st.sidebar.metric("Program lines", output.get("line_count", len(program_lines)))
    st.sidebar.metric("Circled matrices", output.get("circled_matrix_count", 0))
if not metrics_df.empty:
    st.sidebar.write(f"Metrics events: {len(metrics_df)}")

st.subheader("D-RASP Program")
if program_lines:
    st.code("\n".join(program_lines), language=None)
else:
    st.info("No generated program found.")

st.subheader("Coverage Summary")
col1, col2, col3, col4 = st.columns(4)

att_cov = coverage.get("attention", {})
lm_cov = coverage.get("lm_head", {})
mlp_cov = coverage.get("mlp", {})

col1.metric(
    "Attention predefined",
    att_cov.get("predefined", 0),
    delta=_pct(att_cov.get("pct_predefined")),
)
col2.metric(
    "Attention rounded",
    att_cov.get("rounded", 0),
    delta=_pct(att_cov.get("pct_rounded")),
)
col3.metric(
    "LM head predefined",
    lm_cov.get("predefined", 0),
    delta=_pct(lm_cov.get("pct_predefined")),
)
col4.metric(
    "MLP converted",
    f"{mlp_cov.get('converted', 0)}/{mlp_cov.get('total', 0)}",
    delta=_pct(mlp_cov.get("pct_converted")),
)

upstream = coverage.get("upstream", {})
if upstream:
    st.markdown("### Upstream Metrics")
    ucol1, ucol2, ucol3, ucol4 = st.columns(4)
    pruning = upstream.get("pruning", {})
    att_stats = upstream.get("att", {})
    ucol1.metric("Pruning acc_match", _pct(pruning.get("acc_match")))
    ucol2.metric("Pruning acc_task", _pct(pruning.get("acc_task")))
    acc_match = (att_stats.get("acc_match") or {}).get("after")
    ucol3.metric("Att acc_match (after)", _pct(acc_match))
    acc_task = (att_stats.get("acc") or {}).get("after")
    ucol4.metric("Att acc (after)", _pct(acc_task))

circled = (manifest or {}).get("circled_matrices", [])
if circled:
    st.subheader("Circled Matrices")
    for entry in circled:
        label = entry.get("label", "?")
        code_var = entry.get("code_var", "")
        with st.expander(f"\\circled{{{label}}} — {code_var}", expanded=len(circled) <= 3):
            meta_cols = st.columns(3)
            if entry.get("layer") is not None:
                meta_cols[0].write(f"Layer {entry['layer']}, Head {entry['head']}")
            if entry.get("q_path"):
                meta_cols[1].write(f"Q: `{entry['q_path']}`")
            if entry.get("k_path"):
                meta_cols[2].write(f"K: `{entry['k_path']}`")
            if entry.get("inp_path"):
                st.write(f"Input: `{entry['inp_path']}`")

            heatmap_paths = entry.get("heatmap_paths", {})
            if heatmap_paths:
                img_cols = st.columns(min(len(heatmap_paths), 4))
                for idx, (kind, rel_path) in enumerate(heatmap_paths.items()):
                    img_path = run_dir / rel_path
                    if img_path.exists():
                        with img_cols[idx % len(img_cols)]:
                            st.caption(kind.replace("_", " ").title())
                            st.image(str(img_path), use_container_width=True)
            else:
                st.caption("No heatmap PNGs found for this interaction.")

unexplained = (manifest or {}).get("unexplained_mlp", [])
mlp_io_path = run_dir / "mlp_primitives" / "mlp_input_output.pt"
mlp_input_output = load_mlp_heatmap_cache(str(mlp_io_path))

if unexplained:
    st.subheader("Unexplained MLP Paths")
    st.caption(
        "Paths where primitive replacement failed or was skipped. "
        "Expand a path to view LogitLens heatmaps (same as the MLP Primitives page)."
    )

    for entry in unexplained:
        path = entry.get("path", "?")
        status = entry.get("status", "unknown")
        primitive = entry.get("primitive", "N/A")
        accuracy = entry.get("accuracy")
        acc_str = f"{100 * accuracy:.1f}%" if accuracy is not None else "N/A"
        cached = entry.get("logit_lens_cached", entry.get("has_logit_lens", False))
        cache_icon = "📊" if cached else "—"

        with st.expander(
            f"{cache_icon} `{path}` — {status} (best: {primitive}, acc: {acc_str})",
            expanded=len(unexplained) <= 2,
        ):
            meta1, meta2, meta3 = st.columns(3)
            meta1.write(f"**Status:** {status}")
            meta2.write(f"**Best primitive:** `{primitive}`")
            meta3.write(f"**Best accuracy:** {acc_str}")

            cache_keys = entry.get("logit_lens_cache_keys", [])
            if cache_keys:
                st.caption(f"LogitLens cache keys: {', '.join(f'`{k}`' for k in cache_keys)}")

            if mlp_input_output is None:
                st.info("LogitLens cache not available. Run the MLP primitive stage to generate `mlp_input_output.pt`.")
            else:
                render_path_heatmaps(path, mlp_input_output, key_prefix=f"rasp_{run_name}_")

if not metrics_df.empty:
    st.subheader("RASP Metrics")
    st.dataframe(metrics_df, use_container_width=True)
