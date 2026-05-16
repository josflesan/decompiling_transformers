"""Mechanistic interpretability dashboard (attribution and attention inspection)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.corruption_descriptions import load_corruption_description  # noqa: E402
from data.CountDataset import CountCorruption  # noqa: E402
from mech_page.common import (  # noqa: E402
    TASK_SIDEBAR_HELP,
    default_task_config,
    find_yaml_for_run,
    resolve_device,
    task_config_from_yaml_dict,
    task_sidebar_markdown,
)
from mech_page.context import MechPageContext  # noqa: E402
from mech_page.tabs import (  # noqa: E402
    render_activation_patching_tab,
    render_attribution_tab,
    render_attention_tab,
    render_path_patching_tab,
)
from utils.constants import PRUNING_EXPERIMENT_DIR  # noqa: E402

st.set_page_config(layout="wide")

st.title("Mechanistic interpretability")

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No experiment folders found under `src/out`. Run pruning (or create a folder) first.")
    st.stop()

st.sidebar.markdown("### Experiment (same as Pruning / MLP)")
run_name = st.sidebar.selectbox("Select run", runs)

run_yaml = find_yaml_for_run(run_name)
if run_yaml is None:
    st.sidebar.error("No matching YAML")
    st.error(
        f"No config file declares `exp_name: {run_name!r}`. "
        "Add or adjust a YAML under `src/mechanistic/config/` (or pruning / MLP configs) so it matches this run folder name."
    )
    model_path: str | None = None
    task_cfg = None
    raw_config: dict = {}
else:
    raw_config, _matched_path = run_yaml
    model_path = raw_config.get("model_path")
    if not model_path:
        st.error("Matched YAML has no `model_path`.")
        model_path = None
    task_cfg = task_config_from_yaml_dict(raw_config) or default_task_config()

corruption: CountCorruption | None = None
if task_cfg is not None:
    st.sidebar.markdown("### Training task")
    st.sidebar.markdown(task_sidebar_markdown(task_cfg))

    with st.sidebar.expander("Task Description", expanded=False):
        extra = TASK_SIDEBAR_HELP.get(
            task_cfg.name,
            f"See `tasks/` and `data/` for how `{task_cfg.name}` is defined in this codebase.",
        )
        st.markdown(extra)

    st.sidebar.markdown("### Corruption")
    corruption = st.sidebar.selectbox(
        "Corruption mode",
        options=list(CountCorruption),
        format_func=lambda c: c.name,
        key=f"mech_sidebar_corruption_{run_name}",
        help="Used for attribution batches and for which subfolder stores plots and notes.",
    )
    with st.sidebar.expander("What this corruption means", expanded=False):
        st.markdown(load_corruption_description(task_cfg.name, corruption.name))

device_choice = st.sidebar.selectbox(
    "Device",
    ["auto", "mps", "cuda", "cpu"],
    index=0,
)
device = resolve_device(device_choice)

compat = st.sidebar.checkbox(
    "TransformerLens compatibility mode",
    value=False,
    help="Fold LN and center weights (matches `exploration.ipynb` setup).",
)

ctx = MechPageContext(
    run_name=run_name,
    model_path=model_path,
    task_cfg=task_cfg,
    raw_config=raw_config,
    corruption=corruption,
    device=device,
    compat=compat,
)

tab_attr, tab_attention, tab_patch, tab_path = st.tabs(
    [
        "Attribution Analysis",
        "Attention Analysis",
        "Activation Patching",
        "Path Patching",
    ]
)

with tab_attr:
    render_attribution_tab(ctx)

with tab_attention:
    render_attention_tab(ctx)

with tab_patch:
    render_activation_patching_tab(ctx)

with tab_path:
    render_path_patching_tab(ctx)
