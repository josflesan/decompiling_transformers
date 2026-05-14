"""Mechanistic interpretability dashboard (attribution, activation patching placeholder)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.corruption_descriptions import load_corruption_description  # noqa: E402
from data.CountDataset import CountCorruption  # noqa: E402
from mechanistic.core.attribution import (  # noqa: E402
    attention_head_attribution,
    layerwise_attribution,
    residual_stream_attribution,
)
from tasks.registry import get_task  # noqa: E402
from transformer_lens.model_bridge import TransformerBridge  # noqa: E402
from utilities.core import TaskConfig  # noqa: E402

from utils.constants import PRUNING_EXPERIMENT_DIR  # noqa: E402

st.set_page_config(layout="wide")

_CONFIG_SEARCH_DIRS: list[Path] = [
    _SRC_ROOT / "mechanistic" / "config",
    _SRC_ROOT / "pruning" / "configs",
    _SRC_ROOT / "primitives_mlp" / "configs",
]

_ATTRIBUTION_HTML = {
    "Residual Stream Attribution": "residual_stream_attribution.html",
    "Residual Stream Component-Level Attribution": "residual_stream_component_attribution.html",
    "Attention Head Attribution": "attention_head_attribution.html",
}

MANUAL_ANALYSIS_FILENAME = "manual_analysis.md"

_TASK_SIDEBAR_HELP: dict[str, str] = {
    "counting": (
        "Models are trained to **continue counting** after `<sep>`: the prompt gives a start integer, an end "
        "integer, then the body counts upward to the end. Token IDs are digit strings from the task vocabulary; "
        "position IDs respect `max_test_length` for evaluation windows."
    ),
}


def _task_sidebar_markdown(tc: TaskConfig) -> str:
    return (
        f"**Task:** `{tc.name}`  \n"
        f"**Train span length range:** `{tc.train_length_range}`  \n"
        f"**Val span length range:** `{tc.val_length_range}`  \n"
        f"**Max test length:** `{tc.max_test_length}`  \n\n"
    )


def _repo_root() -> Path:
    return _SRC_ROOT.parent


def _safe_display_path(path: Path) -> str:
    """Repo-relative path for UI; avoids relative_to errors when one path is cwd-relative."""
    try:
        return path.resolve().relative_to(_repo_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _default_task_config() -> TaskConfig:
    return TaskConfig(
        name="counting",
        train_length_range=[50, 150],
        val_length_range=[50, 150],
        max_test_length=150,
    )


def _task_config_from_yaml_dict(raw: dict) -> TaskConfig | None:
    tc = raw.get("task_config")
    if not isinstance(tc, dict):
        return None
    try:
        return TaskConfig(
            name=str(tc["name"]),
            train_length_range=list(tc["train_length_range"]),
            val_length_range=list(tc["val_length_range"]),
            max_test_length=int(tc["max_test_length"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _find_yaml_for_run(run_name: str) -> tuple[dict, Path] | None:
    """First YAML (mechanistic configs, then pruning, then MLP) whose exp_name matches the run folder."""
    for d in _CONFIG_SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.yaml")):
            try:
                raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(raw, dict):
                continue
            if raw.get("exp_name") == run_name:
                return raw, p
    return None


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return choice


@st.cache_resource
def _load_transformer_bridge(model_path: str, device: str, compatibility_mode: bool):
    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
    path = Path(model_path)
    if not path.is_absolute():
        path = _repo_root() / path
    model = TransformerBridge.boot_transformers(str(path), device=device)
    model.eval()
    if compatibility_mode:
        model.enable_compatibility_mode(
            fold_ln=True,
            center_unembed=True,
            center_writing_weights=True,
            refactor_factored_attn_matrices=True,
        )
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _embed_html_file(path: Path, height: int) -> None:
    html = path.read_text(encoding="utf-8")
    # Tight iframe: responsive Plotly HTML (from attribution write) scales to width; no inner scrollbars.
    components.html(html, height=height, scrolling=False)


def _attribution_html_saved(attr_dir: Path, html_filename: str) -> bool:
    return (attr_dir / html_filename).is_file()


def _prepare_fig_for_streamlit(fig) -> None:
    """Ensure Plotly fills column width without fixed pixel width (helps older in-memory figures)."""
    if fig is None:
        return
    fig.update_layout(autosize=True, width=None)


def _display_attribution_block(
    title: str,
    html_filename: str,
    attr_dir: Path,
    fig,
    *,
    embed_height: int = 300,
) -> None:
    html_path = attr_dir / html_filename
    saved = _attribution_html_saved(attr_dir, html_filename)

    if not saved and fig is None:
        st.caption("No saved figure yet. Run attribution analysis to generate this plot.")
        return

    if saved:
        st.markdown(f"**{title}**")

    if fig is not None:
        _prepare_fig_for_streamlit(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        _embed_html_file(html_path, height=embed_height)


st.title("Mechanistic interpretability")

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No experiment folders found under `src/out`. Run pruning (or create a folder) first.")
    st.stop()

st.sidebar.markdown("### Experiment (same as Pruning / MLP)")
run_name = st.sidebar.selectbox("Select run", runs)

run_yaml = _find_yaml_for_run(run_name)
if run_yaml is None:
    st.sidebar.error("No matching YAML")
    st.error(
        f"No config file declares `exp_name: {run_name!r}`. "
        "Add or adjust a YAML under `src/mechanistic/config/` (or pruning / MLP configs) so it matches this run folder name."
    )
    model_path: str | None = None
    task_cfg: TaskConfig | None = None
    raw_config: dict = {}
else:
    raw_config, _matched_path = run_yaml
    model_path = raw_config.get("model_path")
    if not model_path:
        st.error("Matched YAML has no `model_path`.")
        model_path = None
    task_cfg = _task_config_from_yaml_dict(raw_config) or _default_task_config()

corruption: CountCorruption | None = None
if task_cfg is not None:
    st.sidebar.markdown("### Training task")
    st.sidebar.markdown(_task_sidebar_markdown(task_cfg))
    
    with st.sidebar.expander("Task Description", expanded=False):
        extra = _TASK_SIDEBAR_HELP.get(
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
device = _resolve_device(device_choice)

compat = st.sidebar.checkbox(
    "TransformerLens compatibility mode",
    value=False,
    help="Fold LN and center weights (matches `exploration.ipynb` setup).",
)

tab_attr, tab_patch = st.tabs(["Attribution Analysis", "Activation Patching"])

with tab_attr:
    st.markdown(
        """
        These experiments assess the influence of different layer components on the residual stream
        of the model after the **final layer**. This lets us gauge how much each component contributes to
        the final prediction. Note that this is similar to the **LogitLens** attribution method.
        """
    )

    if model_path is None or task_cfg is None:
        st.stop()

    attr_dir = _repo_root() / PRUNING_EXPERIMENT_DIR / run_name / "mechanistic" / "attribution"

    if task_cfg.name != "counting":
        st.warning(
            "Attribution corruption batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert corruption is not None
    artifact_dir = attr_dir / corruption.name

    default_bs = int(raw_config.get("batch_size", 25)) if raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_attr_bs_{run_name}",
    )
    save_html = st.checkbox(
        "Save Plot",
        value=True,
    )

    run_attr = st.button(
        "Run attribution analysis",
        type="primary",
        disabled=task_cfg.name != "counting",
    )

    fig_residual = fig_layer = fig_heads = None
    if run_attr:
        if task_cfg.name != "counting":
            st.error("Switch `task_config.name` to `counting` in the matched YAML.")
        else:
            warnings.filterwarnings("ignore")
            if device == "mps":
                os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
            torch.set_grad_enabled(False)

            with st.spinner("Loading model…"):
                try:
                    model = _load_transformer_bridge(str(model_path), device, compat)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            with st.spinner("Building corrupted counting batch…"):
                task = get_task("counting", task_cfg)
                _tokenizer, datasets = task.build()
                train_ds = datasets["train"]
                corrupted = train_ds.get_corrupted(
                    corruption,
                    batch_size=int(batch_size),
                )

            common_kw = dict(
                model=model,
                tokens=corrupted.clean_tokens,
                position_ids=corrupted.clean_pos,
                answer_tokens=corrupted.answer_tokens[:, 0],
                exp_name=run_name,
                device=device,
                save_html=save_html,
                artifact_subdir=corruption.name,
            )

            with st.spinner("Running all three attribution passes…"):
                fig_residual = residual_stream_attribution(**common_kw)
                fig_layer = layerwise_attribution(**common_kw)
                fig_heads = attention_head_attribution(**common_kw)

            if save_html:
                st.success(f"Wrote HTML under `{_safe_display_path(artifact_dir)}`.")

    plot_specs: list[tuple[str, str, object]] = []
    for title, fname in _ATTRIBUTION_HTML.items():
        fig = None
        if title == "Residual Stream Attribution":
            fig = fig_residual
        elif title == "Residual Stream Component-Level Attribution":
            fig = fig_layer
        elif title == "Attention Head Attribution":
            fig = fig_heads
        plot_specs.append((title, fname, fig))

    col1, col2, col3 = st.columns(3, gap="small")
    for col, (title, fname, fig) in zip((col1, col2, col3), plot_specs, strict=True):
        with col:
            _display_attribution_block(title, fname, artifact_dir, fig, embed_height=300)

    all_plots_ready = all(_attribution_html_saved(artifact_dir, fname) for _, fname, _ in plot_specs)

    if all_plots_ready:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
        notes_key = f"mech_manual_analysis_{run_name}_{corruption.name}"
        init_flag = f"_mech_manual_notes_init_{run_name}_{corruption.name}"
        if init_flag not in st.session_state:
            st.session_state[notes_key] = (
                notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
            )
            st.session_state[init_flag] = True

        # st.divider()
        st.markdown("#### Interpretation and Notes")
        st.caption(f"Notes for corruption **{corruption.name}** (saved next to that corruption's plots).")
        st.text_area(
            "Interpretation and notes for these attribution plots",
            height=240,
            key=notes_key,
            label_visibility="visible",
        )
        if st.button("Save interpretation", key=f"mech_save_manual_{run_name}_{corruption.name}"):
            notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
            st.success(f"Saved to `{_safe_display_path(notes_path)}`.")

    st.caption(
        "The forward pass uses **clean** tokens and positions; **corruption** defines the counterfactual "
        "batch construction and the **answer-token** column used for logit directions, so attribution depends on the "
        "corruption setting. Plots and notes are stored under `mechanistic/attribution/<CORRUPTION_NAME>/`."
    )

with tab_patch:
    st.subheader("Activation patching")
    st.info(
        "Placeholder for activation patching experiments (e.g. from `mechanistic/core/activation_patching.py`). "
        "This tab will host runs and visualizations once wired up."
    )
