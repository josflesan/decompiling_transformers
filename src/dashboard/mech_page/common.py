from __future__ import annotations

import os
import warnings
from pathlib import Path

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

from data.CountDataset import CountCorruption
from mechanistic.core.attention_inspection import (
    inspect_ov_circuit,
    inspect_ov_eigenspectrum,
    inspect_qk_circuit,
    plot_positional_attention,
)
from mechanistic.utilities.mechinterp_dataclasses import CircuitNode
from mechanistic.utilities.mechinterp_utils import attn_v, attn_z, mlp_out, resid_post
from tasks.registry import get_task
from transformer_lens.model_bridge import TransformerBridge
from utilities.core import TaskConfig

from utils.constants import PRUNING_EXPERIMENT_DIR

_SRC_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SRC_ROOT.parent

CONFIG_SEARCH_DIRS: list[Path] = [
    _SRC_ROOT / "mechanistic" / "config",
    _SRC_ROOT / "pruning" / "configs",
    _SRC_ROOT / "primitives_mlp" / "configs",
]

MANUAL_ANALYSIS_FILENAME = "manual_analysis.md"

TASK_SIDEBAR_HELP: dict[str, str] = {
    "counting": (
        "Models are trained to **continue counting** after `<sep>`: the prompt gives a start integer, an end "
        "integer, then the body counts upward to the end. Token IDs are digit strings from the task vocabulary; "
        "position IDs respect `max_test_length` for evaluation windows."
    ),
}


def repo_root() -> Path:
    return _REPO_ROOT


def safe_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def default_task_config() -> TaskConfig:
    return TaskConfig(
        name="counting",
        train_length_range=[50, 150],
        val_length_range=[50, 150],
        max_test_length=150,
    )


def task_config_from_yaml_dict(raw: dict) -> TaskConfig | None:
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


def find_yaml_for_run(run_name: str) -> tuple[dict, Path] | None:
    for d in CONFIG_SEARCH_DIRS:
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


def resolve_device(choice: str) -> str:
    if choice == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return choice


def task_sidebar_markdown(tc: TaskConfig) -> str:
    return (
        f"**Task:** `{tc.name}`  \n"
        f"**Train span length range:** `{tc.train_length_range}`  \n"
        f"**Val span length range:** `{tc.val_length_range}`  \n"
        f"**Max test length:** `{tc.max_test_length}`  \n\n"
    )


@st.cache_resource
def load_transformer_bridge(model_path: str, device: str, compatibility_mode: bool):
    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
    path = Path(model_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
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


def embed_html_file(path: Path, height: int) -> None:
    html = path.read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=False)


def wrap_iframe_html(fragment: str) -> str:
    if "<html" in fragment.lower():
        return fragment
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;padding:0;overflow:hidden;}</style>"
        f"</head><body>{fragment}</body></html>"
    )


def embed_html_snippet(html: str, *, height: int, scrolling: bool = False) -> None:
    components.html(wrap_iframe_html(html), height=height, scrolling=scrolling)


def plotly_fig_embed_html(fig) -> str:
    """Self-contained Plotly HTML for iframe embed (avoids st.plotly_chart subplot layout issues)."""
    return pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": True},
    )


def streamlit_plotly_chart(fig, *, key: str) -> None:
    """Render a Plotly figure with layout height respected (tall subplot grids)."""
    fig.update_layout(template="plotly_white")
    layout_height = fig.layout.height
    try:
        if layout_height is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                height=int(layout_height),
                theme=None,
                key=key,
            )
        else:
            st.plotly_chart(
                fig,
                width="stretch",
                height="content",
                theme=None,
                key=key,
            )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def prepare_fig_for_streamlit(
    fig,
    *,
    subplot_cell_px: int | None = None,
    n_subplot_rows: int | None = None,
) -> None:
    if fig is None:
        return
    layout_kw: dict = dict(
        autosize=True,
        width=None,
        margin=dict(l=48, r=24, t=56, b=40),
    )
    if subplot_cell_px is not None and n_subplot_rows is not None:
        layout_kw["height"] = subplot_cell_px * max(1, n_subplot_rows) + 52
    elif fig.layout.height is None:
        layout_kw["height"] = None
    fig.update_layout(**layout_kw)


def attribution_html_saved(attr_dir: Path, html_filename: str) -> bool:
    return (attr_dir / html_filename).is_file()


def display_attribution_block(
    title: str,
    html_filename: str,
    attr_dir: Path,
    fig,
    *,
    embed_height: int = 300,
) -> None:
    html_path = attr_dir / html_filename
    saved = attribution_html_saved(attr_dir, html_filename)

    if not saved and fig is None:
        st.caption("No saved figure yet. Run attribution analysis to generate this plot.")
        return

    if saved:
        st.markdown(f"**{title}**")

    if fig is not None:
        prepare_fig_for_streamlit(fig)
        streamlit_plotly_chart(fig, key=f"{title}_{html_filename}_live")
    else:
        embed_html_file(html_path, height=embed_height)


def layer_head_html(prefix: str, layer: int, head: int) -> str:
    return f"{prefix}_L{layer}H{head}.html"


def save_plotly_figure(fig, path: Path, *, for_streamlit: bool = True) -> None:
    fig.update_layout(template="plotly_white")
    fig.for_each_xaxis(lambda ax: ax.update(matches=None))
    fig.for_each_yaxis(lambda ax: ax.update(matches=None))
    if for_streamlit:
        prepare_fig_for_streamlit(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(path.with_suffix(".json"))
    fig.write_html(path, config={"responsive": True})


def load_plotly_figure(html_path: Path):
    json_path = html_path.with_suffix(".json")
    if json_path.is_file():
        return pio.read_json(json_path)
    return None


def resolve_artifact_path(
    artifact_dir: Path,
    primary: str,
    *,
    legacy: tuple[str, ...] = (),
) -> Path | None:
    for name in (primary, *legacy):
        path = artifact_dir / name
        if path.is_file() or path.with_suffix(".json").is_file():
            return path
    return None


def save_html_content(html: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def attention_state_key(run_name: str, corruption: str, suffix: str) -> str:
    return f"mech_attn_show_{run_name}_{corruption}_{suffix}"


def attention_body_key(show_key: str) -> str:
    return f"{show_key}_body"


def display_inspection_block(
    title: str,
    html_filename: str,
    artifact_dir: Path,
    fig,
    *,
    use_plotly: bool = True,
    subplot_cell_px: int | None = None,
    n_subplot_rows: int | None = None,
    html_embed_height: int = 480,
    html_embed_scrolling: bool = False,
    legacy_html_filenames: tuple[str, ...] = (),
    session_show_key: str | None = None,
) -> None:
    html_path = resolve_artifact_path(
        artifact_dir,
        html_filename,
        legacy=legacy_html_filenames,
    )
    session_body = None
    if session_show_key:
        session_body = st.session_state.get(attention_body_key(session_show_key))

    if fig is None and html_path is not None and use_plotly:
        fig = load_plotly_figure(html_path)

    session_html = None
    if fig is None and session_body and not use_plotly:
        if str(session_body).lstrip().startswith("{"):
            try:
                fig = pio.from_json(session_body)
            except (ValueError, TypeError):
                pass
        else:
            session_html = session_body
    elif fig is None and session_body and use_plotly and str(session_body).lstrip().startswith("{"):
        try:
            fig = pio.from_json(session_body)
        except (ValueError, TypeError):
            pass

    has_content = fig is not None or html_path is not None or session_html

    if not has_content:
        st.caption("No saved figure yet. Run this analysis to generate the plot.")
        return

    st.markdown(f"**{title}**")
    chart_key = f"{session_show_key or title}_{html_filename}".replace(".", "_")

    if fig is not None:
        if use_plotly:
            prepare_fig_for_streamlit(
                fig,
                subplot_cell_px=subplot_cell_px,
                n_subplot_rows=n_subplot_rows,
            )
            streamlit_plotly_chart(fig, key=f"{chart_key}_live")
        else:
            embed_html_snippet(
                plotly_fig_embed_html(fig),
                height=html_embed_height,
                scrolling=html_embed_scrolling,
            )
    elif fig is None and html_path is not None:
        if use_plotly:
            loaded = load_plotly_figure(html_path)
            if loaded is not None:
                prepare_fig_for_streamlit(
                    loaded,
                    subplot_cell_px=subplot_cell_px,
                    n_subplot_rows=n_subplot_rows,
                )
                streamlit_plotly_chart(loaded, key=f"{chart_key}_saved")
            else:
                embed_html_snippet(
                    html_path.read_text(encoding="utf-8"),
                    height=html_embed_height,
                    scrolling=html_embed_scrolling,
                )
        else:
            embed_html_snippet(
                html_path.read_text(encoding="utf-8"),
                height=html_embed_height,
                scrolling=html_embed_scrolling,
            )
    elif session_html:
        embed_html_snippet(
            session_html,
            height=html_embed_height,
            scrolling=html_embed_scrolling,
        )


PER_HEAD_PREFIXES = ("ov_circuit", "ov_eigenspectrum", "qk_circuit", "positional_attention")


def per_head_plot_paths(artifact_dir: Path, layer: int, head: int) -> dict[str, Path]:
    return {
        prefix: artifact_dir / layer_head_html(prefix, layer, head)
        for prefix in PER_HEAD_PREFIXES
    }


def per_head_plots_complete(artifact_dir: Path, layer: int, head: int) -> bool:
    return all(p.is_file() for p in per_head_plot_paths(artifact_dir, layer, head).values())


def run_per_head_inspections(
    model,
    corrupted,
    layer: int,
    head: int,
    artifact_dir: Path,
    *,
    save_html: bool,
    force_recompute: bool,
) -> list[str]:
    paths = per_head_plot_paths(artifact_dir, layer, head)
    saved_names: list[str] = []

    def _needs_run(path: Path) -> bool:
        return force_recompute or not path.is_file()

    if _needs_run(paths["ov_circuit"]):
        fig = inspect_ov_circuit(model, layer, head)
        if save_html and fig is not None:
            save_plotly_figure(fig, paths["ov_circuit"])
            saved_names.append(paths["ov_circuit"].name)

    if _needs_run(paths["ov_eigenspectrum"]):
        fig = inspect_ov_eigenspectrum(model, layer, head)
        if save_html and fig is not None:
            save_plotly_figure(fig, paths["ov_eigenspectrum"])
            saved_names.append(paths["ov_eigenspectrum"].name)

    if _needs_run(paths["qk_circuit"]):
        fig = inspect_qk_circuit(model, layer, head)
        if save_html and fig is not None:
            save_plotly_figure(fig, paths["qk_circuit"])
            saved_names.append(paths["qk_circuit"].name)

    if _needs_run(paths["positional_attention"]):
        fig = plot_positional_attention(model, corrupted, layer, head)
        if save_html and fig is not None:
            save_plotly_figure(fig, paths["positional_attention"])
            saved_names.append(paths["positional_attention"].name)

    return saved_names


def resolve_model_dims(model_path: str, device: str, compatibility_mode: bool) -> tuple[int, int]:
    """Layer/head counts from the run's loaded TransformerBridge (not YAML defaults)."""
    model = load_transformer_bridge(model_path, device, compatibility_mode)
    return int(model.cfg.n_layers), int(model.cfg.n_heads)


def build_circuit_node_catalog(n_layers: int, n_heads: int) -> dict[str, CircuitNode]:
    """All standard hook sites for path patching, keyed by human-readable labels."""
    catalog: dict[str, CircuitNode] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            catalog[f"attn_z L{layer} H{head}"] = CircuitNode(
                name=attn_z(layer), layer_idx=layer, head_idx=head
            )
            catalog[f"attn_v L{layer} H{head}"] = CircuitNode(
                name=attn_v(layer), layer_idx=layer, head_idx=head
            )
        catalog[f"mlp_out L{layer}"] = CircuitNode(name=mlp_out(layer), layer_idx=layer)
        catalog[f"resid_post L{layer}"] = CircuitNode(name=resid_post(layer), layer_idx=layer)
    return catalog


def render_circuit_node_checkboxes(
    catalog: dict[str, CircuitNode],
    *,
    role: str,
    run_name: str,
    corruption: str,
) -> dict[str, CircuitNode]:
    """Checkboxes grouped by layer inside an expander (sender or receiver selection)."""
    layers = sorted({node.layer_idx for node in catalog.values() if node.layer_idx is not None})
    selected: dict[str, CircuitNode] = {}

    with st.expander(f"Select {role} nodes", expanded=False):
        for layer in layers:
            layer_keys = [k for k, n in catalog.items() if n.layer_idx == layer]
            if not layer_keys:
                continue
            st.markdown(f"**Layer {layer}**")
            cols = st.columns(4)
            for i, key in enumerate(layer_keys):
                with cols[i % 4]:
                    if st.checkbox(
                        key,
                        key=f"mech_path_{role}_{run_name}_{corruption}_{key}",
                    ):
                        selected[key] = catalog[key]
    return selected


def load_counting_corrupt_batch(
    task_cfg: TaskConfig,
    corruption: CountCorruption,
    batch_size: int,
    device: str,
    compat: bool,
    model_path: str,
):
    warnings.filterwarnings("ignore")
    if device == "mps":
        os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
    torch.set_grad_enabled(False)

    model = load_transformer_bridge(str(model_path), device, compat)
    task = get_task("counting", task_cfg)
    tokenizer, datasets = task.build()
    train_ds = datasets["train"]
    corrupted = train_ds.get_corrupted(corruption, batch_size=int(batch_size))
    return model, tokenizer, corrupted
