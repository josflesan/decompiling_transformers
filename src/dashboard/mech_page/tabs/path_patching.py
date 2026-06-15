from __future__ import annotations

import json
import uuid
from typing import Any

import plotly.io as pio
import streamlit as st

from mechanistic.core.path_patching import path_patch
from mechanistic.utilities.metrics import logit_diff_metric
from mechanistic.utilities.mechinterp_viz import plot_path_patching_matrix

from mech_page.common import (
    MANUAL_ANALYSIS_FILENAME,
    build_circuit_node_catalog,
    load_counting_corrupt_batch,
    render_circuit_node_checkboxes,
    repo_root,
    resolve_model_dims,
    safe_display_path,
    save_plotly_figure,
    streamlit_plotly_chart,
)
from mech_page.context import MechPageContext
from utils.constants import PRUNING_EXPERIMENT_DIR

PATH_PATCHING_SUBFOLDER = "path_patching"
PINNED_MANIFEST = "pinned_runs.json"


def _session_keys(run_name: str, corruption: str) -> tuple[str, str]:
    base = f"mech_path_{run_name}_{corruption}"
    return f"{base}_current", f"{base}_pinned"


def _default_title(senders: list[str], receivers: list[str]) -> str:
    return f"Path patch ({len(senders)} senders → {len(receivers)} receivers)"


def _fig_from_entry(entry: dict[str, Any]):
    return pio.from_json(entry["fig_json"])


def _display_path_patch_fig(fig, *, chart_key: str) -> None:
    fig.update_layout(template="plotly_white", autosize=True)
    streamlit_plotly_chart(fig, key=chart_key)


def _save_current_plot(fig, artifact_dir, entry: dict[str, Any], *, key_prefix: str) -> None:
    if st.button("Save current plot", key=f"{key_prefix}_save_btn"):
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in entry["title"]
        ).strip("_") or "path_patching"
        out = artifact_dir / f"{safe_name}.html"
        save_plotly_figure(fig, out)
        st.success(f"Saved `{out.name}` under `{safe_display_path(artifact_dir)}`.")


def render_path_patching_tab(ctx: MechPageContext) -> None:
    st.markdown(
        """
        **Path patching** isolates causal edges between circuit components: each sender is run
        clean with only that node's corrupted activation, then each receiver is patched with the
        resulting activation. Scores use the calibrated logit-diff metric (0 = corrupted, 1 = clean).
        """
    )

    if ctx.model_path is None or ctx.task_cfg is None:
        st.stop()

    path_base = (
        repo_root() / PRUNING_EXPERIMENT_DIR / ctx.run_name / "mechanistic" / PATH_PATCHING_SUBFOLDER
    )

    if ctx.task_cfg.name != "counting":
        st.warning(
            "Path patching batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert ctx.corruption is not None
    artifact_dir = path_base / ctx.corruption.name
    counting_ok = ctx.task_cfg.name == "counting"
    current_key, pinned_key = _session_keys(ctx.run_name, ctx.corruption.name)

    try:
        with st.spinner("Loading model architecture…"):
            n_layers, n_heads = resolve_model_dims(
                str(ctx.model_path), ctx.device, ctx.compat
            )
    except Exception as e:
        st.exception(e)
        st.stop()

    catalog = build_circuit_node_catalog(n_layers, n_heads)

    st.caption(
        f"Node catalog from `{ctx.model_path}`: **{n_layers}** layers, **{n_heads}** heads per layer "
        f"({len(catalog)} hook sites)."
    )

    default_bs = int(ctx.raw_config.get("batch_size", 25)) if ctx.raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_path_bs_{ctx.run_name}",
    )

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        sender_nodes = render_circuit_node_checkboxes(
            catalog,
            role="sender",
            run_name=ctx.run_name,
            corruption=ctx.corruption.name,
        )
    with sel_col2:
        receiver_nodes = render_circuit_node_checkboxes(
            catalog,
            role="receiver",
            run_name=ctx.run_name,
            corruption=ctx.corruption.name,
        )

    title_input = st.text_input(
        "Plot title (optional)",
        value="",
        key=f"mech_path_title_{ctx.run_name}",
        placeholder="Auto-generated from selection if left blank",
    )

    run_path = st.button(
        "Run path patching",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_path_{ctx.run_name}",
    )

    if run_path and counting_ok:
        if not sender_nodes or not receiver_nodes:
            st.error("Select at least one sender and one receiver node.")
        else:
            try:
                with st.spinner("Loading model and batch…"):
                    model, _tokenizer, corrupted = load_counting_corrupt_batch(
                        ctx.task_cfg,
                        ctx.corruption,
                        batch_size,
                        ctx.device,
                        ctx.compat,
                        str(ctx.model_path),
                    )

                with st.spinner(
                    f"Path patching ({len(sender_nodes)}×{len(receiver_nodes)} edges)…"
                ):
                    results = path_patch(
                        model=model,
                        sender_nodes=sender_nodes,
                        receiver_nodes=receiver_nodes,
                        clean_corrupt_data=corrupted,
                        metric=logit_diff_metric,
                    )
                    title = title_input.strip() or _default_title(
                        list(sender_nodes), list(receiver_nodes)
                    )
                    fig = plot_path_patching_matrix(results, title=title)
                    st.session_state[current_key] = {
                        "fig_json": fig.to_json(),
                        "title": title,
                        "senders": list(sender_nodes.keys()),
                        "receivers": list(receiver_nodes.keys()),
                    }
            except Exception as e:
                st.exception(e)

    if pinned_key not in st.session_state:
        st.session_state[pinned_key] = []

    current = st.session_state.get(current_key)
    pinned: list[dict[str, Any]] = st.session_state[pinned_key]

    st.divider()
    st.markdown("#### Current result")
    if current is None:
        st.caption("No run yet. Select nodes and click **Run path patching**.")
    else:
        fig_current = _fig_from_entry(current)
        st.markdown(f"**{current['title']}**")
        st.caption(
            f"Senders: {', '.join(current['senders'])}  \n"
            f"Receivers: {', '.join(current['receivers'])}"
        )
        _display_path_patch_fig(fig_current, chart_key=f"{current_key}_chart")

        pin_col, _ = st.columns([1, 3])
        with pin_col:
            if st.button("Pin current result", key=f"mech_path_pin_{ctx.run_name}"):
                st.session_state[pinned_key].append(
                    {
                        "pin_id": uuid.uuid4().hex[:8],
                        **current,
                    }
                )
                st.rerun()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        _save_current_plot(
            fig_current,
            artifact_dir,
            current,
            key_prefix=f"mech_path_cur_{ctx.run_name}_{ctx.corruption.name}",
        )

    if pinned:
        st.divider()
        st.markdown("#### Pinned results")
        st.caption("Pinned runs stay visible when you run a new experiment.")
        for idx, entry in enumerate(list(pinned)):
            pin_id = entry["pin_id"]
            with st.container(border=True):
                st.markdown(f"**{entry['title']}** · pin `{pin_id}`")
                st.caption(
                    f"Senders: {', '.join(entry['senders'])}  \n"
                    f"Receivers: {', '.join(entry['receivers'])}"
                )
                fig_pinned = _fig_from_entry(entry)
                _display_path_patch_fig(
                    fig_pinned,
                    chart_key=f"{pinned_key}_{pin_id}_{idx}",
                )
                unp_col, _ = st.columns([1, 3])
                with unp_col:
                    if st.button(
                        "Unpin",
                        key=f"mech_path_unpin_{ctx.run_name}_{pin_id}",
                    ):
                        st.session_state[pinned_key] = [
                            p for p in st.session_state[pinned_key] if p["pin_id"] != pin_id
                        ]
                        st.rerun()

        if st.button(
            "Clear all pins",
            key=f"mech_path_clear_pins_{ctx.run_name}",
        ):
            st.session_state[pinned_key] = []
            st.rerun()

        if st.button(
            "Save pinned manifest",
            key=f"mech_path_save_manifest_{ctx.run_name}",
            help="Writes pinned run metadata (titles and node lists) to JSON.",
        ):
            artifact_dir.mkdir(parents=True, exist_ok=True)
            manifest = [
                {
                    "pin_id": p["pin_id"],
                    "title": p["title"],
                    "senders": p["senders"],
                    "receivers": p["receivers"],
                }
                for p in pinned
            ]
            (artifact_dir / PINNED_MANIFEST).write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            st.success(f"Saved `{PINNED_MANIFEST}`.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
    notes_key = f"mech_path_manual_{ctx.run_name}_{ctx.corruption.name}"
    init_flag = f"_mech_path_notes_init_{ctx.run_name}_{ctx.corruption.name}"
    if init_flag not in st.session_state:
        st.session_state[notes_key] = (
            notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
        )
        st.session_state[init_flag] = True

    st.divider()
    st.markdown("#### Interpretation and Notes")
    st.caption(
        f"Notes for corruption **{ctx.corruption.name}** (saved under path patching artifacts)."
    )
    st.text_area(
        "Interpretation and notes for path patching experiments",
        height=240,
        key=notes_key,
        label_visibility="visible",
    )
    if st.button(
        "Save interpretation",
        key=f"mech_save_path_manual_{ctx.run_name}_{ctx.corruption.name}",
    ):
        notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
        st.success(f"Saved to `{safe_display_path(notes_path)}`.")

    st.caption(
        "Use **Save current plot** to write the latest run to disk (filename from the plot title). "
        "Session pins are cleared on refresh; use **Save pinned manifest** to persist node lists. "
        "Artifacts live under `mechanistic/path_patching/<CORRUPTION_NAME>/`."
    )
