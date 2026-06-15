from __future__ import annotations

import streamlit as st

from mechanistic.core.activation_patching import (
    attention_head_patching,
    residual_stream_patching,
)
from mechanistic.utilities.metrics import logit_diff_metric

from mech_page.common import (
    MANUAL_ANALYSIS_FILENAME,
    attribution_html_saved,
    display_attribution_block,
    load_counting_corrupt_batch,
    repo_root,
    safe_display_path,
    save_plotly_figure,
)
from mech_page.context import MechPageContext
from utils.constants import PRUNING_EXPERIMENT_DIR

ACTIVATION_PATCHING_SUBFOLDER = "activation_patching"
RESIDUAL_STREAM_HTML = "residual_stream_patching.html"
ATTENTION_HEAD_HTML = "attention_head_patching.html"


def render_activation_patching_tab(ctx: MechPageContext) -> None:
    st.markdown(
        """
        **Activation patching** measures how much restoring clean activations at each site
        recovers the model's logit difference on corrupted inputs. Higher values indicate
        stronger causal dependence on that component.
        """
    )

    if ctx.model_path is None or ctx.task_cfg is None:
        st.stop()

    patch_base = (
        repo_root() / PRUNING_EXPERIMENT_DIR / ctx.run_name / "mechanistic" / ACTIVATION_PATCHING_SUBFOLDER
    )

    if ctx.task_cfg.name != "counting":
        st.warning(
            "Activation patching batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert ctx.corruption is not None
    artifact_dir = patch_base / ctx.corruption.name
    counting_ok = ctx.task_cfg.name == "counting"

    default_bs = int(ctx.raw_config.get("batch_size", 25)) if ctx.raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_patch_bs_{ctx.run_name}",
    )
    save_html = st.checkbox(
        "Save plots",
        value=True,
        key=f"mech_patch_save_{ctx.run_name}",
    )

    run_patch = st.button(
        "Run activation patching",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_patch_{ctx.run_name}",
    )

    fig_residual = fig_heads = None
    if run_patch and counting_ok:
        try:
            with st.spinner("Loading model and corrupted batch…"):
                model, _tokenizer, corrupted = load_counting_corrupt_batch(
                    ctx.task_cfg,
                    ctx.corruption,
                    batch_size,
                    ctx.device,
                    ctx.compat,
                    str(ctx.model_path),
                )

            with st.spinner("Residual stream patching…"):
                fig_residual = residual_stream_patching(model, corrupted, logit_diff_metric)
                if save_html and fig_residual is not None:
                    save_plotly_figure(fig_residual, artifact_dir / RESIDUAL_STREAM_HTML)

            with st.spinner("Attention head patching…"):
                fig_heads = attention_head_patching(model, corrupted, logit_diff_metric)
                if save_html and fig_heads is not None:
                    save_plotly_figure(fig_heads, artifact_dir / ATTENTION_HEAD_HTML)

            if save_html and (fig_residual is not None or fig_heads is not None):
                st.success(f"Wrote HTML under `{safe_display_path(artifact_dir)}`.")
        except Exception as e:
            st.exception(e)

    display_attribution_block(
        "Residual Stream Patching",
        RESIDUAL_STREAM_HTML,
        artifact_dir,
        fig_residual,
        embed_height=520,
    )

    display_attribution_block(
        "Attention Head Patching",
        ATTENTION_HEAD_HTML,
        artifact_dir,
        fig_heads,
        embed_height=520,
    )

    has_saved = attribution_html_saved(artifact_dir, RESIDUAL_STREAM_HTML) or attribution_html_saved(
        artifact_dir, ATTENTION_HEAD_HTML
    )
    has_live = fig_residual is not None or fig_heads is not None

    if has_saved or has_live:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
        notes_key = f"mech_patch_manual_{ctx.run_name}_{ctx.corruption.name}"
        init_flag = f"_mech_patch_notes_init_{ctx.run_name}_{ctx.corruption.name}"
        if init_flag not in st.session_state:
            st.session_state[notes_key] = (
                notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
            )
            st.session_state[init_flag] = True

        st.divider()
        st.markdown("#### Interpretation and Notes")
        st.caption(
            f"Notes for corruption **{ctx.corruption.name}** (saved next to activation patching plots)."
        )
        st.text_area(
            "Interpretation and notes for these activation patching analyses",
            height=240,
            key=notes_key,
            label_visibility="visible",
        )
        if st.button(
            "Save interpretation",
            key=f"mech_save_patch_manual_{ctx.run_name}_{ctx.corruption.name}",
        ):
            notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
            st.success(f"Saved to `{safe_display_path(notes_path)}`.")

    st.caption(
        "The corrupted batch is run forward while clean activations are patched in from cache; "
        "scores use the calibrated **logit diff** metric (0 = corrupted baseline, 1 = clean baseline). "
        "Plots and notes are stored under `mechanistic/activation_patching/<CORRUPTION_NAME>/`."
    )
