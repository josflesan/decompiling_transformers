from __future__ import annotations

import os
import warnings

import streamlit as st
import torch

from mechanistic.core.attribution import (
    attention_head_attribution,
    layerwise_attribution,
    residual_stream_attribution,
)
from tasks.registry import get_task

from mech_page.common import (
    MANUAL_ANALYSIS_FILENAME,
    attribution_html_saved,
    display_attribution_block,
    load_transformer_bridge,
    repo_root,
    safe_display_path,
)
from mech_page.context import MechPageContext
from utils.constants import PRUNING_EXPERIMENT_DIR

ATTRIBUTION_HTML = {
    "Residual Stream Attribution": "residual_stream_attribution.html",
    "Residual Stream Component-Level Attribution": "residual_stream_component_attribution.html",
    "Attention Head Attribution": "attention_head_attribution.html",
}


def render_attribution_tab(ctx: MechPageContext) -> None:
    st.markdown(
        """
        These experiments assess the influence of different layer components on the residual stream
        of the model after the **final layer**. This lets us gauge how much each component contributes to
        the final prediction. Note that this is similar to the **LogitLens** attribution method.
        """
    )

    if ctx.model_path is None or ctx.task_cfg is None:
        st.stop()

    attr_dir = repo_root() / PRUNING_EXPERIMENT_DIR / ctx.run_name / "mechanistic" / "attribution"

    if ctx.task_cfg.name != "counting":
        st.warning(
            "Attribution corruption batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert ctx.corruption is not None
    artifact_dir = attr_dir / ctx.corruption.name

    default_bs = int(ctx.raw_config.get("batch_size", 25)) if ctx.raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_attr_bs_{ctx.run_name}",
    )
    save_html = st.checkbox("Save Plot", value=True)

    run_attr = st.button(
        "Run attribution analysis",
        type="primary",
        disabled=ctx.task_cfg.name != "counting",
    )

    fig_residual = fig_layer = fig_heads = None
    if run_attr:
        if ctx.task_cfg.name != "counting":
            st.error("Switch `task_config.name` to `counting` in the matched YAML.")
        else:
            warnings.filterwarnings("ignore")
            if ctx.device == "mps":
                os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
            torch.set_grad_enabled(False)

            with st.spinner("Loading model…"):
                try:
                    model = load_transformer_bridge(str(ctx.model_path), ctx.device, ctx.compat)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            with st.spinner("Building corrupted counting batch…"):
                task = get_task("counting", ctx.task_cfg)
                _tokenizer, datasets = task.build()
                train_ds = datasets["train"]
                corrupted = train_ds.get_corrupted(
                    ctx.corruption,
                    batch_size=int(batch_size),
                )

            common_kw = dict(
                model=model,
                tokens=corrupted.clean_tokens,
                position_ids=corrupted.clean_pos,
                answer_tokens=corrupted.answer_tokens[:, 0],
                exp_name=ctx.run_name,
                device=ctx.device,
                save_html=save_html,
                artifact_subdir=ctx.corruption.name,
            )

            with st.spinner("Running all three attribution passes…"):
                fig_residual = residual_stream_attribution(**common_kw)
                fig_layer = layerwise_attribution(**common_kw)
                fig_heads = attention_head_attribution(**common_kw)

            if save_html:
                st.success(f"Wrote HTML under `{safe_display_path(artifact_dir)}`.")

    plot_specs: list[tuple[str, str, object]] = []
    for title, fname in ATTRIBUTION_HTML.items():
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
            display_attribution_block(title, fname, artifact_dir, fig, embed_height=300)

    all_plots_ready = all(attribution_html_saved(artifact_dir, fname) for _, fname, _ in plot_specs)

    if all_plots_ready:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
        notes_key = f"mech_manual_analysis_{ctx.run_name}_{ctx.corruption.name}"
        init_flag = f"_mech_manual_notes_init_{ctx.run_name}_{ctx.corruption.name}"
        if init_flag not in st.session_state:
            st.session_state[notes_key] = (
                notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
            )
            st.session_state[init_flag] = True

        st.markdown("#### Interpretation and Notes")
        st.caption(f"Notes for corruption **{ctx.corruption.name}** (saved next to that corruption's plots).")
        st.text_area(
            "Interpretation and notes for these attribution plots",
            height=240,
            key=notes_key,
            label_visibility="visible",
        )
        if st.button("Save interpretation", key=f"mech_save_manual_{ctx.run_name}_{ctx.corruption.name}"):
            notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
            st.success(f"Saved to `{safe_display_path(notes_path)}`.")

    st.caption(
        "The forward pass uses **clean** tokens and positions; **corruption** defines the counterfactual "
        "batch construction and the **answer-token** column used for logit directions, so attribution depends on the "
        "corruption setting. Plots and notes are stored under `mechanistic/attribution/<CORRUPTION_NAME>/`."
    )
