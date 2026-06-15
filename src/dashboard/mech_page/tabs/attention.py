from __future__ import annotations

import streamlit as st

from mechanistic.core.attention_inspection import (
    classify_all_heads,
    investigate_topk_attention_heads,
    plot_all_attention_patterns,
    plot_head_classification_heatmap,
)
from mechanistic.utilities.metrics import logit_diff_metric

from mech_page.common import (
    MANUAL_ANALYSIS_FILENAME,
    attention_body_key,
    attention_state_key,
    display_inspection_block,
    layer_head_html,
    load_counting_corrupt_batch,
    per_head_plot_paths,
    per_head_plots_complete,
    repo_root,
    resolve_model_dims,
    run_per_head_inspections,
    safe_display_path,
    save_html_content,
    save_plotly_figure,
)
from mech_page.context import MechPageContext
from utils.constants import PRUNING_EXPERIMENT_DIR

ATTENTION_INSPECTION_SUBFOLDER = "attention_inspection"
TOPK_ATTENTION_HTML_LEGACY = "topk_attention_heads.html"
ALL_ATTENTION_PATTERNS_HTML = "all_attention_patterns.html"
HEAD_CLASSIFICATION_HTML = "head_classification_heatmap.html"
CLASSIFY_FIG_HEIGHT = 220


def _topk_attention_html_filename(k: int, view: int) -> str:
    return f"topk_attention_heads_k{k}_view{view}.html"


def _circuitsvis_embed_height(k: int) -> int:
    return min(420, 96 + 68 * max(1, k))


def render_attention_tab(ctx: MechPageContext) -> None:
    st.markdown(
        """
        These plots enable inspection of the **attention patterns** across the heads of the model. The set of
        visualizations supported include attention patterns ranked by activation-patching scores, full attention
        pattern grids, **OV/QK circuit visualizations** for individual heads and automatic **head-type classification**
        using heuristic methods.
        """
    )

    if ctx.model_path is None or ctx.task_cfg is None:
        st.stop()

    inspect_base = (
        repo_root() / PRUNING_EXPERIMENT_DIR / ctx.run_name / "mechanistic" / ATTENTION_INSPECTION_SUBFOLDER
    )

    if ctx.task_cfg.name != "counting":
        st.warning(
            "Attention inspection batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert ctx.corruption is not None
    artifact_dir = inspect_base / ctx.corruption.name

    default_bs = int(ctx.raw_config.get("batch_size", 25)) if ctx.raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_attn_bs_{ctx.run_name}",
    )
    save_html_attn = st.checkbox(
        "Save plots",
        value=True,
        key=f"mech_attn_save_{ctx.run_name}",
    )

    counting_ok = ctx.task_cfg.name == "counting"
    n_layers, n_heads = resolve_model_dims(str(ctx.model_path), ctx.device, ctx.compat)
    layer_max = max(0, n_layers - 1)
    head_max = max(0, n_heads - 1)

    st.markdown("#### All attention patterns")
    run_all_patterns = st.button(
        "Plot all attention patterns",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_all_attn_{ctx.run_name}",
    )
    all_patterns_show_key = attention_state_key(ctx.run_name, ctx.corruption.name, "all_patterns")

    fig_all_patterns = None
    if run_all_patterns and counting_ok:
        with st.spinner("Computing mean attention for every head…"):
            try:
                model, tokenizer, corrupted = load_counting_corrupt_batch(
                    ctx.task_cfg,
                    ctx.corruption,
                    batch_size,
                    ctx.device,
                    ctx.compat,
                    str(ctx.model_path),
                )
                fig_all_patterns = plot_all_attention_patterns(model, tokenizer, corrupted)
                if fig_all_patterns is not None:
                    st.session_state[attention_body_key(all_patterns_show_key)] = (
                        fig_all_patterns.to_json()
                    )
                if save_html_attn and fig_all_patterns is not None:
                    save_plotly_figure(
                        fig_all_patterns,
                        artifact_dir / ALL_ATTENTION_PATTERNS_HTML,
                    )
                    st.success(f"Saved `{ALL_ATTENTION_PATTERNS_HTML}`.")
            except Exception as e:
                st.exception(e)

    display_inspection_block(
        "All Attention Patterns",
        ALL_ATTENTION_PATTERNS_HTML,
        artifact_dir,
        fig_all_patterns,
        session_show_key=all_patterns_show_key,
    )
    
    st.divider()
    st.markdown("#### Per-head circuit and positional plots")
    st.caption(
        "Select layer and head. Saved plots load automatically; one button runs any missing analyses."
    )
    lh1, lh2, lh3 = st.columns([1, 1, 2])
    with lh1:
        sel_layer = st.number_input(
            "Layer",
            min_value=0,
            max_value=layer_max,
            value=0,
            key=f"mech_attn_layer_{ctx.run_name}",
        )
    with lh2:
        sel_head = st.number_input(
            "Head",
            min_value=0,
            max_value=head_max,
            value=0,
            key=f"mech_attn_head_{ctx.run_name}",
        )
    with lh3:
        force_per_head = st.checkbox(
            "Recompute even if saved",
            value=False,
            key=f"mech_per_head_force_{ctx.run_name}",
        )

    layer_i, head_i = int(sel_layer), int(sel_head)
    per_head_paths = per_head_plot_paths(artifact_dir, layer_i, head_i)
    ov_html = per_head_paths["ov_circuit"].name
    ov_eig_html = per_head_paths["ov_eigenspectrum"].name
    qk_html = per_head_paths["qk_circuit"].name
    pos_html = per_head_paths["positional_attention"].name
    run_per_head = st.button(
        "Run per-head circuit analyses",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_per_head_{ctx.run_name}",
    )

    if run_per_head and counting_ok:
        if per_head_plots_complete(artifact_dir, layer_i, head_i) and not force_per_head:
            st.info(
                f"All four plots for L{layer_i}H{head_i} are already saved — displaying cached results."
            )
        else:
            with st.spinner(f"Running per-head analyses for L{layer_i}H{head_i}…"):
                try:
                    model, _tokenizer, corrupted = load_counting_corrupt_batch(
                        ctx.task_cfg,
                        ctx.corruption,
                        batch_size,
                        ctx.device,
                        ctx.compat,
                        str(ctx.model_path),
                    )
                    saved = run_per_head_inspections(
                        model,
                        corrupted,
                        layer_i,
                        head_i,
                        artifact_dir,
                        save_html=save_html_attn,
                        force_recompute=force_per_head,
                    )
                    if save_html_attn and saved:
                        st.success(f"Saved: {', '.join(saved)}.")
                    elif save_html_attn:
                        st.info("All plots were already on disk; nothing new to save.")
                except Exception as e:
                    st.exception(e)

    c_ov, c_eig = st.columns(2)
    with c_ov:
        display_inspection_block(
            f"OV Circuit (L{layer_i}H{head_i})",
            ov_html,
            artifact_dir,
            None,
            html_embed_height=520,
        )
    with c_eig:
        display_inspection_block(
            f"OV Eigenspectrum (L{layer_i}H{head_i})",
            ov_eig_html,
            artifact_dir,
            None,
            html_embed_height=380,
        )

    c_qk, c_pos = st.columns(2)
    with c_qk:
        display_inspection_block(
            f"QK Circuit (L{layer_i}H{head_i})",
            qk_html,
            artifact_dir,
            None,
            html_embed_height=520,
        )
    with c_pos:
        display_inspection_block(
            f"Positional Attention (L{layer_i}H{head_i})",
            pos_html,
            artifact_dir,
            None,
            html_embed_height=480,
        )

    st.divider()
    st.markdown("#### Automatic head-type detection")
    st.caption(
        "Classifies every head (previous-token, BOS, diagonal, induction, copying) and plots score heatmaps."
    )
    run_classify = st.button(
        "Run head classification",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_classify_{ctx.run_name}",
    )
    classify_show_key = attention_state_key(ctx.run_name, ctx.corruption.name, "classify")

    fig_classify = None
    if run_classify and counting_ok:
        with st.spinner("Classifying all heads (see terminal for summary table)…"):
            try:
                model, _tokenizer, corrupted = load_counting_corrupt_batch(
                    ctx.task_cfg,
                    ctx.corruption,
                    batch_size,
                    ctx.device,
                    ctx.compat,
                    str(ctx.model_path),
                )
                classification = classify_all_heads(model, corrupted)
                fig_classify = plot_head_classification_heatmap(model, classification)
                if fig_classify is not None:
                    fig_classify.update_layout(height=CLASSIFY_FIG_HEIGHT, autosize=True, width=None)
                    st.session_state[attention_body_key(classify_show_key)] = fig_classify.to_json()
                if save_html_attn and fig_classify is not None:
                    save_plotly_figure(fig_classify, artifact_dir / HEAD_CLASSIFICATION_HTML)
                    st.success(f"Saved `{HEAD_CLASSIFICATION_HTML}`.")
            except Exception as e:
                st.exception(e)

    display_inspection_block(
        "Head Type Scores",
        HEAD_CLASSIFICATION_HTML,
        artifact_dir,
        fig_classify,
        html_embed_height=CLASSIFY_FIG_HEIGHT + 40,
        session_show_key=classify_show_key,
    )
    
    st.divider()
    st.markdown("#### Top-K attention heads (activation patching)")
    st.caption(
        "Ranks heads by activation-patching score on the pattern view, then plots mean attention "
        "patterns for the top-k heads (CircuitsVis)."
    )
    tk1, tk2, tk3 = st.columns(3)
    with tk1:
        topk_k = st.number_input("k", min_value=1, max_value=16, value=4, key=f"mech_topk_k_{ctx.run_name}")
    with tk2:
        topk_view = st.number_input(
            "Patching view index",
            min_value=0,
            max_value=4,
            value=3,
            help="Facet index from activation patching (3 = attention pattern).",
            key=f"mech_topk_view_{ctx.run_name}",
        )
    with tk3:
        run_topk = st.button(
            "Run top-k inspection",
            type="primary",
            disabled=not counting_ok,
            key=f"mech_run_topk_{ctx.run_name}",
        )

    topk_html_name = _topk_attention_html_filename(int(topk_k), int(topk_view))
    topk_show_key = attention_state_key(ctx.run_name, ctx.corruption.name, f"topk_k{topk_k}_v{topk_view}")

    fig_topk = None
    if run_topk and counting_ok:
        with st.spinner("Running top-k attention head inspection…"):
            try:
                model, tokenizer, corrupted = load_counting_corrupt_batch(
                    ctx.task_cfg,
                    ctx.corruption,
                    batch_size,
                    ctx.device,
                    ctx.compat,
                    str(ctx.model_path),
                )
                fig_topk = investigate_topk_attention_heads(
                    model=model,
                    tokenizer=tokenizer,
                    clean_corrupt_data=corrupted,
                    metric=logit_diff_metric,
                    k=int(topk_k),
                    view=int(topk_view),
                )
                if fig_topk is not None:
                    html_body = fig_topk._repr_html_()
                    st.session_state[attention_body_key(topk_show_key)] = html_body
                    if save_html_attn:
                        save_html_content(html_body, artifact_dir / topk_html_name)
                        st.success(f"Saved `{topk_html_name}`.")
            except Exception as e:
                st.exception(e)

    display_inspection_block(
        f"Top-K Attention Heads (k={topk_k}, view={topk_view})",
        topk_html_name,
        artifact_dir,
        fig_topk,
        use_plotly=False,
        html_embed_height=_circuitsvis_embed_height(int(topk_k)),
        legacy_html_filenames=(TOPK_ATTENTION_HTML_LEGACY,),
        session_show_key=topk_show_key,
    )

    saved_attn_plots = list(artifact_dir.glob("*.html")) if artifact_dir.is_dir() else []
    if saved_attn_plots:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
        notes_key = f"mech_attn_manual_{ctx.run_name}_{ctx.corruption.name}"
        init_flag = f"_mech_attn_notes_init_{ctx.run_name}_{ctx.corruption.name}"
        if init_flag not in st.session_state:
            st.session_state[notes_key] = (
                notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
            )
            st.session_state[init_flag] = True

        st.divider()
        st.markdown("#### Interpretation and Notes")
        st.caption(f"Notes for corruption **{ctx.corruption.name}** (saved next to attention inspection plots).")
        st.text_area(
            "Interpretation and notes for these attention analyses",
            height=240,
            key=notes_key,
            label_visibility="visible",
        )
        if st.button("Save interpretation", key=f"mech_save_attn_manual_{ctx.run_name}_{ctx.corruption.name}"):
            notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
            st.success(f"Saved to `{safe_display_path(notes_path)}`.")
    
    
    st.caption(
        "Clean/corrupt batches match the attribution tab. Per-head filenames include layer and head "
        f"(e.g. `{layer_head_html('ov_circuit', 0, 0)}`)."
    )
