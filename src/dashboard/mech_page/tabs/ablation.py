from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mechanistic.core.ablations import AblationMode, ablate_circuit, ablation_sweep
from mechanistic.utilities.metrics import ablation_metric

from mech_page.common import (
    MANUAL_ANALYSIS_FILENAME,
    build_circuit_node_catalog,
    load_counting_corrupt_batch,
    render_circuit_node_checkboxes,
    repo_root,
    resolve_model_dims,
    safe_display_path,
)
from mech_page.context import MechPageContext
from utils.constants import PRUNING_EXPERIMENT_DIR

ABLATION_SUBFOLDER = "ablation"
SWEEP_CSV = "ablation_sweep.csv"
SWEEP_META = "ablation_sweep.meta.json"
CIRCUIT_RESULT_JSON = "circuit_ablation.json"


def _sweep_paths(artifact_dir: Path) -> tuple[Path, Path]:
    return artifact_dir / SWEEP_CSV, artifact_dir / SWEEP_META


def _save_sweep_results(
    artifact_dir: Path,
    results: dict[str, float],
    *,
    mode: AblationMode,
    batch_size: int,
    node_keys: list[str],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _sweep_paths(artifact_dir)
    df = pd.DataFrame(
        {"node": list(results.keys()), "ablation_score": list(results.values())}
    ).sort_values("ablation_score", ascending=False)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "batch_size": batch_size,
                "n_nodes": len(node_keys),
                "nodes": node_keys,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_sweep_table(artifact_dir: Path) -> pd.DataFrame | None:
    csv_path, _ = _sweep_paths(artifact_dir)
    if not csv_path.is_file():
        return None
    return pd.read_csv(csv_path)


def _load_sweep_meta(artifact_dir: Path) -> dict[str, Any] | None:
    _, meta_path = _sweep_paths(artifact_dir)
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_circuit_result(
    artifact_dir: Path,
    *,
    nodes: list[str],
    score: float,
    mode: AblationMode,
    batch_size: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": nodes,
        "ablation_score": score,
        "mode": mode,
        "batch_size": batch_size,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_dir / CIRCUIT_RESULT_JSON).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _load_circuit_result(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / CIRCUIT_RESULT_JSON
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _display_sweep_table(df: pd.DataFrame, meta: dict[str, Any] | None) -> None:
    if meta:
        st.caption(
            f"Saved sweep · mode **{meta.get('mode', '?')}** · "
            f"{meta.get('n_nodes', len(df))} nodes · batch **{meta.get('batch_size', '?')}**"
        )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "node": st.column_config.TextColumn("Node"),
            "ablation_score": st.column_config.NumberColumn(
                "Ablation score",
                help="1 ≈ unaffected; lower (or negative) ≈ more critical for the task.",
                format="%.4f",
            ),
        },
    )


def render_ablation_tab(ctx: MechPageContext) -> None:
    st.markdown(
        """
        **Ablation analysis** measures how much each component (or a chosen circuit) is needed
        on clean inputs. Scores use the **ablation metric** (1 ≈ clean performance retained;
        lower values mean stronger dependence on the ablated part).
        """
    )

    if ctx.model_path is None or ctx.task_cfg is None:
        st.stop()

    ablation_base = (
        repo_root() / PRUNING_EXPERIMENT_DIR / ctx.run_name / "mechanistic" / ABLATION_SUBFOLDER
    )

    if ctx.task_cfg.name != "counting":
        st.warning(
            "Ablation batches are only wired for the **counting** task. "
            "Update `task_config` in the matched YAML if needed."
        )

    assert ctx.corruption is not None
    artifact_dir = ablation_base / ctx.corruption.name
    counting_ok = ctx.task_cfg.name == "counting"

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
        f"Hook catalog from `{ctx.model_path}`: **{n_layers}** layers, **{n_heads}** heads "
        f"({len(catalog)} sites)."
    )

    default_bs = int(ctx.raw_config.get("batch_size", 25)) if ctx.raw_config else 25
    default_bs = max(1, min(default_bs, 256))
    batch_size = st.number_input(
        "Number of Prompts",
        min_value=1,
        max_value=256,
        value=default_bs,
        step=1,
        key=f"mech_abl_bs_{ctx.run_name}",
    )

    mode: AblationMode = st.radio(
        "Ablation mode",
        options=["mean", "zero"],
        horizontal=True,
        key=f"mech_abl_mode_{ctx.run_name}",
        help="Mean ablation replaces activations with the batch mean; zero ablation zeros them out.",
    )

    st.divider()
    st.markdown("#### Per-node ablation sweep")
    st.caption(
        "Ablates each selected node individually and records scores in a table (saved automatically)."
    )

    sweep_all = st.checkbox(
        "Sweep all hook sites",
        value=True,
        key=f"mech_abl_sweep_all_{ctx.run_name}",
    )
    sweep_nodes = dict(catalog) if sweep_all else render_circuit_node_checkboxes(
        catalog,
        role="sweep",
        run_name=ctx.run_name,
        corruption=ctx.corruption.name,
    )

    run_sweep = st.button(
        "Run ablation sweep",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_abl_sweep_{ctx.run_name}",
    )

    sweep_df: pd.DataFrame | None = None
    sweep_meta: dict[str, Any] | None = None

    if run_sweep and counting_ok:
        if not sweep_nodes:
            st.error("Select at least one node for the sweep, or enable **Sweep all hook sites**.")
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
                with st.spinner(f"Ablating {len(sweep_nodes)} nodes…"):
                    results = ablation_sweep(
                        model=model,
                        nodes=sweep_nodes,
                        clean_corrupt_data=corrupted,
                        metric=ablation_metric,
                        mode=mode,
                    )
                _save_sweep_results(
                    artifact_dir,
                    results,
                    mode=mode,
                    batch_size=int(batch_size),
                    node_keys=list(sweep_nodes.keys()),
                )
                sweep_df = _load_sweep_table(artifact_dir)
                sweep_meta = _load_sweep_meta(artifact_dir)
                st.success(f"Sweep saved to `{safe_display_path(artifact_dir / SWEEP_CSV)}`.")
            except Exception as e:
                st.exception(e)

    if sweep_df is None:
        sweep_df = _load_sweep_table(artifact_dir)
        sweep_meta = _load_sweep_meta(artifact_dir)

    if sweep_df is not None and not sweep_df.empty:
        _display_sweep_table(sweep_df, sweep_meta)
    else:
        st.caption("No sweep results yet. Run an ablation sweep to populate the table.")

    st.divider()
    st.markdown("#### Ablate a circuit")
    st.caption(
        "Select multiple nodes to ablate **together** and measure their combined effect."
    )

    circuit_nodes = render_circuit_node_checkboxes(
        catalog,
        role="circuit",
        run_name=ctx.run_name,
        corruption=ctx.corruption.name,
    )

    run_circuit = st.button(
        "Run circuit ablation",
        type="primary",
        disabled=not counting_ok,
        key=f"mech_run_abl_circuit_{ctx.run_name}",
    )

    circuit_result: dict[str, Any] | None = None

    if run_circuit and counting_ok:
        if not circuit_nodes:
            st.error("Select at least one node for the circuit.")
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
                with st.spinner(f"Ablating circuit ({len(circuit_nodes)} nodes)…"):
                    score = ablate_circuit(
                        model=model,
                        circuit=circuit_nodes,
                        clean_corrupt_data=corrupted,
                        metric=ablation_metric,
                        mode=mode,
                    )
                score_val = score.item() if hasattr(score, "item") else float(score)
                circuit_result = {
                    "nodes": list(circuit_nodes.keys()),
                    "ablation_score": score_val,
                    "mode": mode,
                    "batch_size": int(batch_size),
                }
                _save_circuit_result(
                    artifact_dir,
                    nodes=circuit_result["nodes"],
                    score=score_val,
                    mode=mode,
                    batch_size=int(batch_size),
                )
            except Exception as e:
                st.exception(e)

    if circuit_result is None:
        circuit_result = _load_circuit_result(artifact_dir)

    if circuit_result is not None:
        st.metric(
            "Combined circuit ablation score",
            f"{circuit_result['ablation_score']:.4f}",
            help="1 ≈ clean performance retained when all selected nodes are ablated together.",
        )
        st.markdown("**Circuit nodes**")
        st.code("\n".join(circuit_result["nodes"]), language=None)
        if circuit_result.get("saved_at"):
            st.caption(
                f"Last saved · mode **{circuit_result.get('mode', '?')}** · "
                f"batch **{circuit_result.get('batch_size', '?')}**"
            )
    else:
        st.caption("No circuit ablation yet. Select nodes and run **Run circuit ablation**.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    notes_path = artifact_dir / MANUAL_ANALYSIS_FILENAME
    notes_key = f"mech_abl_manual_{ctx.run_name}_{ctx.corruption.name}"
    init_flag = f"_mech_abl_notes_init_{ctx.run_name}_{ctx.corruption.name}"
    if init_flag not in st.session_state:
        st.session_state[notes_key] = (
            notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
        )
        st.session_state[init_flag] = True

    st.divider()
    st.markdown("#### Interpretation and Notes")
    st.caption(
        f"Notes for corruption **{ctx.corruption.name}** (saved under ablation artifacts)."
    )
    st.text_area(
        "Interpretation and notes for ablation experiments",
        height=240,
        key=notes_key,
        label_visibility="visible",
    )
    if st.button(
        "Save interpretation",
        key=f"mech_save_abl_manual_{ctx.run_name}_{ctx.corruption.name}",
    ):
        notes_path.write_text(st.session_state[notes_key], encoding="utf-8")
        st.success(f"Saved to `{safe_display_path(notes_path)}`.")

    st.caption(
        "Sweep tables are written to `ablation_sweep.csv` on each run. Circuit results go to "
        "`circuit_ablation.json`. Artifacts live under `mechanistic/ablation/<CORRUPTION_NAME>/`."
    )
