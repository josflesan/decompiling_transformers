import math

import altair as alt
import pandas as pd
import psutil
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from components.charts import altair_chart, altair_histogram, altair_trial_chart
from components.layout import metric_card
from utils.constants import PRUNING_EXPERIMENT_DIR, PRUNING_STAGES
from utils.dataloader import get_stages, load_metrics, split_train_val

st.set_page_config(layout="wide")

st_autorefresh(interval=2000)

st.title("Causal Pruning Dashboard")


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{100 * float(value):.2f}%"


def _format_lamb(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.2e}"


def _lambda_probe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "task" not in df.columns:
        return pd.DataFrame()
    probes = df[df["task"] == "lambda_probe"].copy()
    if probes.empty:
        return probes
    for col in ("lamb", "acc_match", "num_edges", "threshold_acc", "baseline_acc", "reference_acc"):
        if col in probes.columns:
            probes[col] = pd.to_numeric(probes[col], errors="coerce")
    if "feasible" in probes.columns:
        probes["feasible"] = probes["feasible"].astype(bool)
    return probes


def _lambda_selected_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "task" not in df.columns:
        return pd.DataFrame()
    selected = df[df["task"] == "lambda_search_selected"].copy()
    if selected.empty:
        return selected
    for col in ("lamb", "acc_match", "num_edges", "threshold_acc"):
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    if "used_fallback" in selected.columns:
        selected["used_fallback"] = selected["used_fallback"].astype(bool)
    return selected


def _trial_train_df(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty or "split" not in stage_df.columns:
        return pd.DataFrame()
    train = stage_df[stage_df["split"] == "train"].copy()
    if "lambda_trial" not in train.columns:
        return pd.DataFrame()
    return train[train["lambda_trial"].notna()].copy()


def _trial_sort_key(label: str) -> tuple:
    if label.startswith("final"):
        return (1, label)
    return (0, label)


def _timestamp_span_seconds(ts: pd.Series) -> float:
    ts = pd.to_numeric(ts, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return float(ts.max() - ts.min())


def _stage_elapsed_minutes(stage_df: pd.DataFrame) -> float | None:
    if stage_df.empty or "timestamp" not in stage_df.columns:
        return None

    timed = stage_df.copy()
    timed["_ts"] = pd.to_numeric(timed["timestamp"], errors="coerce")
    timed = timed[timed["_ts"] > 1_000_000_000]
    if timed.empty:
        return None

    return _timestamp_span_seconds(timed["_ts"]) / 60.0


def _stage_progress_state(
    stage: str,
    stage_df: pd.DataFrame,
    trial_train: pd.DataFrame,
    train_s: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:
    elapsed = _stage_elapsed_minutes(stage_df)
    selected = _lambda_selected_df(df)
    stage_selected = selected[selected["stage"] == stage] if not selected.empty else pd.DataFrame()

    if not trial_train.empty and "lambda_trial" in trial_train.columns:
        ordered = trial_train.sort_values(["timestamp", "step"], na_position="last")
        active_label = str(ordered.iloc[-1]["lambda_trial"])
        active = trial_train[trial_train["lambda_trial"] == active_label].copy()

        current_step = int(active["step"].max())
        max_step = int(active["current_maxstep"].iloc[-1])
        progress = min((current_step + 1) / max_step, 1.0) if max_step > 0 else 0.0

        final_train = trial_train[trial_train["lambda_trial"].str.startswith("final", na=False)]
        final_complete = False
        if not final_train.empty:
            final_step = int(final_train["step"].max())
            final_max = int(final_train["current_maxstep"].iloc[-1])
            final_complete = final_step + 1 >= final_max

        stage_complete = final_complete and not stage_selected.empty
        caption = f"Current run: {active_label}"
        return {
            "progress": progress,
            "complete": stage_complete,
            "elapsed": elapsed,
            "caption": caption,
        }

    if train_s.empty or "current_maxstep" not in train_s.columns:
        return {"progress": 0.0, "complete": False, "elapsed": elapsed, "caption": None}

    current_step = int(train_s["step"].max())
    max_step = int(train_s["current_maxstep"].iloc[-1])
    progress = min((current_step + 1) / max_step, 1.0) if max_step > 0 else 0.0
    stage_complete = current_step + 1 >= max_step
    return {
        "progress": progress,
        "complete": stage_complete,
        "elapsed": elapsed,
        "caption": None,
    }


def _render_stage_progress(
    stage: str,
    stage_df: pd.DataFrame,
    trial_train: pd.DataFrame,
    train_s: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    if "step" not in stage_df.columns:
        return

    state = _stage_progress_state(stage, stage_df, trial_train, train_s, df)
    if state["caption"]:
        st.caption(state["caption"])

    st.progress(state["progress"])

    if state["complete"] and state["elapsed"] is not None:
        st.success(f"{stage} complete | Total stage time: {state['elapsed']:.1f} mins.")
    elif state["elapsed"] is not None:
        st.caption(f"Total stage time so far: {state['elapsed']:.1f} mins.")


def _render_lambda_trial_summary(df: pd.DataFrame, stage: str) -> None:
    probes = _lambda_probe_df(df)
    selected = _lambda_selected_df(df)
    stage_probes = probes[probes["stage"] == stage] if not probes.empty else pd.DataFrame()
    stage_selected = selected[selected["stage"] == stage] if not selected.empty else pd.DataFrame()

    if stage_probes.empty and stage_selected.empty:
        return

    st.subheader("Lambda Search")

    if not stage_selected.empty:
        row = stage_selected.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Lambda", _format_lamb(row.get("lamb")))
        c2.metric("Match Acc at Selection", _pct(row.get("acc_match")))
        c3.metric("Final Edge Count", int(row["num_edges"]) if not pd.isna(row.get("num_edges")) else "N/A")
        c4.metric("Used Fallback", "Yes" if row.get("used_fallback") else "No")

    if not stage_probes.empty:
        latest = stage_probes.iloc[-1]
        t1, t2, t3 = st.columns(3)
        t1.metric("Baseline Acc Match", _pct(latest.get("baseline_acc")))
        t2.metric("Reference Acc Match", _pct(latest.get("reference_acc")))
        t3.metric("Threshold Acc Match", _pct(latest.get("threshold_acc")))

        table = stage_probes.copy()
        table["lambda"] = table["lamb"].map(_format_lamb)
        table["match accuracy"] = table["acc_match"].map(_pct)
        table["status"] = table["feasible"].map(lambda ok: "feasible" if ok else "below threshold")
        display_cols = ["lambda", "match accuracy", "num_edges", "status", "probe"]
        st.markdown("#### Trial Match Accuracies")
        st.dataframe(
            table[[c for c in display_cols if c in table.columns]],
            use_container_width=True,
            hide_index=True,
        )

        plot_df = stage_probes.copy()
        plot_df["lambda"] = plot_df["lamb"].map(_format_lamb)
        plot_df["feasible_label"] = plot_df["feasible"].map(
            {True: "feasible", False: "below threshold"}
        )
        threshold_acc = float(plot_df["threshold_acc"].dropna().iloc[0])

        acc_chart = (
            alt.Chart(plot_df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("lamb:Q", scale=alt.Scale(type="log"), title="Lambda"),
                y=alt.Y("acc_match:Q", title="Match Accuracy"),
                color=alt.Color("feasible_label:N", title="Status"),
                tooltip=["lambda", "acc_match", "num_edges", "feasible_label"],
            )
            .properties(height=240, title="Probe Match Accuracy vs Lambda")
        )
        threshold_rule = (
            alt.Chart(pd.DataFrame({"threshold_acc": [threshold_acc]}))
            .mark_rule(color="#888888", strokeDash=[6, 4])
            .encode(y="threshold_acc:Q")
        )
        st.altair_chart(acc_chart + threshold_rule, use_container_width=True)


def _render_trial_training_charts(trial_train: pd.DataFrame, stage: str) -> None:
    if trial_train.empty:
        return

    trial_train = trial_train.copy()
    trial_train["lambda_trial"] = pd.Categorical(
        trial_train["lambda_trial"],
        categories=sorted(trial_train["lambda_trial"].unique(), key=_trial_sort_key),
        ordered=True,
    )

    st.subheader("Training by Lambda Trial")
    cols = st.columns(2)
    with cols[0]:
        altair_trial_chart(
            trial_train,
            x="step",
            y="loss",
            key=f"{stage}_trial_loss",
            title="Loss",
        )
    with cols[1]:
        altair_trial_chart(
            trial_train,
            x="step",
            y="kl_div",
            key=f"{stage}_trial_kl",
            title="KL Divergence",
        )

    cols2 = st.columns(2)
    with cols2[0]:
        altair_trial_chart(
            trial_train,
            x="step",
            y="task_loss",
            key=f"{stage}_trial_task",
            title="Task Loss",
        )
    with cols2[1]:
        edge_col = "num_edges" if "num_edges" in trial_train.columns else "reg_edge"
        altair_trial_chart(
            trial_train,
            x="step",
            y=edge_col,
            key=f"{stage}_trial_edges",
            title="Num Edges",
        )


# ----------------------------
# Sidebar: Run selection
# ----------------------------

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No runs found.")
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name / "pruning"
metrics_file = run_dir / "metrics.jsonl"

# ----------------------------
# Load data
# ----------------------------

df = load_metrics(metrics_file)
if df.empty:
    st.warning("No metrics yet.")
    st.stop()

train_df, val_df = split_train_val(df)

# ----------------------------
# Sidebar: System Stats
# ----------------------------

st.sidebar.markdown("### System")
col1, col2 = st.sidebar.columns(2)
col1.metric("CPU %", psutil.cpu_percent())
col2.metric("RAM %", psutil.virtual_memory().percent)

st.sidebar.markdown("### Run Info")
st.sidebar.write(f"Steps Logged: {len(df)}")
lambda_probes = _lambda_probe_df(df)
lambda_selected = _lambda_selected_df(df)
if not lambda_probes.empty:
    st.sidebar.write(f"Lambda Probes: {len(lambda_probes)}")
if not lambda_selected.empty:
    st.sidebar.write(f"Stages with Selection: {lambda_selected['stage'].nunique()}")


# ----------------------------
# Overview
# ----------------------------

def render_overview():
    st.subheader("Overview")

    combined = train_df.assign(type="train")
    if not val_df.empty:
        combined = pd.concat([
            train_df.assign(type="train"),
            val_df.assign(type="val"),
        ])

    st.markdown("### Loss")
    altair_chart(combined, x="step", y="loss", key="overview_loss")

    st.markdown("### Validation Summary")
    if not val_df.empty:
        summary = (
            val_df[["stage", "step", "kl_div", "task_loss", "acc_task", "acc_match"]]
            .groupby("stage")
            .mean(numeric_only=True)
        )
        st.dataframe(summary)


# ----------------------------
# Stage Rendering
# ----------------------------

def render_stage(stage: str, pretrain: bool = False):
    stage_df = df[df["stage"] == stage]
    train_s, val_s = split_train_val(stage_df)
    trial_train = _trial_train_df(stage_df)

    st.header(stage)

    _render_stage_progress(stage, stage_df, trial_train, train_s, df)

    if not pretrain:
        _render_lambda_trial_summary(df, stage)
        _render_trial_training_charts(trial_train, stage)
    
    if trial_train.empty:
        combined = train_s.assign(type="train")
        if not val_s.empty:
            combined = pd.concat([
                train_s.assign(type="train"),
                val_s.assign(type="val"),
            ])

        cols = st.columns(3)
        with cols[0]:
            metric_card("Loss", lambda: altair_chart(df=combined, x="step", y="loss", key=f"{stage}_loss"))
        with cols[1]:
            metric_card("KL Divergence", lambda: altair_chart(df=combined, x="step", y="kl_div", key=f"{stage}_kl"))
        with cols[2]:
            metric_card("Task Loss", lambda: altair_chart(df=combined, x="step", y="task_loss", key=f"{stage}_task"))

        if not pretrain and "reg_edge" in train_s.columns and not train_s.empty:
            st.subheader("Edge Count")
            altair_chart(df=train_s, x="step", y="reg_edge", color=None, key=f"{stage}_edges")
    elif not val_s.empty:
        st.subheader("Final Run Validation")
        val_final = val_s[val_s["lambda_trial"].str.startswith("final", na=False)] if "lambda_trial" in val_s.columns else val_s
        if val_final.empty:
            val_final = val_s
        val_chart_df = val_final.assign(type="val")
        vcols = st.columns(2)
        with vcols[0]:
            altair_chart(val_chart_df, x="step", y="acc_match", color=None, key=f"{stage}_val_acc")
        with vcols[1]:
            altair_chart(val_chart_df, x="step", y="kl_div", color=None, key=f"{stage}_val_kl")

    if train_s.empty and trial_train.empty:
        st.caption("No training metrics for this stage yet.")
        return

    if not pretrain and "sampler_params" in train_s.columns and not train_s["sampler_params"].dropna().empty:
        altair_histogram(train_s)

    with st.expander("Raw Data"):
        st.dataframe(stage_df.tail(200))


def render_lambda_search(df_all: pd.DataFrame) -> None:
    st.subheader("Automated Lambda Search")
    probes = _lambda_probe_df(df_all)
    selected = _lambda_selected_df(df_all)
    if probes.empty and selected.empty:
        st.info("No lambda search metrics yet.")
        return

    stages_with_data = sorted(
        set(probes.get("stage", pd.Series(dtype=str)).dropna().tolist())
        | set(selected.get("stage", pd.Series(dtype=str)).dropna().tolist()),
        key=lambda s: PRUNING_STAGES.index(s) if s in PRUNING_STAGES else s,
    )
    for stage in stages_with_data:
        render_stage(stage, pretrain=False)
        st.divider()


# ----------------------------
# Tabs
# ----------------------------

stages = get_stages(df)
if "Lambda Search" not in stages:
    stages = ["Overview", "Lambda Search"] + [s for s in stages if s not in ("Overview", "Lambda Search")]
selected_stage = st.selectbox("Stage", stages)

if selected_stage == "Overview":
    render_overview()
elif selected_stage == "Lambda Search":
    render_lambda_search(df)
else:
    render_stage(selected_stage, "Pretrain" in selected_stage)
