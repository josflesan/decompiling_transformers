import streamlit as st
import pandas as pd
import psutil
from streamlit_autorefresh import st_autorefresh

from components.charts import altair_chart, altair_histogram
from components.layout import metric_card
from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics, split_train_val, get_stages

st.set_page_config(layout="wide")

st_autorefresh(interval=2000)

st.title("Causal Pruning Dashboard")

# ----------------------------
# Sidebar: Run selection
# ----------------------------

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No runs found.")
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name
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

# ----------------------------
# Overview
# ----------------------------

def render_overview():
    st.subheader("Overview")
    
    combined = train_df.assign(type="train")
    if not val_df.empty:
        combined = pd.concat([
            train_df.assign(type="train"),
            val_df.assign(type="val")
        ])
    
    
    st.markdown("### Loss")
    altair_chart(combined, x='step', y='loss', key="overview_loss")
    
    st.markdown("### Validation Summary")
    if not val_df.empty:
        summary = val_df[['stage', 'step', 'kl_div', 'task_loss', 'acc_task', 'acc_match']].groupby("stage").mean(numeric_only=True)
        st.dataframe(summary)


# ----------------------------
# Stage Rendering
# ----------------------------

def render_stage(stage: str, pretrain: bool=False):
    stage_df = df[df["stage"] == stage]
    train_s, val_s = split_train_val(stage_df)
    
    combined = train_s.assign(type="train")
    if not val_s.empty:
        combined = pd.concat([
            train_s.assign(type="train"),
            val_s.assign(type="val")
        ])
    
    st.header(stage)
    
    # Progress bar
    if "step" in stage_df.columns:
        current_step = stage_df["step"].max()
        #TODO: delete this
        max_step = 5000 if 'current_maxstep' not in stage_df.columns else stage_df['current_maxstep']
        st.progress(min(current_step / max_step, 1.0))
    
    if current_step == max_step:
        st.success(f"{stage} Complete! | Time Elapsed: {(stage_df['timestamp'].max() - stage_df['timestamp'].min()) / 60.0} mins.")
    
    cols = st.columns(3)
    
    with cols[0]:
        metric_card("Loss", lambda: altair_chart(df=combined, x="step", y="loss", key=f"{stage}_loss"))
    
    with cols[1]:
        metric_card("KL Divergence", lambda: altair_chart(df=combined, x="step", y="kl_div", key=f"{stage}_kl"))
    
    with cols[2]:
        metric_card("Task Loss", lambda: altair_chart(df=combined, x="step", y="task_loss", key=f"{stage}_task"))

    # Pruning-specific
    if not pretrain and "reg_edge" in train_s.columns:
        st.subheader("Edge Count")
        altair_chart(df=train_s, x="step", y="reg_edge", color=None, key=f"{stage}_edges")

    # Sampler Histogram
    if not pretrain and "sampler_params" in train_df.columns and not train_df["sampler_params"].dropna().empty:
        altair_histogram(train_df)

    # Raw data
    with st.expander("Raw Data"):
        st.dataframe(stage_df.tail(200))

# ----------------------------
# Tabs
# ----------------------------

stages = get_stages(df)
selected_stage = st.selectbox("Stage", stages)

if selected_stage == "Overview":
    render_overview()
else:
    render_stage(selected_stage, "Pretrain" in selected_stage)