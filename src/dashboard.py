import streamlit as st
import pandas as pd
import json
import psutil
import time
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(layout="wide")

EXPERIMENT_DIR = Path("src/pruning/out")

st.title("Decompiling Transformers Dashboard")

if "selected_stage" not in st.session_state:
    st.session_state.selected_stage = "Overview"

# ----------------------------
# Run selector
# ----------------------------

runs = sorted([p.name for p in EXPERIMENT_DIR.iterdir() if p.is_dir()])
run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = EXPERIMENT_DIR / run_name
metrics_file = run_dir / "metrics.jsonl"

refresh = st.sidebar.checkbox("Auto Refresh", True)
refresh_rate = st.sidebar.slider("Refresh seconds", 1, 10, 2)

# ----------------------------
# Load metrics
# ----------------------------

@st.cache_data(ttl=2)
def load_metrics(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_json(path, lines=True)

df = load_metrics(metrics_file)

if df.empty:
    st.warning("No metrics yet")
    st.stop()

# ----------------------------
# Sidebar stats
# ----------------------------

st.sidebar.markdown("### Run Info")

stages = ["Overview"] + sorted(df["stage"].unique())
st.sidebar.write("Stages:", stages)

selected_stage = st.radio(
    "Select Pruning Stage",
    stages,
    index=stages.index(st.session_state.selected_stage),
    label_visibility="collapsed",
    horizontal=True
)

st.session_state.selected_stage = selected_stage

st.sidebar.write("Steps logged: ", len(df))

# ----------------------------
# Live CPU/GPU Monitoring
# ----------------------------
st.sidebar.write("CPU: ", psutil.cpu_percent())

# ----------------------------
# Content
# ----------------------------

def render_overview():
    st.subheader("Global Pruning Metrics")
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    # Show loss chart
    st.line_chart(train_df, x="step", y="loss", color="stage")
    
    with st.expander("Validation Summary"):
        summary = val_df.groupby("stage").agg({
            "kl_div": "mean",
            "task_loss": "mean",
            "acc_match": "mean",
            "acc_task": "mean"
        })
        
        st.dataframe(summary)

def render_stage(stage: str):
    stage_df = df[df["stage"] == stage]
    train_df = stage_df[stage_df["split"] == "train"]
    val_df = stage_df[stage_df["split"] == "val"]
    
    plot_df = pd.concat([
        train_df.assign(type="train"),
        val_df.assign(type="val")
    ])
        
    st.header(f"{stage} Metrics")

    # ----------------------------
    # Layout
    # ----------------------------

    col1, col2, col3 = st.columns(3)

    # ----------------------------
    # Training curves
    # ----------------------------

    with col1:
        st.subheader("Pruning Stage 1 Overall Loss")
        
        if "loss" in df.columns:
            st.line_chart(
                plot_df,
                x="step",
                y="loss",
                color="type"
            )
        
        st.subheader("Edge Regularization")
        
        if "reg_edge" in df.columns:
            st.line_chart(train_df["reg_edge"])

    # ----------------------------
    # Regularization metrics
    # ----------------------------

    with col2:
        st.subheader("Pruning Stage 1 KL Loss")
        
        if "kl_div" in df.columns:
            st.line_chart(
                plot_df,
                x="step",
                y="kl_div",
                color="type"
            )
        
        st.subheader("Node Regularization")
        
        if "reg_node" in df.columns:
            st.line_chart(train_df["reg_node"])

    with col3:
        st.subheader("Pruning Stage 1 Task Loss")
        
        if "task_loss" in df.columns:
            st.line_chart(
                plot_df,
                x="step",
                y="task_loss",
                color="type"
            )

    # ----------------------------
    # Edge pruning progress
    # ----------------------------

    if "num_edges" in train_df.columns:
        st.subheader("Edge Count Progress")
        
        st.line_chart(train_df["num_edges"])

    # ----------------------------
    # Sampler parameter histogram
    # ----------------------------

    if "sampler_params" in train_df.columns:
        st.subheader("Sampler Parameter Distribution")
        
        latest_params = train_df["sampler_params"].dropna().iloc[-1]
        
        fig, ax = plt.subplots()
        ax.hist(latest_params, bins=20)
        
        st.pyplot(fig, width="content")

    # ----------------------------
    # Raw metrics table
    # ----------------------------

    with st.expander("Raw Training Metrics", expanded=False):
        st.dataframe(train_df.tail(100))
    
    with st.expander("Raw Validation Metrics", expanded=False):
        st.dataframe(val_df.tail(100))


if selected_stage == "Overview":
    render_overview()
else:
    render_stage(st.session_state.selected_stage)

# ----------------------------
# Auto Refresh
# ----------------------------

if refresh:
    time.sleep(refresh_rate)
    st.rerun()