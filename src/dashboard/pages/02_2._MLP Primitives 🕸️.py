import psutil
import math
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics

st.set_page_config(layout="wide")
st_autorefresh(interval=2000)

st.title("MLP Primitive Replacement Dashboard")

runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No runs found.")
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name

# Prefer per-task metrics file, fallback to run-root metrics.
metrics_file = run_dir / "mlp_primitives" / "metrics.jsonl"
if not metrics_file.exists():
    metrics_file = run_dir / "metrics.jsonl"

st.sidebar.markdown("### System")
col1, col2 = st.sidebar.columns(2)
col1.metric("CPU %", psutil.cpu_percent())
col2.metric("RAM %", psutil.virtual_memory().percent)

if not metrics_file.exists():
    st.warning(f"No metrics file found at `{run_dir}`.")
    st.stop()

df = load_metrics(metrics_file)
if df.empty:
    st.info("No primitive replacement metrics logged yet.")
    st.stop()

records = df.to_dict(orient="records")

paths = {}
path_order = []
global_path_idx = 0
global_total_paths = 0

for rec in records:
    task = rec.get("task")

    if task == "primitive_replacement":
        global_path_idx = int(rec.get("path_idx", global_path_idx) or 0)
        global_total_paths = int(rec.get("total_paths", global_total_paths) or 0)
        continue

    path = rec.get("path")
    if not path:
        continue

    if path not in paths:
        paths[path] = {
            "collected": 0,
            "collection_total": 0,
            "current_primitive": 0,
            "total_primitives": 0,
            "current_primitive_name": None,
            "current_acc_task": None,
            "replacement_done": False,
            "replacement_failed": False,
            "best_primitive": None,
            "best_accuracy": None,
        }
        path_order.append(path)

    info = paths[path]

    if task == "mlp_data_collection":
        info["collected"] = int(rec.get("collected", info["collected"]) or 0)
        info["collection_total"] = int(rec.get("total", info["collection_total"]) or 0)
    elif task == "primitive_search":
        if "current_primitive" in rec and not math.isnan(rec['current_primitive']):
            info["current_primitive"] = int(rec.get("current_primitive", info["current_primitive"]) or 0)
            info["total_primitives"] = int(rec.get("total_primitives", info["total_primitives"]) or 0)
            info["current_primitive_name"] = rec.get("primitive", info["current_primitive_name"])
            info["current_acc_task"] = rec.get("acc_task", info["current_acc_task"])
        if "failed" in rec and not math.isnan(rec['failed']):
            info["replacement_done"] = True
            info["replacement_failed"] = bool(rec.get("failed"))
            info["best_primitive"] = rec.get("best_primitive", info["best_primitive"])
            info["best_accuracy"] = rec.get("best_primitive_accuracy", info["best_accuracy"])

if global_total_paths > 0:
    global_progress = min(global_path_idx / global_total_paths, 1.0)
else:
    done_count = sum(1 for p in paths.values() if p["replacement_done"])
    global_total_paths = max(len(paths), 1)
    global_progress = min(done_count / global_total_paths, 1.0)
    global_path_idx = done_count

st.subheader("Global Progress")
st.progress(global_progress, text=f"Primitive replacement: {global_path_idx}/{global_total_paths} paths")

completed = sum(1 for p in paths.values() if p["replacement_done"] and not p["replacement_failed"])
failed = sum(1 for p in paths.values() if p["replacement_done"] and p["replacement_failed"])
in_progress = max(len(paths) - completed - failed, 0)
stat1, stat2, stat3 = st.columns(3)
stat1.metric("Succeeded", completed)
stat2.metric("Failed", failed)
stat3.metric("In Progress", in_progress)

st.subheader("Per-Path Progress")

for idx, path in enumerate(path_order, start=1):
    info = paths[path]
    collection_total = max(info["collection_total"], 1)
    primitive_total = max(info["total_primitives"], 1)
    collection_progress = min(info["collected"] / collection_total, 1.0)
    primitive_progress = min(info["current_primitive"] / primitive_total, 1.0)

    with st.expander(f"Path {idx}: {path}", expanded=False):
        st.markdown("**Data Collection**")
        st.progress(
            collection_progress,
            text=f"{info['collected']}/{info['collection_total']} examples collected"
            if info["collection_total"] > 0
            else "Waiting for data collection to start",
        )

        st.markdown("**Primitive Testing**")
        st.progress(
            primitive_progress,
            text=f"{info['current_primitive']}/{info['total_primitives']} primitives tested"
            if info["total_primitives"] > 0
            else "Waiting for primitive testing to start",
        )

        st.markdown("**Replacement Status**")
        best_primitive = info["best_primitive"] or "N/A"
        best_accuracy = info["best_accuracy"]
        best_accuracy_pct = f"{100 * best_accuracy:.2f}%" if best_accuracy is not None else "N/A"

        if info["replacement_done"] and not info["replacement_failed"]:
            st.success(
                f"Primitive replacement succeeded. Best primitive: `{best_primitive}` "
                f"with accuracy: **{best_accuracy_pct}**."
            )
        elif info["replacement_done"] and info["replacement_failed"]:
            st.warning(
                f"Primitive replacement failed. Best candidate: `{best_primitive}` "
                f"with accuracy: **{best_accuracy_pct}**."
            )
        else:
            current_name = info["current_primitive_name"] or "N/A"
            current_acc = info["current_acc_task"]
            current_acc_pct = f"{100 * current_acc:.2f}%" if current_acc is not None else "N/A"
            st.info(
                f"Primitive replacement in progress. Current primitive: `{current_name}` "
                f"(task accuracy: **{current_acc_pct}**)."
            )