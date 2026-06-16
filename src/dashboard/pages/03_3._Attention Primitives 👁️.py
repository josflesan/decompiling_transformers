import math
import psutil
import streamlit as st
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics

st.set_page_config(layout="wide")
st_autorefresh(interval=2000)

st.title("Attention Primitive Replacement Dashboard")


def _short_primitive(label: str | None) -> str:
    if not label:
        return "N/A"
    return str(label).split(" | ")[0]


def _format_primitive_pair(primitive: str | None, special: str | None) -> str:
    if not primitive and not special:
        return "N/A"
    if primitive == special or special is None:
        return _short_primitive(primitive)
    return f"{_short_primitive(primitive)} + {_short_primitive(special)}"


def _format_best_primitive(best: dict | None) -> str:
    if not best:
        return "N/A"
    if best.get("replacement_matrix"):
        rounded = best.get("rounded")
        if rounded is True:
            return "Replacement Matrix [rounded]"
        if rounded is False:
            return "Replacement Matrix [continuous]"
        return "Replacement Matrix"
    if best.get("name"):
        return str(best["name"])
    return _format_primitive_pair(best.get("primitive"), best.get("special_primitive"))


def _pct(value: float | None) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{100 * float(value):.2f}%"


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _optional_int(value) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _new_interaction_state() -> dict:
    return {
        "interaction_idx": 0,
        "total_interactions": 0,
        "phase": None,
        "layer": None,
        "head": None,
        "activation_q": None,
        "activation_k": None,
        "activation": None,
        "num_candidates": 0,
        "current_candidate": 0,
        "current_primitive": None,
        "current_special_primitive": None,
        "current_acc_match": None,
        "current_acc": None,
        "current_accepted": None,
        "best_acc_match": None,
        "best_acc": None,
        "best_candidate_primitive": None,
        "best_candidate_special": None,
        "done": False,
        "found": False,
        "best_primitive": None,
        "converted_so_far": 0,
    }


runs = sorted([p.name for p in PRUNING_EXPERIMENT_DIR.iterdir() if p.is_dir()])
if not runs:
    st.warning("No runs found.")
    st.stop()

run_name = st.sidebar.selectbox("Select Run", runs)
run_dir = PRUNING_EXPERIMENT_DIR / run_name

metrics_file = run_dir / "att_primitives" / "metrics.jsonl"
if not metrics_file.exists():
    metrics_file = run_dir / "metrics.jsonl"

st.sidebar.markdown("### System")
col1, col2 = st.sidebar.columns(2)
col1.metric("CPU %", psutil.cpu_percent())
col2.metric("RAM %", psutil.virtual_memory().percent)

if not metrics_file.exists():
    st.warning(f"No metrics file found at `{run_dir / 'att_primitives'}`.")
    st.stop()

df = load_metrics(metrics_file)
if df.empty:
    st.info("No attention primitive replacement metrics logged yet.")
    st.stop()

records = df.to_dict(orient="records")

search_info: dict = {}
interactions: dict[str, dict] = {}
interaction_order: list[str] = []
head_states: dict[tuple[int, int], dict] = {}
current_interaction_id: str | None = None
round_stages: list[dict] = []
round_acceptance: list[dict] = []
greedy_then_round_complete: dict | None = None

for rec in records:
    task = rec.get("task")

    if task == "att_search_start":
        search_info = {
            "search_method": rec.get("search_method"),
            "total_interactions": int(rec.get("total_interactions", 0) or 0),
            "baseline_acc_match": rec.get("baseline_acc_match"),
            "baseline_acc": rec.get("baseline_acc"),
            "baseline_kl": rec.get("baseline_kl"),
            "baseline_task_loss": rec.get("baseline_task_loss"),
            "acc_match_threshold": rec.get("acc_match_threshold"),
            "search_threshold": rec.get("search_threshold"),
            "threshold_on_acc": rec.get("threshold_on_acc"),
            "num_test_steps": rec.get("num_test_steps"),
            "eval_batch_size": rec.get("eval_batch_size"),
            "complete": None,
            "pipeline_complete": None,
        }
        continue

    if task in ("att_search_complete", "att_greedy_complete"):
        search_info["complete"] = rec
        continue

    if task == "att_round_stage_complete":
        round_stages.append(rec)
        continue

    if task == "att_round_acceptance":
        round_acceptance.append(rec)
        continue

    if task in ("att_round_complete", "att_greedy_then_round_complete"):
        greedy_then_round_complete = rec
        continue

    if task == "att_pipeline_complete":
        search_info["pipeline_complete"] = rec
        continue

    if task == "att_head_state":
        layer = _optional_int(rec.get("layer"))
        head = _optional_int(rec.get("head"))
        if layer is not None and head is not None:
            head_states[(layer, head)] = rec.get("interactions", {})
        continue

    interaction_id = _optional_str(rec.get("interaction_id"))
    if not interaction_id:
        continue

    if interaction_id not in interactions:
        interactions[interaction_id] = _new_interaction_state()
        interaction_order.append(interaction_id)

    info = interactions[interaction_id]

    if task == "att_interaction_start":
        current_interaction_id = interaction_id
        info["interaction_idx"] = int(rec.get("interaction_idx", info["interaction_idx"]) or 0)
        info["total_interactions"] = int(rec.get("total_interactions", info["total_interactions"]) or 0)
        info["phase"] = _optional_str(rec.get("phase")) or info["phase"]
        info["layer"] = _optional_int(rec.get("layer"))
        info["head"] = _optional_int(rec.get("head"))
        info["activation_q"] = _optional_str(rec.get("activation_q"))
        info["activation_k"] = _optional_str(rec.get("activation_k"))
        info["activation"] = _optional_str(rec.get("activation"))
        info["num_candidates"] = int(rec.get("num_candidates", info["num_candidates"]) or 0)
        info["converted_so_far"] = int(rec.get("converted_so_far", info["converted_so_far"]) or 0)
        info["current_candidate"] = 0
        info["done"] = False
        info["found"] = False
        info["best_primitive"] = None
        info["best_acc_match"] = None
        info["best_acc"] = None
        info["best_candidate_primitive"] = None
        info["best_candidate_special"] = None

    elif task == "att_candidate_eval":
        current_interaction_id = interaction_id
        if rec.get("phase"):
            info["phase"] = _optional_str(rec.get("phase"))
        info["layer"] = _optional_int(rec.get("layer"))
        info["head"] = _optional_int(rec.get("head"))
        activation = _optional_str(rec.get("activation"))
        if activation is not None:
            info["activation"] = activation
        candidate_idx = int(rec.get("candidate_idx", 0) or 0)
        info["current_candidate"] = candidate_idx + 1
        info["num_candidates"] = max(
            info["num_candidates"],
            0,
            candidate_idx + 1,
        )
        info["current_primitive"] = rec.get("primitive")
        info["current_special_primitive"] = rec.get("special_primitive")
        info["current_acc_match"] = rec.get("acc_match")
        info["current_acc"] = rec.get("acc")
        info["current_accepted"] = rec.get("accepted")

        acc_match = rec.get("acc_match")
        if not _is_missing(acc_match):
            if info["best_acc_match"] is None or float(acc_match) > float(info["best_acc_match"]):
                info["best_acc_match"] = float(acc_match)
                info["best_acc"] = rec.get("acc")
                info["best_candidate_primitive"] = rec.get("primitive")
                info["best_candidate_special"] = rec.get("special_primitive")

        if rec.get("accepted"):
            info["best_primitive"] = {
                "primitive": rec.get("primitive"),
                "special_primitive": rec.get("special_primitive"),
                "scaling_factor": rec.get("scaling_factor"),
            }

    elif task == "att_interaction_complete":
        info["done"] = True
        info["found"] = bool(rec.get("found"))
        info["best_primitive"] = rec.get("best_primitive", info["best_primitive"])
        info["converted_so_far"] = int(rec.get("converted_so_far", info["converted_so_far"]) or 0)
        info["layer"] = _optional_int(rec.get("layer"))
        info["head"] = _optional_int(rec.get("head"))
        activation = _optional_str(rec.get("activation"))
        if activation is not None:
            info["activation"] = activation
        if info["found"]:
            current_interaction_id = None
        elif info["done"]:
            current_interaction_id = None

total_interactions = int(search_info.get("total_interactions", 0) or 0)
if total_interactions <= 0:
    for info in interactions.values():
        total_interactions = max(total_interactions, int(info.get("total_interactions", 0) or 0))

completed_count = sum(1 for info in interactions.values() if info["done"])
found_count = sum(1 for info in interactions.values() if info["done"] and info["found"])
failed_count = sum(1 for info in interactions.values() if info["done"] and not info["found"])
in_progress_count = max(len(interactions) - completed_count, 0)

if total_interactions > 0:
    if completed_count < total_interactions and current_interaction_id:
        active_idx = interactions[current_interaction_id]["interaction_idx"]
    else:
        active_idx = completed_count
    global_progress = min(active_idx / total_interactions, 1.0)
    progress_label = f"Interaction search: {active_idx}/{total_interactions}"
else:
    global_progress = min(completed_count / max(len(interactions), 1), 1.0)
    progress_label = f"Interactions completed: {completed_count}/{max(len(interactions), 1)}"

latest_converted = 0
for info in interactions.values():
    latest_converted = max(latest_converted, info["converted_so_far"])

st.subheader("Global Progress")
st.progress(global_progress, text=progress_label)

stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Converted", latest_converted)
stat2.metric("Found", found_count)
stat3.metric("Not Found", failed_count)
stat4.metric("In Progress", in_progress_count)

if search_info:
    st.markdown("**Search Configuration**")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    cfg1.metric("Search Method", search_info.get("search_method", "N/A"))
    cfg2.metric("Baseline Acc Match", _pct(search_info.get("baseline_acc_match")))
    cfg3.metric("Acc Match Threshold", _pct(search_info.get("acc_match_threshold")))
    cfg4.metric("Total Interactions", total_interactions or "N/A")

    baseline_cols = st.columns(3)
    baseline_cols[0].caption(f"Baseline accuracy: {_pct(search_info.get('baseline_acc'))}")
    baseline_cols[1].caption(f"Baseline KL: {search_info.get('baseline_kl', 'N/A')}")
    baseline_cols[2].caption(f"Eval: {search_info.get('num_test_steps', 'N/A')} steps, batch {search_info.get('eval_batch_size', 'N/A')}")

search_complete_rec = search_info.get("complete")
pipeline_complete_rec = search_info.get("pipeline_complete")
if search_complete_rec or pipeline_complete_rec:
    st.markdown("**Search Complete**")
    if search_complete_rec:
        c1, c2, c3 = st.columns(3)
        c1.metric("Acc Match Before", _pct(search_complete_rec.get("acc_match_before")))
        c2.metric("Acc Match After", _pct(search_complete_rec.get("acc_match_after")))
        c3.metric("Fully Replaced", "Yes" if search_complete_rec.get("fully_replaced") else "No")
    if pipeline_complete_rec:
        p1, p2, p3 = st.columns(3)
        total_count = pipeline_complete_rec.get("total_count", total_interactions)
        p1.metric("Converted", f"{int(pipeline_complete_rec.get('converted_count', 0) or 0)}/{int(total_count or 0)}")
        p2.metric("Full-Batch Acc Match", _pct(pipeline_complete_rec.get("acc_match_after_full_batch")))
        p3.metric(
            "Fully Replaced",
            "Yes" if pipeline_complete_rec.get("fully_replaced") else "No",
        )

if search_info.get("search_method") in ("round", "greedy_search_then_round"):
    st.subheader("Round Fallback Progress")
    if round_stages:
        st.markdown("**Round Training Stages**")
        for stage in round_stages:
            stage_id = stage.get("stage")
            acc_match = _pct(stage.get("acc_match"))
            kl = stage.get("kl")
            st.caption(f"Stage `{stage_id}` complete — acc match: **{acc_match}**, KL: **{kl}**")
    else:
        st.caption("Round training stages have not started yet.")

    if round_acceptance:
        accepted = sum(1 for rec in round_acceptance if bool(rec.get("accepted")))
        total = len(round_acceptance)
        st.markdown("**Per-Interaction Round Acceptance**")
        a1, a2 = st.columns(2)
        a1.metric("Rounded Accepted", f"{accepted}/{total}")
        a2.metric("Rounded Rejected", f"{total - accepted}/{total}")
        for rec in round_acceptance:
            key = rec.get("rounder_key", "unknown")
            acc = _pct(rec.get("acc_match"))
            nz = int(rec.get("non_zero", 0) or 0)
            tp = int(rec.get("total_params", 0) or 0)
            if rec.get("accepted"):
                st.success(
                    f"`{key}` accepted at {acc} "
                    f"(non-zero: {nz}/{tp})"
                )
            else:
                st.warning(
                    f"`{key}` rejected at {acc} "
                    f"(non-zero: {nz}/{tp})"
                )
    else:
        st.caption("No round acceptance decisions logged yet.")

    if greedy_then_round_complete:
        st.markdown("**Greedy-Then-Round Summary**")
        g1, g2, g3 = st.columns(3)
        g1.metric(
            "Converted Interactions",
            int(greedy_then_round_complete.get("converted_count", 0) or 0),
        )
        g2.metric(
            "Total Interactions",
            int(greedy_then_round_complete.get("total_interactions", total_interactions) or 0),
        )
        g3.metric(
            "Fully Replaced (count)",
            int(greedy_then_round_complete.get("fully_replaced", 0) or 0),
        )

st.subheader("Per-Interaction Progress")

if not interaction_order:
    st.info("Waiting for the first interaction to start.")
    st.stop()

for idx, interaction_id in enumerate(interaction_order, start=1):
    info = interactions[interaction_id]
    candidate_total = max(info["num_candidates"], 1)
    candidate_progress = min(info["current_candidate"] / candidate_total, 1.0)

    if info["done"] and info["found"]:
        status_icon = "✅"
        status_text = "Found"
    elif info["done"] and not info["found"]:
        status_icon = "❌"
        status_text = "Not Found"
    else:
        status_icon = "⏳"
        status_text = "Searching"

    is_active = interaction_id == current_interaction_id
    expander_label = f"{status_icon} [{status_text}] Interaction {info['interaction_idx'] or idx}: `{interaction_id}`"

    with st.expander(expander_label, expanded=is_active):
        if info["phase"] == "attention" and info["layer"] is not None and info["head"] is not None:
            if info["activation_q"] is None:
                st.caption(
                    f"Layer {info['layer']}, Head {info['head']} — "
                    f"bias K: `{info['activation_k']}`"
                )
            else:
                st.caption(
                    f"Layer {info['layer']}, Head {info['head']} — "
                    f"Q: `{info['activation_q']}`, K: `{info['activation_k']}`"
                )
        elif info["phase"] == "lm_head":
            st.caption(f"LM head projection — activation: `{info['activation']}`")

        st.markdown("**Candidate Search**")
        st.progress(
            candidate_progress,
            text=(
                f"{info['current_candidate']}/{info['num_candidates']} candidates evaluated"
                if info["num_candidates"] > 0
                else "Waiting for candidate evaluation to start"
            ),
        )

        st.markdown("**Current Candidate**")
        if not info["done"] and info["current_primitive"] is not None:
            st.info(
                f"Testing `{_format_primitive_pair(info['current_primitive'], info['current_special_primitive'])}` "
                f"(acc match: **{_pct(info['current_acc_match'])}**, "
                f"acc: **{_pct(info['current_acc'])}**)"
            )
        elif info["done"]:
            st.caption("Candidate search finished for this interaction.")
        else:
            st.caption("No candidates evaluated yet.")

        st.markdown("**Best Candidate So Far**")
        if info["best_primitive"]:
            st.success(
                f"Accepted primitive: `{_format_best_primitive(info['best_primitive'])}`"
            )
        elif info["best_candidate_primitive"] is not None:
            st.warning(
                f"Best tested (not accepted): "
                f"`{_format_primitive_pair(info['best_candidate_primitive'], info['best_candidate_special'])}` "
                f"with acc match **{_pct(info['best_acc_match'])}** "
                f"(threshold: **{_pct(search_info.get('acc_match_threshold'))}**)"
            )
        else:
            st.caption("No candidates evaluated yet.")

        if info["done"]:
            st.markdown("**Result**")
            if info["found"]:
                st.success(
                    f"Primitive found: `{_format_best_primitive(info['best_primitive'])}`"
                )
            else:
                st.warning("No primitive met the accuracy threshold for this interaction.")

        layer = _optional_int(info.get("layer"))
        head = _optional_int(info.get("head"))
        if layer is not None and head is not None:
            head_key = (layer, head)
            head_state = head_states.get(head_key)
            if head_state:
                st.markdown("**Head State**")
                for state_id, primitive in sorted(head_state.items()):
                    if primitive is None:
                        st.caption(f"`{state_id}`: pending")
                    else:
                        st.caption(f"`{state_id}`: `{_format_best_primitive(primitive)}`")
