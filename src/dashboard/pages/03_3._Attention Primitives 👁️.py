import base64
import html
import math
import psutil
import re
import streamlit as st
import streamlit.components.v1 as components
from datetime import timedelta
from pathlib import Path

from utils.att_heatmap_paths import collect_interaction_heatmaps, collect_round_heatmaps
from utils.constants import PRUNING_EXPERIMENT_DIR
from utils.dataloader import load_metrics

st.set_page_config(layout="wide")
st.title("Attention Primitive Replacement")

st.markdown(
"""
The attention primitive replacement step involves replacing the attention matrices for the interactions remaining in the
pruned model with simpler primitives that, when applied to the original input, produce the same (or very similar) output. We thus simplify obscure
learned attention matrices with simpler primitives that we can reason about. The replacement approach taken by the authors of the
paper employs two steps:

1. **Greedy Search for Best Primitives**: we cycle through all valid primitives for a given interaction, 
and select the primitive that maximizes the match accuracy between the original and the replaced attention matrix.

2. **Round Fallback**: if the greedy search does not find a primitive that achieves a high enough match accuracy, we fall back to 
a process that encourages the attention matrix to be sparse and as close to integer values as possible. This is done by training a
new matrix with learnable parameters and suitable penalties for each stage.
"""
)


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


def _render_heatmap_carousel(
    heatmaps: list[tuple[str, Path]],
    carousel_id: str,
    *,
    show_hint: bool = True,
) -> None:
    safe_carousel_id = re.sub(r"[^a-zA-Z0-9_-]", "_", carousel_id)
    slides_html = []
    for label, image_path in heatmaps:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        safe_label = html.escape(label)
        slides_html.append(
            f"""
            <div class="heatmap-slide">
                <div class="heatmap-label">{safe_label}</div>
                <img
                    class="heatmap-thumb"
                    src="data:image/png;base64,{encoded}"
                    alt="{safe_label}"
                    data-label="{safe_label}"
                    title="Click to zoom"
                />
            </div>
            """
        )

    components.html(
        f"""
        <style>
            body {{
                margin: 0;
                font-family: sans-serif;
            }}
            .heatmap-carousel-wrap {{
                margin-top: 0.25rem;
            }}
            .heatmap-carousel {{
                display: flex;
                flex-direction: row;
                gap: 1rem;
                overflow-x: auto;
                overflow-y: hidden;
                scroll-snap-type: x mandatory;
                scroll-behavior: smooth;
                padding: 0.25rem 0.25rem 0.75rem;
                -webkit-overflow-scrolling: touch;
            }}
            .heatmap-slide {{
                flex: 0 0 auto;
                scroll-snap-align: start;
                width: min(72vw, 640px);
            }}
            .heatmap-label {{
                font-size: 0.85rem;
                color: #6b7280;
                margin-bottom: 0.4rem;
                text-align: center;
            }}
            .heatmap-thumb {{
                display: block;
                width: 100%;
                max-height: 420px;
                object-fit: contain;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                background: #fff;
                cursor: zoom-in;
                transition: box-shadow 0.15s ease;
            }}
            .heatmap-thumb:hover {{
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
            }}
            .heatmap-hint {{
                font-size: 0.8rem;
                color: #9ca3af;
                text-align: center;
                margin-top: 0.15rem;
            }}
        </style>
        <div class="heatmap-carousel-wrap">
            <div class="heatmap-carousel" id="carousel-{safe_carousel_id}">
                {''.join(slides_html)}
            </div>
        </div>
        <script>
            (function() {{
                function mountRoot() {{
                    try {{
                        if (window.parent && window.parent.document && window.parent !== window) {{
                            return window.parent.document.body;
                        }}
                    }} catch (err) {{
                        /* ignore cross-origin parent access errors */
                    }}
                    return document.body;
                }}

                function ensureOverlay() {{
                    const root = mountRoot();
                    let overlay = root.querySelector("#att-heatmap-overlay");
                    if (overlay) {{
                        return overlay;
                    }}

                    overlay = document.createElement("div");
                    overlay.id = "att-heatmap-overlay";
                    overlay.innerHTML = `
                        <style>
                            #att-heatmap-overlay {{
                                display: none;
                                position: fixed;
                                inset: 0;
                                z-index: 1000000;
                                background: rgba(17, 24, 39, 0.94);
                                flex-direction: column;
                            }}
                            #att-heatmap-overlay.is-open {{
                                display: flex;
                            }}
                            #att-heatmap-overlay .heatmap-lightbox-toolbar {{
                                display: flex;
                                align-items: center;
                                justify-content: space-between;
                                gap: 0.75rem;
                                padding: 0.85rem 1.1rem;
                                color: #f9fafb;
                                font-family: sans-serif;
                                font-size: 0.95rem;
                                flex: 0 0 auto;
                            }}
                            #att-heatmap-overlay .heatmap-lightbox-hint {{
                                font-size: 0.8rem;
                                color: #d1d5db;
                            }}
                            #att-heatmap-overlay .heatmap-lightbox-stage {{
                                flex: 1 1 auto;
                                overflow: auto;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                padding: 1rem;
                            }}
                            #att-heatmap-overlay .heatmap-lightbox-image {{
                                max-width: min(96vw, 1600px);
                                width: auto;
                                height: auto;
                                transform-origin: center center;
                                border-radius: 4px;
                                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                                cursor: zoom-in;
                            }}
                        </style>
                        <div class="heatmap-lightbox-toolbar">
                            <span id="att-heatmap-overlay-label"></span>
                            <span class="heatmap-lightbox-hint">Scroll to zoom · Esc to close</span>
                        </div>
                        <div class="heatmap-lightbox-stage" id="att-heatmap-overlay-stage">
                            <img class="heatmap-lightbox-image" id="att-heatmap-overlay-image" alt="" />
                        </div>
                    `;
                    root.appendChild(overlay);

                    const overlayImage = overlay.querySelector("#att-heatmap-overlay-image");
                    const overlayStage = overlay.querySelector("#att-heatmap-overlay-stage");
                    let scale = 1;

                    function applyZoom() {{
                        overlayImage.style.transform = "scale(" + scale + ")";
                    }}

                    window.attHeatmapZoom = function(delta) {{
                        scale = Math.min(8, Math.max(0.4, scale + delta));
                        applyZoom();
                    }};

                    window.attHeatmapResetZoom = function() {{
                        scale = 1;
                        applyZoom();
                        overlayStage.scrollTop = 0;
                        overlayStage.scrollLeft = 0;
                    }};

                    window.attHeatmapClose = function() {{
                        overlay.classList.remove("is-open");
                        overlayImage.src = "";
                        scale = 1;
                        applyZoom();
                    }};

                    window.attHeatmapOpen = function(img) {{
                        const label = img.dataset.label || img.alt || "Heatmap";
                        overlay.querySelector("#att-heatmap-overlay-label").textContent = label;
                        overlayImage.src = img.src;
                        overlayImage.alt = label;
                        scale = 1;
                        applyZoom();
                        overlay.classList.add("is-open");
                    }};

                    overlayStage.addEventListener("click", function(event) {{
                        if (event.target === overlayStage) {{
                            window.attHeatmapClose();
                        }}
                    }});

                    overlayStage.addEventListener("dblclick", function() {{
                        if (!overlay.classList.contains("is-open")) {{
                            return;
                        }}
                        window.attHeatmapResetZoom();
                    }});

                    overlayStage.addEventListener("wheel", function(event) {{
                        if (!overlay.classList.contains("is-open")) {{
                            return;
                        }}
                        event.preventDefault();
                        const delta = event.deltaY < 0 ? 0.2 : -0.2;
                        window.attHeatmapZoom(delta);
                    }}, {{ passive: false }});

                    const keyTarget = root.ownerDocument || document;
                    keyTarget.addEventListener("keydown", function(event) {{
                        if (!overlay.classList.contains("is-open")) {{
                            return;
                        }}
                        if (event.key === "Escape") {{
                            window.attHeatmapClose();
                        }}
                    }});
                }}

                ensureOverlay();

                const carousel = document.getElementById("carousel-{safe_carousel_id}");
                carousel.addEventListener("click", function(event) {{
                    const img = event.target.closest(".heatmap-thumb");
                    if (!img) {{
                        return;
                    }}
                    ensureOverlay();
                    window.attHeatmapOpen(img);
                }});
            }})();
        </script>
        """,
        height=500,
        scrolling=False,
    )


def _render_interaction_heatmaps(
    run_dir: Path,
    info: dict,
    interaction_id: str,
) -> None:
    heatmaps = collect_interaction_heatmaps(run_dir, info)
    if not heatmaps:
        if info.get("done"):
            st.caption("No saved heatmaps for this interaction yet.")
        return

    st.markdown("**Interaction Heatmaps**")
    _render_heatmap_carousel(heatmaps, interaction_id, show_hint=True)


def _render_round_fallback_heatmaps(run_dir: Path, round_acceptance: list[dict]) -> None:
    heatmaps = collect_round_heatmaps(run_dir, round_acceptance)
    if not heatmaps:
        if round_acceptance:
            st.caption(
                "Round fallback matrices will appear here after the run completes "
                "and heatmaps are saved."
            )
        return

    st.markdown("**Round Fallback Matrices**")
    _render_heatmap_carousel(heatmaps, "round_fallback", show_hint=True)


def _interaction_expander_label(interaction_id: str, info: dict, idx: int) -> str:
    if info["done"] and info["found"]:
        status_icon, status_text = "✅", "Found"
    elif info["done"] and not info["found"]:
        status_icon, status_text = "❌", "Not Found"
    else:
        status_icon, status_text = "⏳", "Searching"
    return (
        f"{status_icon} [{status_text}] Interaction {info['interaction_idx'] or idx}: "
        f"`{interaction_id}`"
    )


def _render_interaction_card(
    run_dir: Path,
    interaction_id: str,
    info: dict,
    idx: int,
    search_info: dict,
    head_states: dict,
) -> None:
    with st.expander(_interaction_expander_label(interaction_id, info, idx)):
        _render_interaction_detail(
            run_dir,
            interaction_id,
            info,
            search_info,
            head_states,
        )


def _render_interaction_detail(
    run_dir: Path,
    interaction_id: str,
    info: dict,
    search_info: dict,
    head_states: dict,
) -> None:
    candidate_total = max(info["num_candidates"], 1)
    candidate_progress = min(info["current_candidate"] / candidate_total, 1.0)

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

    if not info["done"]:
        st.markdown("**Current Candidate**")
        if info["current_primitive"] is not None:
            st.info(
                f"Testing `{_format_primitive_pair(info['current_primitive'], info['current_special_primitive'])}` "
                f"(acc match: **{_pct(info['current_acc_match'])}**, "
                f"acc: **{_pct(info['current_acc'])}**)"
            )
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

    _render_interaction_heatmaps(run_dir, info, interaction_id)

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


def _summary_counts(
    total_interactions: int,
    latest_converted: int,
    search_info: dict,
    greedy_then_round_complete: dict | None,
    round_complete: dict | None,
    greedy_complete: dict | None,
    round_acceptance: list[dict],
    found_count: int,
) -> tuple[int, int, bool]:
    if round_acceptance and total_interactions > 0:
        round_accepted = sum(1 for rec in round_acceptance if bool(rec.get("accepted")))
        if greedy_complete is not None:
            greedy_converted = _optional_int(greedy_complete.get("converted_count"))
            if greedy_converted is not None:
                converted = greedy_converted + round_accepted
                if converted <= total_interactions:
                    return converted, total_interactions, converted == total_interactions
        elif found_count > 0:
            converted = found_count + round_accepted
            if converted <= total_interactions:
                return converted, total_interactions, converted == total_interactions

    pipeline_rec = search_info.get("pipeline_complete")

    for rec in (pipeline_rec, greedy_then_round_complete, round_complete):
        if not rec:
            continue
        converted = _optional_int(rec.get("converted_count"))
        total = _optional_int(rec.get("total_interactions") or rec.get("total_count"))
        if converted is None or total is None:
            continue
        if total != total_interactions or converted > total:
            continue
        return converted, total, converted == total and total > 0

    search_rec = search_info.get("complete")
    if search_rec:
        converted = _optional_int(search_rec.get("converted_count"))
        total = _optional_int(
            search_rec.get("total_interactions") or search_rec.get("total_count")
        )
        if (
            converted is not None
            and total == total_interactions
            and converted <= total
        ):
            return converted, total, converted == total and total > 0

    return latest_converted, total_interactions, False


def _run_finished(
    search_method: str | None,
    search_complete_rec: dict | None,
    pipeline_complete_rec: dict | None,
    greedy_then_round_complete_rec: dict | None,
    round_complete_rec: dict | None,
) -> bool:
    if pipeline_complete_rec:
        return True
    if search_method == "greedy_search_then_round":
        return greedy_then_round_complete_rec is not None
    if search_method == "round":
        return round_complete_rec is not None
    return search_complete_rec is not None


def _round_phase_started(
    round_training_start: dict,
    round_training_latest_step: dict,
    round_training_complete: dict,
    round_stages: list[dict],
    round_acceptance: list[dict],
) -> bool:
    return bool(
        round_training_start
        or round_training_latest_step
        or round_training_complete
        or round_stages
        or round_acceptance
    )


def _parse_att_metrics(records: list[dict]) -> dict:
    search_info: dict = {}
    interactions: dict[str, dict] = {}
    interaction_order: list[str] = []
    head_states: dict[tuple[int, int], dict] = {}
    current_interaction_id: str | None = None
    round_stages: list[dict] = []
    round_acceptance: list[dict] = []
    round_complete_rec: dict | None = None
    greedy_then_round_complete_rec: dict | None = None
    greedy_complete: dict | None = None
    round_training_start: dict[int | None, dict] = {}
    round_training_latest_step: dict[int | None, dict] = {}
    round_training_complete: dict[int | None, dict] = {}

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

        if task == "att_greedy_complete":
            greedy_complete = rec
            continue

        if task == "att_search_complete":
            search_info["complete"] = rec
            continue

        if task == "att_round_stage_complete":
            round_stages.append(rec)
            continue

        if task == "att_round_training_start":
            round_training_start[_optional_int(rec.get("stage"))] = rec
            continue

        if task == "att_round_training_step":
            round_training_latest_step[_optional_int(rec.get("stage"))] = rec
            continue

        if task == "att_round_training_complete":
            round_training_complete[_optional_int(rec.get("stage"))] = rec
            continue

        if task == "att_round_acceptance":
            round_acceptance.append(rec)
            continue

        if task == "att_round_complete":
            round_complete_rec = rec
            continue

        if task == "att_greedy_then_round_complete":
            greedy_then_round_complete_rec = rec
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
            info["total_interactions"] = int(
                rec.get("total_interactions", info["total_interactions"]) or 0
            )
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
            info["converted_so_far"] = int(
                rec.get("converted_so_far", info["converted_so_far"]) or 0
            )
            info["layer"] = _optional_int(rec.get("layer"))
            info["head"] = _optional_int(rec.get("head"))
            activation = _optional_str(rec.get("activation"))
            if activation is not None:
                info["activation"] = activation
            if info["found"] or info["done"]:
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
        progress_label = (
            f"Interactions completed: {completed_count}/{max(len(interactions), 1)}"
        )

    latest_converted = 0
    for info in interactions.values():
        latest_converted = max(latest_converted, info["converted_so_far"])

    converted_count, summary_total, fully_replaced = _summary_counts(
        total_interactions,
        latest_converted,
        search_info,
        greedy_then_round_complete_rec,
        round_complete_rec,
        greedy_complete,
        round_acceptance,
        found_count,
    )

    search_complete_rec = search_info.get("complete")
    pipeline_complete_rec = search_info.get("pipeline_complete")
    search_method = search_info.get("search_method")
    run_finished = _run_finished(
        search_method,
        search_complete_rec,
        pipeline_complete_rec,
        greedy_then_round_complete_rec,
        round_complete_rec,
    )

    round_phase_started = _round_phase_started(
        round_training_start,
        round_training_latest_step,
        round_training_complete,
        round_stages,
        round_acceptance,
    )
    show_round_progress = False
    if search_method == "round":
        show_round_progress = round_phase_started
    elif search_method == "greedy_search_then_round":
        show_round_progress = greedy_complete is not None and round_phase_started

    greedy_search_finished = (
        greedy_complete is not None
        or (total_interactions > 0 and completed_count >= total_interactions)
        or run_finished
    )

    live_refresh = (
        not run_finished
        and (
            in_progress_count > 0
            or (not greedy_search_finished and bool(interaction_order))
            or (show_round_progress and not round_acceptance)
        )
    )

    return {
        "search_info": search_info,
        "interactions": interactions,
        "interaction_order": interaction_order,
        "head_states": head_states,
        "current_interaction_id": current_interaction_id,
        "round_stages": round_stages,
        "round_acceptance": round_acceptance,
        "round_training_start": round_training_start,
        "round_training_latest_step": round_training_latest_step,
        "round_training_complete": round_training_complete,
        "total_interactions": total_interactions,
        "global_progress": global_progress,
        "progress_label": progress_label,
        "converted_count": converted_count,
        "summary_total": summary_total,
        "fully_replaced": fully_replaced,
        "greedy_found_count": found_count,
        "failed_count": failed_count,
        "in_progress_count": in_progress_count,
        "run_finished": run_finished,
        "show_round_progress": show_round_progress,
        "greedy_search_finished": greedy_search_finished,
        "live_refresh": live_refresh,
        "search_complete_rec": search_complete_rec,
        "pipeline_complete_rec": pipeline_complete_rec,
    }


def _render_dashboard(parsed: dict, run_name: str, run_dir: Path) -> None:
    search_info = parsed["search_info"]
    interaction_order = parsed["interaction_order"]

    st.subheader("Overview")
    st.progress(parsed["global_progress"], text=parsed["progress_label"])

    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Converted", f"{parsed['converted_count']}/{parsed['summary_total']}")
    stat2.metric("Greedy Found", parsed["greedy_found_count"])
    stat3.metric("Greedy Not Found", parsed["failed_count"])
    stat4.metric("In Progress", parsed["in_progress_count"])

    with st.expander("Search Configuration", expanded=False):
        if search_info:
            cfg1, cfg2, cfg3, cfg4 = st.columns(4)
            cfg1.metric("Search Method", search_info.get("search_method", "N/A"))
            cfg2.metric("Baseline Acc Match", _pct(search_info.get("baseline_acc_match")))
            cfg3.metric("Acc Match Threshold", _pct(search_info.get("acc_match_threshold")))
            cfg4.metric("Total Interactions", parsed["total_interactions"] or "N/A")

            baseline_cols = st.columns(3)
            baseline_cols[0].caption(
                f"Baseline accuracy: {_pct(search_info.get('baseline_acc'))}"
            )
            baseline_cols[1].caption(f"Baseline KL: {search_info.get('baseline_kl', 'N/A')}")
            baseline_cols[2].caption(
                f"Eval: {search_info.get('num_test_steps', 'N/A')} steps, "
                f"batch {search_info.get('eval_batch_size', 'N/A')}"
            )
        else:
            st.caption("Search has not started yet.")

    if parsed["run_finished"]:
        acc_before = None
        acc_after = None
        for rec in (parsed["pipeline_complete_rec"], parsed["search_complete_rec"]):
            if not rec:
                continue
            if acc_before is None:
                acc_before = rec.get("acc_match_before")
            if acc_after is None:
                acc_after = rec.get("acc_match_after")

        with st.expander("Final Results", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Converted", f"{parsed['converted_count']}/{parsed['summary_total']}")
            r2.metric("Fully Replaced", "Yes" if parsed["fully_replaced"] else "No")
            r3.metric("Acc Match Before", _pct(acc_before))
            r4.metric("Acc Match After", _pct(acc_after))
            if parsed["pipeline_complete_rec"]:
                st.metric(
                    "Full-Batch Acc Match",
                    _pct(parsed["pipeline_complete_rec"].get("acc_match_after_full_batch")),
                )

    if parsed["show_round_progress"]:
        with st.expander("Round Fallback Progress", expanded=False):
            stage_keys = sorted(
                set(parsed["round_training_start"].keys())
                | set(parsed["round_training_latest_step"].keys())
                | set(parsed["round_training_complete"].keys()),
                key=lambda x: -1 if x is None else x,
            )
            if stage_keys:
                st.markdown("**Round Training (Live)**")
                for stage_key in stage_keys:
                    start_rec = parsed["round_training_start"].get(stage_key, {})
                    step_rec = parsed["round_training_latest_step"].get(stage_key, {})
                    done_rec = parsed["round_training_complete"].get(stage_key, {})

                    total_steps = int(
                        done_rec.get("total_steps")
                        or step_rec.get("total_steps")
                        or start_rec.get("num_steps")
                        or 0
                    )
                    cur_step = int(done_rec.get("final_step") or step_rec.get("step") or 0)
                    progress = float(
                        done_rec.get("progress")
                        if not _is_missing(done_rec.get("progress"))
                        else step_rec.get("progress", 0.0)
                    )
                    progress = min(max(progress, 0.0), 1.0)

                    title = f"Stage `{stage_key}`" if stage_key is not None else "Stage `single`"
                    st.markdown(f"_{title}_")
                    st.progress(
                        progress,
                        text=(
                            f"{cur_step}/{total_steps} steps"
                            if total_steps > 0
                            else f"{cur_step} steps"
                        ),
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Match Acc", _pct(step_rec.get("match_acc")))
                    m2.metric("Threshold", _pct(step_rec.get("match_acc_threshold")))
                    m3.metric("Penalty", f"{step_rec.get('penalty', 'N/A')}")
                    m4.metric("Full Loss", f"{step_rec.get('full_loss', 'N/A')}")

                    stop_reason = done_rec.get("stop_reason")
                    if stop_reason:
                        st.caption(f"Stop reason: `{stop_reason}`")
            else:
                st.caption("Round training has not started yet.")

            if parsed["round_stages"]:
                st.markdown("**Round Training Stages**")
                for stage in parsed["round_stages"]:
                    stage_id = stage.get("stage")
                    acc_match = _pct(stage.get("acc_match"))
                    kl = stage.get("kl")
                    st.caption(
                        f"Stage `{stage_id}` complete — acc match: **{acc_match}**, KL: **{kl}**"
                    )

            if parsed["round_acceptance"]:
                accepted = sum(1 for rec in parsed["round_acceptance"] if bool(rec.get("accepted")))
                total = len(parsed["round_acceptance"])
                st.markdown("**Per-Interaction Round Acceptance**")
                a1, a2 = st.columns(2)
                a1.metric("Rounded Accepted", f"{accepted}/{total}")
                a2.metric("Rounded Rejected", f"{total - accepted}/{total}")
                _render_round_fallback_heatmaps(run_dir, parsed["round_acceptance"])
            else:
                st.caption("No round acceptance decisions logged yet.")

    if not interaction_order:
        st.info("Waiting for the first interaction to start.")
        return

    with st.expander("Greedy Search Progress", expanded=False):
        current_interaction_id = parsed["current_interaction_id"]
        if current_interaction_id and not parsed["greedy_search_finished"]:
            active_info = parsed["interactions"][current_interaction_id]
            st.caption(
                f"Currently searching: `{current_interaction_id}` "
                f"({active_info['current_candidate']}/"
                f"{max(active_info['num_candidates'], 1)} candidates)"
            )
        for idx, interaction_id in enumerate(interaction_order, start=1):
            _render_interaction_card(
                run_dir,
                interaction_id,
                parsed["interactions"][interaction_id],
                idx,
                search_info,
                parsed["head_states"],
            )


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
parsed = _parse_att_metrics(records)

st.sidebar.markdown("### Refresh")
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh",
    value=parsed["live_refresh"],
    disabled=not parsed["live_refresh"],
    key="att_auto_refresh_enabled",
)

use_live_fragment = auto_refresh and parsed["live_refresh"]


@st.fragment(run_every=timedelta(seconds=2) if use_live_fragment else None)
def render_live_dashboard() -> None:
    if use_live_fragment:
        live_df = load_metrics(metrics_file)
        if live_df.empty:
            live_parsed = parsed
        else:
            live_parsed = _parse_att_metrics(live_df.to_dict(orient="records"))
    else:
        live_parsed = parsed
    _render_dashboard(live_parsed, run_name, run_dir)


render_live_dashboard()
