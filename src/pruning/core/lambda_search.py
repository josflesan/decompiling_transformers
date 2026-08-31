"""
Adaptive lambda search for pruning stages.

Ports the geometric-mean bracketing heuristic from the original repo's
delineate_curve_for_model.py without requiring a full hyperparameter sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LambdaTrialResult:
    """Stores the result of a single lambda probe trial"""
    acc_match: float
    num_edges: int
    lamb: float
    probe: bool
    feasible: bool = False


def compute_threshold_acc(
    baseline_acc: float,
    reference_acc: float,
    relative_gap: float = 0.9,
) -> float:
    """Used to determine whether a lambda probe is feasible"""
    return baseline_acc + relative_gap * (reference_acc - baseline_acc)


def suggest_probe_lambdas(
    center_lamb: float,
    probe_lambdas: Optional[Sequence[float]] = None,
    scale_factor: float = 10.0,
    max_probes: int = 3,
) -> List[float]:
    if probe_lambdas is not None:
        candidates = list(probe_lambdas)
    else:
        candidates = [
            center_lamb / scale_factor,
            center_lamb,
            center_lamb * scale_factor,
        ]

    unique: List[float] = []
    seen = set()
    for lamb in candidates:
        key = round(lamb, 15)
        if key not in seen:
            seen.add(key)
            unique.append(lamb)

    return unique[:max_probes]


def suggest_refined_lambdas(
    above_thr: Sequence[float],
    below_thr: Sequence[float],
) -> Optional[float]:
    if not above_thr or not below_thr:
        return None

    coef_high = max(above_thr)
    coef_low = min(below_thr)

    if coef_high > 0 and coef_low > 0:
        return math.sqrt(coef_high * coef_low)
    if coef_high < 0 and coef_low < 0:
        return -math.sqrt(coef_high * coef_low)
    return (coef_high + coef_low) / 2


def bracket_exists(trials: Sequence[LambdaTrialResult], threshold_acc: float) -> bool:
    above = any(t.acc_match >= threshold_acc for t in trials)
    below = any(t.acc_match < threshold_acc for t in trials)
    return above and below


def select_best_trial(
    trials: Sequence[LambdaTrialResult],
    threshold_acc: float,
) -> Tuple[LambdaTrialResult, bool]:
    """
    Pick the trial with fewest edges among feasible results.

    Returns (best_trial, used_fallback) where used_fallback is True when no
    trial met the threshold and the smallest-lambda trial was returned instead.
    """
    if not trials:
        raise ValueError("select_best_trial requires at least one trial result")

    feasible = [t for t in trials if t.acc_match >= threshold_acc]
    if feasible:
        best = min(feasible, key=lambda t: (t.num_edges, -t.lamb))
        return best, False

    fallback = min(trials, key=lambda t: t.lamb)
    return fallback, True


def resolve_probe_num_steps(num_steps: int, probe_num_steps: Optional[int]) -> int:
    """Determines the number of steps to use for a lambda probe"""
    if probe_num_steps is not None:
        return probe_num_steps
    
    return max(100, int(0.25 * num_steps))
