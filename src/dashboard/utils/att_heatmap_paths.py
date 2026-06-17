"""Resolve saved attention primitive heatmap paths for the dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_ROUNDER_KEY_ATTN = re.compile(r"^(\d+)-(\d+)-(.+)$")

def _interaction_save_name(info: dict) -> Optional[str]:
    if info.get("phase") == "lm_head":
        activation = info.get("activation")
        if not activation:
            return None
        return "bias" if activation == "vocab_bias" else activation

    activation_k = info.get("activation_k")
    if not activation_k:
        return None
    activation_q = info.get("activation_q")
    if activation_q is None:
        return f"bias-{activation_k}"
    return f"{activation_q}-{activation_k}"


def _is_flat_heatmap_root(root: Path) -> bool:
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "lm_head" or child.name == "pos-tok":
            return True
        if re.fullmatch(r"\d+-\d+", child.name):
            return True
    return False


def _heatmap_root(run_dir: Path) -> Optional[Path]:
    root = run_dir / "att_primitives" / "heatmaps"
    if not root.exists():
        return None
    if _is_flat_heatmap_root(root):
        return root
    
    # Legacy layout: heatmaps/<task_name>/...
    task_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not task_dirs:
        return None
    return task_dirs[0]


def _k_q_suffix_matches_save_name(k_q_suffix: str, save_name: str) -> bool:
    """Match rounder suffix ``k-q`` to on-disk save name ``q-k``."""
    if save_name.startswith("bias-"):
        return k_q_suffix == f"{save_name[5:]}-None"
    for i in range(len(save_name)):
        if save_name[i] != "-":
            continue
        q, k = save_name[:i], save_name[i + 1 :]
        if f"{k}-{q}" == k_q_suffix:
            return True
    return k_q_suffix == save_name


def _format_rounder_key_display(rounder_key: str) -> str:
    """Format a rounder key for dashboard display."""
    attn_match = _ROUNDER_KEY_ATTN.match(rounder_key)
    if attn_match:
        layer, head, suffix = attn_match.groups()
        if suffix.endswith("-None"):
            return f"{layer}-{head}-bias-{suffix[:-5]}"
        for i in range(len(suffix)):
            if suffix[i] != "-":
                continue
            k, q = suffix[:i], suffix[i + 1 :]
            return f"{layer}-{head}-{q}-{k}"
        return f"{layer}-{head}-{suffix}"
    if rounder_key == "vocab_bias":
        return "lm_head-bias"
    return rounder_key


def resolve_round_primitive_matrix(run_dir: Path, rounder_key: str) -> Optional[Path]:
    """Resolve the saved primitive-matrix PNG for a round-fallback interaction."""
    base = _heatmap_root(run_dir)
    if base is None:
        return None

    attn_match = _ROUNDER_KEY_ATTN.match(rounder_key)
    if attn_match:
        layer, head, suffix = attn_match.groups()
        matrices_dir = base / f"{layer}-{head}" / "primitives-matrices"
        if not matrices_dir.exists():
            return None
        if suffix.endswith("-None"):
            path = matrices_dir / f"bias-{suffix[:-5]}.png"
            return path if path.exists() else None
        for png in sorted(matrices_dir.glob("*.png")):
            if _k_q_suffix_matches_save_name(suffix, png.stem):
                return png
        return None

    matrices_dir = base / "lm_head" / "primitives-matrices"
    if not matrices_dir.exists():
        return None
    save_name = "bias" if rounder_key == "vocab_bias" else rounder_key
    path = matrices_dir / f"{save_name}.png"
    return path if path.exists() else None


def collect_round_heatmaps(
    run_dir: Path,
    round_acceptance: list[dict],
) -> list[tuple[str, Path]]:
    """Collect primitive-matrix heatmaps for round-fallback interactions."""
    results: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for rec in round_acceptance:
        rounder_key = rec.get("rounder_key")
        if not rounder_key or rounder_key in seen:
            continue
        seen.add(rounder_key)
        path = resolve_round_primitive_matrix(run_dir, str(rounder_key))
        if path is None:
            continue
        accepted = bool(rec.get("accepted"))
        icon = "✅" if accepted else "❌"
        nz = int(rec.get("non_zero", 0) or 0)
        tp = int(rec.get("total_params", 0) or 0)
        label = f"{icon} {_format_rounder_key_display(str(rounder_key))} (non-zero: {nz}/{tp})"
        results.append((label, path))
    return results


def collect_interaction_heatmaps(run_dir: Path, info: dict) -> list[tuple[str, Path]]:
    base = _heatmap_root(run_dir)
    save_name = _interaction_save_name(info)
    if base is None or save_name is None:
        return []

    if info.get("phase") == "lm_head":
        prefix = base / "lm_head"
    else:
        layer = info.get("layer")
        head = info.get("head")
        if layer is None or head is None:
            return []
        prefix = base / f"{layer}-{head}"

    candidates: list[tuple[str, str]] = []
    if info.get("found"):
        candidates.extend(
            [
                ("Primitive matrix", f"primitives-matrices/{save_name}.png"),
                ("Primitive example", f"primitives-example/{save_name}.png"),
            ]
        )
    else:
        candidates.append(("Original example", f"original-example/{save_name}.png"))

    if info.get("found"):
        candidates.append(("Original matrix", f"original-matrices/{save_name}.png"))

    results: list[tuple[str, Path]] = []
    for label, rel_path in candidates:
        path = prefix / rel_path
        if path.exists():
            results.append((label, path))
    return results
