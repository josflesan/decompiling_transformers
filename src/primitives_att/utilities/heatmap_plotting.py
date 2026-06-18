"""
Save primitive replacement matrices as heatmap images.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def _simplify_product_labels(labels: Optional[Sequence[str]]) -> Optional[List[str]]:
    if labels is None or len(labels) == 0:
        return None if labels is None else []

    sample_splits = [str(label).split("-") for label in labels[: min(10, len(labels))]]
    has_separator = all(len(parts) > 1 for parts in sample_splits)
    if not has_separator:
        return list(labels)

    num_parts = [len(parts) for parts in sample_splits]
    if not all(n == num_parts[0] for n in num_parts):
        return list(labels)

    simplified: list[str] = []
    last_part = None
    for label in labels:
        first_part = str(label).split("-")[0]
        if first_part != last_part:
            last_part = first_part
            simplified.append(str(label))
        else:
            simplified.append("")
    return simplified


def plot_and_save_primitives_matrices(
    matrix: torch.Tensor,
    save_path: str | Path,
    ticks_x: Optional[Sequence[str]] = None,
    ticks_y: Optional[Sequence[str]] = None,
    add_causal_mask: bool = False,
) -> None:
    if ticks_x is not None:
        ticks_x = _simplify_product_labels(ticks_x)
    if ticks_y is not None:
        ticks_y = _simplify_product_labels(ticks_y)

    is_column = False
    
    if matrix.ndim > 2:
        matrix = matrix[0]
    if matrix.ndim < 2:
        matrix = matrix.unsqueeze(0)
        is_column = True

    matrix_np = matrix.detach().cpu().numpy()
    num_x_ticks = matrix_np.shape[1] if ticks_x is None else len(ticks_x)
    num_y_ticks = matrix_np.shape[0] if ticks_y is None else len(ticks_y)

    if is_column or matrix_np.shape[0] == 1:
        width = max(6, min(num_x_ticks * 0.4, 20))
        figsize = (width, 2.5)
    elif matrix_np.shape[1] == 1:
        height = max(4, min(num_y_ticks * 0.3, 16))
        figsize = (3, height)
    else:
        base_width = max(6, min(num_x_ticks * 0.25, 16))
        base_height = max(4, min(num_y_ticks * 0.25, 16))
        aspect_ratio = matrix_np.shape[1] / max(matrix_np.shape[0], 1)
        if aspect_ratio > 2:
            figsize = (base_width, base_height * 0.7)
        elif aspect_ratio < 0.5:
            figsize = (base_width * 0.7, base_height)
        else:
            figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize)
    base_cmap = plt.cm.get_cmap("Blues")
    custom_cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    im = ax.imshow(matrix_np, cmap=custom_cmap, aspect="auto")

    if add_causal_mask and matrix_np.shape[0] == matrix_np.shape[1]:
        mask = np.triu(np.ones_like(matrix_np), k=1).astype(float)
        mask[mask == 0] = np.nan
        ax.imshow(mask, cmap="Greys", alpha=1.0, aspect="auto", vmin=0, vmax=2)

    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    if matrix_np.shape[1] > 1 and ticks_x is not None:
        rotation = 0 if num_x_ticks <= 3 else 45 if num_x_ticks <= 8 else 60
        ax.set_xticks(range(num_x_ticks))
        ax.set_xticklabels(
            [ticks_x[i] for i in range(num_x_ticks)],
            fontsize=10,
            rotation=rotation,
            ha="right" if rotation > 0 else "center",
        )

    if matrix_np.shape[0] > 1 and not is_column and ticks_y is not None:
        ax.set_yticks(range(num_y_ticks))
        ax.set_yticklabels([ticks_y[i] for i in range(num_y_ticks)], fontsize=10)

    if is_column and ticks_y is None:
        ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
