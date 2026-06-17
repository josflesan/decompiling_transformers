"""Helpers for locating cached LogitLens entries in mlp_input_output.pt."""

from __future__ import annotations

from typing import Any


def matching_cache_keys_for_path(path: str, mlp_input_output: dict) -> list[Any]:
    """Return keys in mlp_input_output.pt that correspond to an MLP activation path."""
    if not mlp_input_output:
        return []

    matches: list[Any] = []
    if path in mlp_input_output:
        matches.append(path)

    for key in mlp_input_output:
        if key in matches:
            continue
        if isinstance(key, str) and key.endswith(path):
            matches.append(key)
        elif (
            isinstance(key, tuple)
            and len(key) == 4
            and isinstance(key[0], str)
            and isinstance(key[1], str)
            and (key[0].endswith(path) or key[1].endswith(path))
        ):
            matches.append(key)
    return matches


def has_logit_lens_cache_entry(path: str, mlp_input_output: dict | None) -> bool:
    """Whether LogitLens input/output tensors were cached for this MLP path."""
    return bool(matching_cache_keys_for_path(path, mlp_input_output or {}))
