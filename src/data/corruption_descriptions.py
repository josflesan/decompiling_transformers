"""Human-readable corruption copy for dashboards (Markdown per task and corruption name).

Layout: ``src/data/corruptions/<task_name>/<CORRUPTION_ENUM_NAME>.md``
(e.g. ``corruptions/counting/CHANGE_START.md`` for the counting task's ``CountCorruption``).
"""

from __future__ import annotations

from pathlib import Path

_CORRUPTIONS_ROOT = Path(__file__).resolve().parent / "corruptions"


def corruption_description_path(task_name: str, corruption_name: str) -> Path:
    return _CORRUPTIONS_ROOT / task_name / f"{corruption_name}.md"


def load_corruption_description(task_name: str, corruption_name: str) -> str:
    """Return Markdown for sidebar / docs; missing files get a short placeholder."""
    path = corruption_description_path(task_name, corruption_name)
    if not path.is_file():
        return (
            f"No description file for task `{task_name}` / corruption `{corruption_name}`. "
            f"Add `data/corruptions/{task_name}/{corruption_name}.md`."
        )
    return path.read_text(encoding="utf-8").strip()
