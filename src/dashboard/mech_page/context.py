from __future__ import annotations

from dataclasses import dataclass

from data.CountDataset import CountCorruption
from utilities.core import TaskConfig


@dataclass
class MechPageContext:
    run_name: str
    model_path: str | None
    task_cfg: TaskConfig | None
    raw_config: dict
    corruption: CountCorruption | None
    device: str
    compat: bool
