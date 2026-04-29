"""This file contains a range of basic utilities used throughout the program"""

from dataclasses import dataclass
from typing import List

def int_key_hook(d):
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}

@dataclass
class TaskConfig:
    name: str
    train_length_range: List[int]
    val_length_range: List[int]
    max_test_length: int