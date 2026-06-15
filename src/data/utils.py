from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float

@dataclass
class CleanCorruptData:
    corrupted_tokens: Float[Tensor, "..."]
    corrupted_pos: Float[Tensor, "..."]
    clean_tokens: Float[Tensor, "..."]
    clean_pos: Float[Tensor, "..."]
    answer_tokens: Float[Tensor, "..."]