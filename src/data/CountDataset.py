import random
import torch
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from torch import Tensor
from torch.utils.data import IterableDataset
from typing import Tuple, List, Optional
from jaxtyping import Float

from data.CustomTokenizer import CustomTokenizer
from data.utils import CleanCorruptData

@dataclass(frozen=True)
class CountSample:
    start: int
    length: int
    offset: int
    copy: bool = False

class CountCorruption(Enum):
    """
    Counterfactual / paired-prompt corruptions for `CountDataset.get_corrupted`.

    Counting-task corruptions; sidebar copy lives under `src/data/corruptions/counting/<ENUM_NAME>.md`
    (see `corruption_descriptions.py`).
    """
    CHANGE_START = 1

class CountDataset(IterableDataset):
    
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        length_range: Tuple[int, int],
        max_test_length: int
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.range_min, self.range_max = length_range
        self.range_min = max(2, self.range_min)
        self.max_test_length = max_test_length
        self.bce = False
        
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)  # pos embedding is initialized based on max_test_length
    
    def _sample_params(self) -> CountSample:
        length = random.randint(self.range_min, self.range_max)
        vocab_size = len(self.tokenizer) - 4
        start = random.randint(0, vocab_size - length)
        
        rendered_len = length + 5  # BOS + start + end + SEP + body(length) + EOS
        if self.max_test_length != -1:
            max_offset = self.max_test_length - rendered_len
            offset = random.randint(0, max_offset) if max_offset >= 0 else 0
        else:
            offset = 0
        
        return CountSample(
            start=start, length=length, offset=offset
        )
    
    def _render(self, sample: CountSample) -> Tuple[List[int], List[int], List[int]]:
        end = sample.start + sample.length - 1
        tokens = [self.tokenizer.bos_token, str(sample.start), str(end), self.tokenizer.sep_token]
        
        if sample.copy:
            tokens.extend(str(sample.start) for i in range(sample.start, end + 1))
        else:
            tokens.extend(str(i) for i in range(sample.start, end + 1))
        
        tokens.append(self.tokenizer.eos_token)
        instance = [self.tokenizer.vocab[t] for t in tokens]
        
        label = deepcopy(instance)
        label[:4] = [self.tokenizer.pad_token_id] * 4
        
        pos_ids = list(range(sample.offset, sample.offset + len(instance)))
        return instance, pos_ids, label

    def _tokenize_prompts(self, prompt: str) -> Tuple[Float[Tensor, "1 seq"], Float[Tensor, "1 seq"]]:
        tokens = self.tokenizer(prompt)['input_ids']
        max_offset = self.max_test_length - len(tokens)
        offset = random.randint(0, max_offset) if max_offset >= 0 else 0
        clean_pos_ids = list(range(offset, offset + len(tokens[0])))
        position_ids = torch.tensor(clean_pos_ids).unsqueeze(0)
        
        return tokens, position_ids

    def get_corrupted(self, corruption: CountCorruption, batch_size=25) -> CleanCorruptData:        
        corrupted_tokens = []
        corrupted_pos = []
        clean_tokens = []
        clean_pos = []
        answer_tokens = []

        for _ in range(batch_size):
            
            match corruption:
                
                case CountCorruption.CHANGE_START:                    
                    start_clean = random.randint(1, 15)
                    start_corrupt = random.randint(1, 15)
                    end = max(start_clean, start_corrupt) + random.randint(1, 5)
                    clean_prompt = (
                        f"{self.tokenizer.bos_token} "
                        f"{start_clean} {end} "
                        f"{self.tokenizer.sep_token}"
                    )

                    corrupt_prompt = (
                        f"{self.tokenizer.bos_token} "
                        f"{start_corrupt} {end} "
                        f"{self.tokenizer.sep_token}"
                    )
            
            clean_tok, clean_pos_tok = self._tokenize_prompts(clean_prompt)
            corrupt_tok, corrupt_pos_tok = self._tokenize_prompts(corrupt_prompt)

            answer = torch.cat([
                self.tokenizer(str(start_clean), return_tensors="pt")["input_ids"],
                self.tokenizer(str(start_corrupt), return_tensors="pt")["input_ids"]
            ], dim=1).squeeze(0)

            clean_tokens.append(clean_tok.squeeze(0))
            clean_pos.append(clean_pos_tok.squeeze(0))
            corrupted_tokens.append(corrupt_tok.squeeze(0))
            corrupted_pos.append(corrupt_pos_tok.squeeze(0))
            answer_tokens.append(answer.squeeze(0))
        
        return CleanCorruptData(
            corrupted_tokens=torch.stack(corrupted_tokens),
            corrupted_pos=torch.stack(corrupted_pos),
            clean_tokens=torch.stack(clean_tokens),
            clean_pos=torch.stack(clean_pos),
            answer_tokens=torch.stack(answer_tokens),
        )
        
    
    def __iter__(self):
        while True:
            yield self._render(self._sample_params())