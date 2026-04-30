import random
import torch
from copy import deepcopy
from torch.utils.data import IterableDataset
from typing import Tuple

from data.CustomTokenizer import CustomTokenizer

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
        
    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            vocab_size = len(self.tokenizer) - 4
            start = random.randint(0, vocab_size - length)
            end = start + length - 1
            
            instance = [self.tokenizer.bos_token]
            instance.append(str(start))
            instance.append(str(end))
            instance.append(self.tokenizer.sep_token)
            instance.extend([str(i) for i in range(start, end+1)])
            instance.append(self.tokenizer.eos_token)
            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))
            
            label = deepcopy(instance)
            label[:4] = [self.tokenizer.pad_token_id,] * 4  # bos + ... + sep
            
            if self.max_test_length != -1:
                # Guard against invalid ranges when generated sequences exceed max_test_length.
                max_offset = self.max_test_length - len(instance)
                offset = random.randint(0, max_offset) if max_offset >= 0 else 0
            else:
                offset = 0
                
            pos_ids = list(range(offset, len(instance) + offset))
            
            yield instance, pos_ids, label