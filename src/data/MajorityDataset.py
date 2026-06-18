import random
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
from torch.utils.data import IterableDataset
from typing import Tuple

from data.CustomTokenizer import CustomTokenizer

class MajorityDataset(IterableDataset):
    
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        length_range: Tuple[int, int],
        max_test_length: int
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        assert len(tokenizer) == 6
        
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        self.bce = False
        
        assert (max_test_length >= self.range_max) or (max_test_length == -1)  # pos embedding is initialized based on max_test_length  
    
    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                num_zero = random.randint(0, length)
                if num_zero != length - num_zero:
                    break
            
            instance = [0, ] * num_zero + [1, ] * (length - num_zero)
            random.shuffle(instance)
            ans = 0 if num_zero > length - num_zero else 1
            
            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            
            label = deepcopy(instance)
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length + 2)
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            
            pos_ids = list(range(offset, len(instance) + offset))
            yield instance, pos_ids, label