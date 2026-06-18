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
        self.range_min, self.range_max = length_range
        self.range_min = max(2, self.range_min)
        self.max_test_length = max_test_length
        self.bce = False
        
        assert (max_test_length >= self.range_max) or (max_test_length == -1)  # pos embedding is initialized based on max_test_length  
    
    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                instance = random.choices(range(len(self.tokenizer) - 4), k=length)
                most_common = Counter(instance).most_common(2)
                if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
                    break
                
            ans = most_common[0][0]
            
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