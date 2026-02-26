import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset, IterableDataset
from typing import Tuple

from model.CustomTokenizer import CustomTokenizer

class AdditionDataset(IterableDataset):
    """
    Custom dataset for the Addition task as defined by the paper
    """
    
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        length_range: Tuple[int, int],
        max_test_length: int
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.range_min, self.range_max = length_range
        self.range_min = max(4, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)  # pos embedding is initialized based on max_test_length
    
    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            
            len_operand1 = random.randint(1, length-2)
            len_operand2 = length - 1 - len_operand1
            
            if len_operand1 > 1:
                operand1 = ["1"] + random.choices(["0", "1"], k=len_operand1 - 1)
            else:
                operand1 = random.choices(["0", "1"], k=1)
                
            if len_operand2 > 1:
                operand2 = ["1"] + random.choices(["0", "1"], k=len_operand2 - 1)
            else:
                operand2 = random.choices(["0", "1"], k=1)
                
            ans = int(f"0b{''.join(operand1)}", 2) + int(f"0b{''.join(operand2)}", 2)
            ans = list(bin(ans)[2:])
            
            instance = [self.tokenizer.bos_token]
            instance.extend(operand1)
            instance.append("+")
            instance.extend(operand2)
            instance.append(self.tokenizer.sep_token)
            instance.extend(ans)
            instance.append(self.tokenizer.eos_token)
            
            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))
            label = deepcopy(instance)
            # Setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length + 2] = [self.tokenizer.pad_token_id,] * (length + 2) # BOS + bits.. + SEP
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length + 1)*2 - len(instance))
            else:
                offset = 0
            
            pos_ids = list(range(offset, offset + len(instance)))
            
            yield instance, pos_ids, label
            