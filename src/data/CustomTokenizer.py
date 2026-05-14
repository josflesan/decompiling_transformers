import torch

from typing import List

class CustomTokenizer():
    def __init__(self, vocab: List[str]):
        normal_tkn_num = len(vocab) # Each element is a token
        
        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # Define the vocabulary mapping
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"
    
    def __call__(self, strings: List[str] | str, **kwargs):
        if type(strings) == str:
            strings = [strings]
            
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append(list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len - len(s)))
        
        return {"input_ids": torch.LongTensor(ids)}
    
    def convert_ids_to_tokens(self, ids: List[int] | torch.Tensor, rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            
            if not isinstance(ids, list):
                ids = ids.tolist()
            
            return list(map(lambda x: self.vocab_inv[x], ids))
    
    def __len__(self):
        return len(self.vocab)