import torch

class CustomCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id
    
    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)
        
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -1000
        [item.extend([item[-1],] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)
        
        batch = {
            "input_ids": input_ids,
            "position_ids": pos_ids,
            "labels": labels
        }
        return batch