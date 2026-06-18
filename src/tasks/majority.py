from tasks.base import Task
from tasks.registry import register_task

from data.MajorityDataset import MajorityDataset
from data.CustomTokenizer import CustomTokenizer

@register_task("majority")
class MajorityTask(Task):
    
    def build_tokenizer(self):
        vocab = ["0", "1"]
        tokenizer = CustomTokenizer(vocab)
        
        return tokenizer
    
    def build_dataset(self, tokenizer):
        train_dataset = MajorityDataset(
            tokenizer=tokenizer,
            length_range=self.config.train_length_range,
            max_test_length=self.config.max_test_length
        )
        
        return {
            "train": train_dataset,
            "val": train_dataset,
        }