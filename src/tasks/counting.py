from tasks.base import Task
from tasks.registry import register_task

from data.CountDataset import CountDataset
from data.CustomTokenizer import CustomTokenizer

@register_task("counting")
class CountingTask(Task):
    
    def build_tokenizer(self):
        vocab = [str(i) for i in range(self.config.max_test_length)]
        tokenizer = CustomTokenizer(vocab)
        
        return tokenizer
    
    def build_dataset(self, tokenizer):
        train_dataset = CountDataset(
            tokenizer=tokenizer,
            length_range=self.config.train_length_range,
            max_test_length=self.config.max_test_length
        )
        
        # val_dataset = CountDataset(
        #     tokenizer=tokenizer,
        #     length_range=self.config.val_length_range,
        #     max_test_length=self.config.max_test_length
        # )
        
        return {
            "train": train_dataset,
            "val": train_dataset,
        }