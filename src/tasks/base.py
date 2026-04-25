from abc import ABC, abstractmethod

class Task(ABC):
    
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def build_tokenizer(self):
        pass
    
    @abstractmethod
    def build_dataset(self, tokenizer):
        pass
    
    def build(self):
        tokenizer = self.build_tokenizer()
        dataset = self.build_dataset(tokenizer)
        return tokenizer, dataset