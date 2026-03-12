import json
from pathlib import Path

class MetricsLogger:
    
    def __init__(self, log_dir):
        self.path = Path(log_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, **metrics):
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")