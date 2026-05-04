import logging
from pathlib import Path

def setup_logger(log_dir, name="pruning"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove previous metrics
    (log_dir / 'run.log').unlink(missing_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File
    file_handler = logging.FileHandler(log_dir / 'run.log')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger