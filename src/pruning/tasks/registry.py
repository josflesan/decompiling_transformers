TASK_REGISTRY = {}

def register_task(name):
    def wrapper(cls):
        TASK_REGISTRY[name] = cls
        return cls
    
    return wrapper

def get_task(name, config):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}")
    
    return TASK_REGISTRY[name](config)