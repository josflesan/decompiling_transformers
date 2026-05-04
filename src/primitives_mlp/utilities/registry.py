PRIMITIVE_REGISTRY = {}

def register(name):
    def wrapper(cls):
        PRIMITIVE_REGISTRY[name] = cls
        return cls
    return wrapper
