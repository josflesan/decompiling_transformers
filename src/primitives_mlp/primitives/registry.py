from enum import Enum

PRIMITIVE_REGISTRY = {}

class PrimitiveType(Enum):
    EQUALS = ("equal", True)  # Name, single_input
    EXISTS = ("exists", True)
    FORALL = ("forall", True)
    HARDEN = ("harden", True)
    NOOP = ("noop", True)
    SHARPEN = ("sharpen", True)
    ZEROONE = ("zeroone", True)

def register(name):
    def wrapper(cls):
        PRIMITIVE_REGISTRY[name] = cls
        return cls
    return wrapper

def build_primitive(ptype: PrimitiveType, **kwargs):
    pname, psingle = ptype.value
    return PRIMITIVE_REGISTRY[pname](name=pname, single_input=psingle, **kwargs)