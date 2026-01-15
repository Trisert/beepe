"""beepe: Fast BPE tokenizer in Rust with Python bindings"""

__version__ = "0.1.0"

try:
    from .beepe import *
except ImportError:
    pass
