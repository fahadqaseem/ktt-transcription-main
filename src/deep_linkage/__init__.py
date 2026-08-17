"""
deep_linkage
~~~~~~~~~~~~

A deep-learning and vector-similarity Entity Resolution pipeline designed for
longitudinal historical school ledger and census transcription projects.
"""

from .config import DeepERConfig
from .engine import DeepLinkageEngine
from .schema import flatten_record
from .trainer import DeepLinkageTrainer

__version__ = "0.1.0"

# Explicitly declare public API to prevent namespace pollution
__all__ = [
    "DeepERConfig",
    "DeepLinkageTrainer",
    "DeepLinkageEngine",
    "flatten_record",
    "__version__",
]