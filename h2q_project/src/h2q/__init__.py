"""
H2Q Core Library - Quaternion-Fractal Self-Improving Framework

This module contains the core mathematical and algorithmic components
of the H2Q-Evo system, including unified architectures, spectral trackers,
and decision engines.
"""

__version__ = "2.3.0"

# Core components
from .core.unified_architecture import UnifiedH2QMathematicalArchitecture
from .core.sst import SpectralShiftTracker
from .core.discrete_decision_engine import get_canonical_dde, LatentConfig

__all__ = [
    'UnifiedH2QMathematicalArchitecture',
    'SpectralShiftTracker',
    'get_canonical_dde',
    'LatentConfig'
]

__version__ = "2.3.1"

# Lazy imports to avoid circular dependencies
__all__ = [
    "default_tokenizer",
    "default_decoder",
    "SimpleTokenizer",
    "SimpleDecoder",
]


def __getattr__(name: str):
    if name == "default_tokenizer":
        from h2q.tokenizer_simple import default_tokenizer
        return default_tokenizer
    if name == "default_decoder":
        from h2q.decoder_simple import default_decoder
        return default_decoder
    if name == "SimpleTokenizer":
        from h2q.tokenizer_simple import SimpleTokenizer
        return SimpleTokenizer
    if name == "SimpleDecoder":
        from h2q.decoder_simple import SimpleDecoder
        return SimpleDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")