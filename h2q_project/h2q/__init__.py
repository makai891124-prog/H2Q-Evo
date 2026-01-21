"""H2Q Core Library - Quaternion-Fractal Self-Improving Framework.

Public API:
- tokenizer_simple: Lightweight text tokenizer
- decoder_simple: Token decoder with trimming utilities
- core.tpq_engine: Topological Phase Quantizer
- core.engine: LatentConfig and core utilities
- core.discrete_decision_engine: DDE factory and config
- core.guards.holomorphic_streaming_middleware: Veracity enforcement
"""

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