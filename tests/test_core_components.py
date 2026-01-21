"""Unit tests for core H2Q components.

Tests cover:
- TPQ Engine: quantization cycle, metrics tracking, numerical stability
- Tokenizer: encode/decode roundtrip, batch operations, special tokens
- Decoder: trim operations, batch decode
"""

import pytest
import torch
import sys
import os

# Ensure h2q_project is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTPQEngine:
    """Tests for TopologicalPhaseQuantizer."""

    def test_quantize_dequantize_roundtrip(self):
        from h2q_project.h2q.core.tpq_engine import TopologicalPhaseQuantizer

        tpq = TopologicalPhaseQuantizer(bits=8)
        # Random unit quaternions
        q = torch.randn(4, 4)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        quantized = tpq.quantize(q, update_metrics=False)
        reconstructed = tpq.dequantize(quantized)

        # Check shapes
        assert quantized.shape == (4, 3)
        assert quantized.dtype == torch.uint8
        assert reconstructed.shape == (4, 4)

        # Reconstruction should be close (within quantization error)
        error = torch.norm(q - reconstructed, dim=-1).max().item()
        assert error < 0.1, f"Reconstruction error too high: {error}"

    def test_spectral_shift_range(self):
        from h2q_project.h2q.core.tpq_engine import TopologicalPhaseQuantizer

        tpq = TopologicalPhaseQuantizer(bits=8)
        q_orig = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Unit quaternion
        q_recon = torch.tensor([[0.99, 0.1, 0.0, 0.0]])
        q_recon = torch.nn.functional.normalize(q_recon, dim=-1)

        eta = tpq.get_spectral_shift(q_orig, q_recon)
        assert 0 <= eta.item() <= 1, f"Spectral shift out of range: {eta.item()}"

    def test_metrics_tracking(self):
        from h2q_project.h2q.core.tpq_engine import TopologicalPhaseQuantizer

        tpq = TopologicalPhaseQuantizer(bits=8)
        tpq.reset_metrics()

        q = torch.randn(10, 4)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        tpq.quantize(q, update_metrics=True)

        metrics = tpq.metrics
        assert metrics.num_samples == 10
        assert metrics.compression_ratio == 4.0  # 32/8
        assert metrics.spectral_shift_mean >= 0

    def test_batch_processing(self):
        from h2q_project.h2q.core.tpq_engine import TopologicalPhaseQuantizer

        tpq = TopologicalPhaseQuantizer(bits=8)
        # Batch of quaternions with extra dimensions
        q = torch.randn(2, 8, 4)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        quantized, reconstructed = tpq.forward(q)
        assert quantized.shape == (2, 8, 3)
        assert reconstructed.shape == (2, 8, 4)


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""

    def test_encode_decode_roundtrip(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        text = "Hello, H2Q!"
        ids = tok.encode(text, add_specials=False, max_length=0, padding=False)
        decoded = tok.decode(ids, skip_specials=False)
        assert decoded == text

    def test_special_tokens(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        text = "test"
        ids = tok.encode(text, add_specials=True, max_length=0, padding=False)
        
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert len(ids) == len(text) + 2  # bos + text + eos

    def test_padding(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        ids = tok.encode("hi", max_length=10, padding=True)
        
        assert len(ids) == 10
        assert ids[-1] == tok.pad_id

    def test_batch_encode(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        texts = ["hello", "world", "!"]
        batch_ids = tok.encode_batch(texts, max_length=16)
        
        assert len(batch_ids) == 3
        assert all(len(ids) == 16 for ids in batch_ids)

    def test_vocab_size(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        assert tok.vocab_size == 4 + 95  # 4 specials + 95 printable ASCII

    def test_unknown_token(self):
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        # Non-ASCII character should map to unk
        ids = tok.encode("你好", add_specials=False, padding=False, max_length=0)
        assert all(i == tok.unk_id for i in ids)


class TestSimpleDecoder:
    """Tests for SimpleDecoder."""

    def test_trim_at_eos(self):
        from h2q_project.h2q.decoder_simple import SimpleDecoder
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        dec = SimpleDecoder(tok)

        # Sequence with EOS in middle
        ids = [tok.bos_id, 10, 11, tok.eos_id, 12, 13]
        trimmed = dec.trim_at_eos(ids)
        assert trimmed == [tok.bos_id, 10, 11]

    def test_trim_padding(self):
        from h2q_project.h2q.decoder_simple import SimpleDecoder
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        dec = SimpleDecoder(tok)

        ids = [10, 11, 12, tok.pad_id, tok.pad_id]
        trimmed = dec.trim_padding(ids)
        assert trimmed == [10, 11, 12]

    def test_batch_decode(self):
        from h2q_project.h2q.decoder_simple import SimpleDecoder
        from h2q_project.h2q.tokenizer_simple import SimpleTokenizer

        tok = SimpleTokenizer()
        dec = SimpleDecoder(tok)

        batch = tok.encode_batch(["ab", "cd"], max_length=8)
        decoded = dec.decode_batch(batch)
        
        assert decoded[0] == "ab"
        assert decoded[1] == "cd"


class TestHolomorphicMiddleware:
    """Tests for HolomorphicStreamingMiddleware."""

    def test_metrics_tracking(self):
        try:
            from h2q_project.h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
        except ImportError:
            pytest.skip("Middleware dependencies not available")

        middleware = HolomorphicStreamingMiddleware(threshold=0.05)
        middleware.reset_metrics()

        # Process some tensors
        for _ in range(5):
            x = torch.randn(1, 256)
            middleware.process_token_latent(x)

        metrics = middleware.get_metrics()
        assert metrics["total_processed"] == 5
        assert "correction_rate" in metrics
        assert "avg_curvature" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
