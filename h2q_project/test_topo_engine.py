"""
test_topo_engine.py — Unit tests for the Topological Pointer-Reuse Engine.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "topo_engine"))
from topo_bridge import get_lib, run_benchmark, taylor_decay  # noqa: E402


class TestTaylorDecay:
    """Verify Taylor decay = 1/s! is computed correctly."""

    @pytest.mark.parametrize("step", range(13))
    def test_decay_matches_factorial(self, step):
        computed = taylor_decay(step, 4)
        expected = 1.0 / math.factorial(step)
        assert abs(computed - expected) < 1e-12

    def test_high_step_returns_zero(self):
        assert taylor_decay(21, 4) == 0.0


class TestTruncationThreshold:
    """Truncation should fire at progressively later steps for higher precision."""

    def test_higher_precision_delays_cutoff(self):
        cutoffs = []
        for precision in [2, 4, 6, 8]:
            threshold = 1.0 / (2 ** precision)
            for step in range(21):
                if taylor_decay(step, precision) < threshold:
                    cutoffs.append(step)
                    break
        # Higher precision → later cutoff step
        assert cutoffs == sorted(cutoffs)
        assert len(cutoffs) == 4


class TestSmallNetwork:
    """Basic invariants on a small network."""

    def test_propagation_produces_morphisms(self):
        r = run_benchmark(num_nodes=100, avg_edges=3, max_steps=8, seed=1)
        assert r["num_nodes"] == 100
        assert r["morphism_count"] > 0

    def test_morphisms_bounded_by_edges(self):
        r = run_benchmark(num_nodes=100, avg_edges=3, max_steps=8, seed=1)
        assert r["morphism_count"] <= r["num_edges"]


class TestScaling:
    """Morphism count must be sub-quadratic in N."""

    @pytest.mark.parametrize("n", [500, 1000, 5000, 10000])
    def test_sub_quadratic(self, n):
        r = run_benchmark(n, avg_edges=4, max_steps=10, seed=42)
        ratio = r["morphism_count"] / (n * n)
        assert ratio < 0.1, f"ratio={ratio:.6f} is not sub-quadratic"


class TestMemoryEfficiency:
    """Topology memory should be ≪ dense N×N float32 matrix."""

    def test_memory_much_smaller_than_dense(self):
        n = 10000
        r = run_benchmark(n, avg_edges=4, max_steps=10, seed=42)
        dense_bytes = n * n * 4  # float32
        assert r["memory_bytes"] < dense_bytes * 0.01


class TestCollisionDetection:
    """Collisions should be detected during propagation."""

    def test_collisions_detected(self):
        r = run_benchmark(1000, avg_edges=4, max_steps=8, seed=6)
        assert r["collision_events"] > 0


class TestThroughput:
    """Engine should sustain > 1M morphism ops/sec."""

    def test_high_throughput(self):
        r = run_benchmark(10000, avg_edges=4, max_steps=10, seed=42)
        assert r["ops_per_second"] > 1_000_000
