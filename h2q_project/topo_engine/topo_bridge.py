"""
topo_bridge.py — Python ctypes bridge for the C Topological Pointer-Reuse Engine.

Provides a clean Python API over the C shared library, enabling
benchmark orchestration and integration with the H2Q-Evo framework.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_int,
    c_uint16,
    c_uint32,
    c_uint64,
)
from pathlib import Path
from typing import Optional


# ─── C structure mirrors ───

class BenchmarkResult(Structure):
    _fields_ = [
        ("num_nodes",          c_uint64),
        ("num_edges",          c_uint64),
        ("build_time_us",      c_double),
        ("propagate_time_us",  c_double),
        ("morphism_count",     c_uint64),
        ("truncation_events",  c_uint64),
        ("collision_events",   c_uint64),
        ("queue_overflow_events", c_uint64),
        ("max_step",           c_uint32),
        ("memory_bytes",       c_double),
        ("ops_per_second",     c_double),
    ]

    def to_dict(self) -> dict:
        return {f[0]: getattr(self, f[0]) for f in self._fields_}


class PropagationStats(Structure):
    _fields_ = [
        ("morphism_count",     c_uint64),
        ("nodes_visited",      c_uint64),
        ("truncation_events",  c_uint64),
        ("collision_events",   c_uint64),
        ("queue_overflow_events", c_uint64),
        ("max_step_reached",   c_uint32),
        ("max_precision_seen", c_uint32),
        ("elapsed_us",         c_double),
    ]

    def to_dict(self) -> dict:
        return {f[0]: getattr(self, f[0]) for f in self._fields_}


# ─── Library loader ───

_LIB: Optional[ctypes.CDLL] = None
_LIB_DIR = Path(__file__).resolve().parent


def _build_library() -> Path:
    """Build the shared library if it does not exist."""
    so_path = _LIB_DIR / "libtopo_engine.so"
    if so_path.exists():
        return so_path

    print("[topo_bridge] Building libtopo_engine.so ...")
    result = subprocess.run(
        ["make", "-C", str(_LIB_DIR), "all"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build topo_engine:\n{result.stderr}\n{result.stdout}"
        )
    if not so_path.exists():
        raise FileNotFoundError(f"Build succeeded but {so_path} not found")

    print(f"[topo_bridge] Built {so_path}")
    return so_path


def get_lib() -> ctypes.CDLL:
    """Load (and build if needed) the C shared library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    so_path = _build_library()
    _LIB = ctypes.CDLL(str(so_path))

    # ── Bind signatures ──
    _LIB.topo_run_benchmark.argtypes = [c_uint64, c_uint16, c_int, c_uint32]
    _LIB.topo_run_benchmark.restype  = BenchmarkResult

    _LIB.calculate_taylor_decay.argtypes = [c_int, c_int]
    _LIB.calculate_taylor_decay.restype  = c_double

    return _LIB


# ─── High-level Python API ───

def run_benchmark(num_nodes: int, avg_edges: int = 4,
                  max_steps: int = 10, seed: int = 42) -> dict:
    """
    Run a full build-propagate-measure cycle in the C engine.

    Returns a dict with all benchmark metrics.
    """
    lib = get_lib()
    result = lib.topo_run_benchmark(
        c_uint64(num_nodes),
        c_uint16(avg_edges),
        c_int(max_steps),
        c_uint32(seed),
    )
    return result.to_dict()


def taylor_decay(step: int, precision: int) -> float:
    """Compute Taylor decay value for a given step and precision."""
    lib = get_lib()
    return lib.calculate_taylor_decay(c_int(step), c_int(precision))


def run_scaling_benchmark(sizes: list[int], avg_edges: int = 4,
                          max_steps: int = 10, seed: int = 42) -> list[dict]:
    """Run benchmarks at multiple network sizes for scaling analysis."""
    results = []
    for n in sizes:
        r = run_benchmark(n, avg_edges, max_steps, seed)
        results.append(r)
    return results
