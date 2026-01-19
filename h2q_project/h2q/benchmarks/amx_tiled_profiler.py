import torch
import time
import numpy as np
from typing import Dict, Any
from h2q.dispatch.amx_tiling_dispatcher import M4RegisterTelemetry, DynamicAMXTilingDispatcher
from h2q.core.ops.hamilton_amx import HamiltonOptimizer
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.engine import DiscreteDecisionEngine

class AMXTiledProfiler:
    """
    Performance Profiler for M4 Silicon AMX-Tiled Hamilton Products.
    Audits 16x16 register pressure and verifies the 10x throughput target.
    """
    def __init__(self, manifold_dim: int = 256):
        self.manifold_dim = manifold_dim
        # Using canonical getter to avoid 'dim' keyword argument error identified in feedback
        self.dde = get_canonical_dde()
        self.telemetry = M4RegisterTelemetry()
        self.dispatcher = DynamicAMXTilingDispatcher()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _generate_quaternionic_batch(self, batch_size: int):
        """Generates a batch of quaternions (Batch, 4, Dim/4)."""
        return torch.randn(batch_size, 4, self.manifold_dim // 4, device=self.device)

    def audit_register_pressure(self, depth: int = 12) -> Dict[str, Any]:
        """
        Audits 16x16 register pressure during deep fractal expansions.
        Fractal expansion involves recursive Hamilton rotations.
        """
        q_state = self._generate_quaternionic_batch(1)
        q_rot = self._generate_quaternionic_batch(1)
        
        pressure_logs = []
        
        print(f"[PROFILER] Starting Fractal Audit: Depth={depth}, TileSize=16x16")
        
        for d in range(depth):
            # Simulate AMX Tiling Dispatch
            tile_config = self.dispatcher.compute_optimal_tiling(q_state.shape, tile_size=16)
            
            # Record telemetry before operation
            self.telemetry.mark_start(stage=f"fractal_depth_{d}")
            
            # Perform Hamilton Product (Simulated AMX path)
            # In a real M4 environment, this triggers the AMX coprocessor
            with torch.no_grad():
                # Simplified Hamilton Product for profiling
                q_state = torch.matmul(q_state, q_rot.transpose(-1, -2))
            
            # Capture register pressure metrics
            stats = self.telemetry.capture_metrics()
            pressure_logs.append({
                "depth": d,
                "register_utilization": stats.get("utilization", 0.0),
                "tile_collisions": stats.get("collisions", 0),
                "l1_cache_misses": stats.get("cache_miss", 0)
            })
            
        return {
            "depth_profile": pressure_logs,
            "peak_pressure": max(p["register_utilization"] for p in pressure_logs)
        }

    def verify_throughput_target(self, batch_size: int = 1024, iterations: int = 100) -> Dict[str, float]:
        """
        Verifies if the AMX-optimized Hamilton product hits the 10x throughput target
        compared to a naive Euclidean baseline.
        """
        q1 = self._generate_quaternionic_batch(batch_size)
        q2 = self._generate_quaternionic_batch(batch_size)

        # Baseline: Naive Euclidean Translation (Standard MatMul)
        torch.mps.synchronize() if self.device.type == "mps" else None
        start_naive = time.perf_counter()
        for _ in range(iterations):
            _ = torch.matmul(q1, q2.transpose(-1, -2))
        torch.mps.synchronize() if self.device.type == "mps" else None
        end_naive = time.perf_counter()
        naive_time = (end_naive - start_naive) / iterations

        # Target: AMX-Tiled Hamilton Product
        # We use the HamiltonOptimizer which leverages the AMX dispatch logic
        optimizer = HamiltonOptimizer(engine=self.dde)
        
        torch.mps.synchronize() if self.device.type == "mps" else None
        start_amx = time.perf_counter()
        for _ in range(iterations):
            _ = optimizer.apply_hamilton_product(q1, q2)
        torch.mps.synchronize() if self.device.type == "mps" else None
        end_amx = time.perf_counter()
        amx_time = (end_amx - start_amx) / iterations

        speedup = naive_time / amx_time
        
        return {
            "naive_latency_ms": naive_time * 1000,
            "amx_latency_ms": amx_time * 1000,
            "speedup_factor": speedup,
            "target_met": speedup >= 10.0
        }

if __name__ == "__main__":
    profiler = AMXTiledProfiler(manifold_dim=256)
    
    print("--- AMX REGISTER PRESSURE AUDIT ---")
    pressure_results = profiler.audit_register_pressure(depth=16)
    print(f"Peak Register Pressure: {pressure_results['peak_pressure']:.2%}")
    
    print("\n--- THROUGHPUT VERIFICATION (10X TARGET) ---")
    throughput_results = profiler.verify_throughput_target()
    print(f"Naive Latency: {throughput_results['naive_latency_ms']:.4f} ms")
    print(f"AMX Latency:   {throughput_results['amx_latency_ms']:.4f} ms")
    print(f"Speedup:       {throughput_results['speedup_factor']:.2f}x")
    print(f"Status:        {'PASSED' if throughput_results['target_met'] else 'FAILED'}")
