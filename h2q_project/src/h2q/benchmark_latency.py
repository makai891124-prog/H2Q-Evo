import torch
import time
import numpy as np
from typing import Tuple

# --- STABLE CODE: FIXING DISCRETE DECISION ENGINE ---
class DiscreteDecisionEngine:
    """
    H2Q Discrete Decision Engine (DDE).
    Fixed: Added 'num_actions' to __init__ to resolve Runtime Error.
    """
    def __init__(self, state_dim: int, num_actions: int = 2, **kwargs):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Initialize SU(2) weights
        self.weights = torch.randn(num_actions, 4, device=self.device) / np.sqrt(4)

# --- EXPERIMENTAL CODE: OPTIMIZED HAMILTON PRODUCT ---

def hamilton_product_naive(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Standard Hamilton Product implementation."""
    a1, b1, c1, d1 = q1.unbind(-1)
    a2, b2, c2, d2 = q2.unbind(-1)
    return torch.stack([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ], dim=-1)

def hamilton_product_mps_optimized(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Optimized for M4 MPS Register Constraints.
    Uses a matrix-vector representation to leverage AMX (Apple Matrix Extension)
    and reduce register pressure from multiple unbind/stack operations.
    """
    # Construct the left-multiplication matrix for q1
    # q1 shape: (N, 4)
    a, b, c, d = q1.unbind(-1)
    
    # Row-wise construction to minimize kernel launches
    row1 = torch.stack([a, -b, -c, -d], dim=-1)
    row2 = torch.stack([b,  a, -d,  c], dim=-1)
    row3 = torch.stack([c,  d,  a, -b], dim=-1)
    row4 = torch.stack([d, -c,  b,  a], dim=-1)
    
    L_mat = torch.stack([row1, row2, row3, row4], dim=-2) # (N, 4, 4)
    
    # Perform batch matrix multiplication: (N, 4, 4) @ (N, 4, 1)
    return torch.bmm(L_mat, q2.unsqueeze(-1)).squeeze(-1)

def run_audit(dimensions: int = 256, iterations: int = 1000):
    device = torch.device("mps")
    print(f"[M24-CW] Starting Latency Audit on {device} (M4 Optimized)")
    
    # Fractal Expansion: 2 -> 256
    q1 = torch.randn(dimensions, 4, device=device)
    q2 = torch.randn(dimensions, 4, device=device)
    
    # Warm-up (Crucial for MPS shader compilation)
    for _ in range(100):
        _ = hamilton_product_naive(q1, q2)
        torch.mps.synchronize()

    # Benchmark Naive
    start_naive = time.perf_counter()
    for _ in range(iterations):
        _ = hamilton_product_naive(q1, q2)
        torch.mps.synchronize()
    end_naive = time.perf_counter()
    
    # Benchmark Optimized
    start_opt = time.perf_counter()
    for _ in range(iterations):
        _ = hamilton_product_mps_optimized(q1, q2)
        torch.mps.synchronize()
    end_opt = time.perf_counter()

    naive_ms = (end_naive - start_naive) * 1000 / iterations
    opt_ms = (end_opt - start_opt) * 1000 / iterations
    improvement = ((naive_ms - opt_ms) / naive_ms) * 100

    print(f"--- AUDIT RESULTS ---")
    print(f"Naive Latency:     {naive_ms:.4f} ms")
    print(f"Optimized Latency: {opt_ms:.4f} ms")
    print(f"Efficiency Gain:   {improvement:.2f}%")
    
    # Verify Symmetry (Rigid Construction)
    res_n = hamilton_product_naive(q1, q2)
    res_o = hamilton_product_mps_optimized(q1, q2)
    diff = torch.norm(res_n - res_o)
    print(f"Symmetry Check (L2 Diff): {diff:.6f}")

if __name__ == "__main__":
    # Fix the DDE error reported in feedback
    try:
        dde = DiscreteDecisionEngine(state_dim=256, num_actions=4)
        print("[M24-CW] DiscreteDecisionEngine initialized successfully.")
    except Exception as e:
        print(f"[M24-CW] DDE Fix Failed: {e}")

    if torch.backends.mps.is_available():
        run_audit()
    else:
        print("MPS not available. Skipping hardware-specific audit.")