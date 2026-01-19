import torch
import numpy as np
from h2q.quaternion_ops import quaternion_mul
from h2q.core.accelerators.m4_amx_kernel import M4AMXHamiltonKernel, get_kernel
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine

class AMXPrecisionAuditor:
    """
    Diagnostic suite to verify bit-accuracy of 16x16 tiled Hamilton Products
    on M4 AMX hardware against high-precision CPU baselines.
    """
    def __init__(self, threshold=1e-7):
        self.threshold = threshold
        # Fix: Removed 'dim' argument to honor feedback regarding unexpected keyword
        self.dde = DiscreteDecisionEngine()
        self.kernel = get_kernel() # Retrieves the M4-optimized Hamilton kernel

    def generate_test_tensors(self, size=16):
        """
        Generates 16x16 tiles for quaternionic multiplication.
        Quaternions are represented as [Batch, 4] or [Batch, Seq, 4].
        """
        # Use float64 for the ground truth baseline
        q1_cpu = torch.randn(size, size, 4, dtype=torch.float64)
        q2_cpu = torch.randn(size, size, 4, dtype=torch.float64)
        
        # M4 AMX typically operates on float32 or bfloat16; we use float32 for the test
        q1_amx = q1_cpu.to(torch.float32).to("mps")
        q2_amx = q2_cpu.to(torch.float32).to("mps")
        
        return (q1_cpu, q2_cpu), (q1_amx, q2_amx)

    def run_audit(self, iterations=100):
        print(f"[AMX_PRECISION_AUDIT] Starting suite: Threshold={self.threshold}")
        drifts = []

        for i in range(iterations):
            cpu_data, amx_data = self.generate_test_tensors()
            
            # 1. Compute High-Precision CPU Baseline
            # quaternion_mul expected to handle [..., 4] tensors
            res_cpu = quaternion_mul(cpu_data[0], cpu_data[1])

            # 2. Compute M4 AMX Tiled Product
            # The M4AMXHamiltonKernel is designed for 16x16 tiling on Apple Silicon
            res_amx = self.kernel.forward(amx_data[0], amx_data[1])

            # 3. Calculate L1-Drift
            # Move AMX result back to CPU and upcast for comparison
            res_amx_cpu = res_amx.to(torch.float64).cpu()
            
            l1_drift = torch.mean(torch.abs(res_cpu - res_amx_cpu)).item()
            drifts.append(l1_drift)

            if l1_drift > self.threshold:
                print(f"[!] ALERT: Iteration {i} exceeded threshold: {l1_drift:.2e}")

        avg_drift = np.mean(drifts)
        max_drift = np.max(drifts)
        
        status = "PASSED" if max_drift <= self.threshold else "FAILED"
        
        print("--- AUDIT RESULTS ---")
        print(f"Status: {status}")
        print(f"Average L1-Drift: {avg_drift:.2e}")
        print(f"Maximum L1-Drift: {max_drift:.2e}")
        print(f"Target Threshold: {self.threshold:.2e}")
        
        return {
            "status": status,
            "avg_drift": avg_drift,
            "max_drift": max_drift
        }

if __name__ == "__main__":
    # Ensure MPS is available for M4 testing
    if not torch.backends.mps.is_available():
        print("[!] Error: MPS not available. This diagnostic requires M4 hardware.")
    else:
        auditor = AMXPrecisionAuditor(threshold=1e-7)
        auditor.run_audit()