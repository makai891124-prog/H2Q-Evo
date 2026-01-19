import torch
import torch.nn as nn
import time
from h2q.core.zwi_engine import GeometricCrystal, to_quaternion_basis
from h2q.core.sst import SpectralShiftTracker
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.utils.mps_compat import ensure_complex_support

class ZWIBenchmark:
    """
    Zero-Weight Inference (ZWI) Benchmark.
    Validates the 'Geometric Crystal' hypothesis: Learning occurs via phase-shifts (η)
    in a persistent state vector ψ on an SU(2) manifold, while neural weights remain frozen.
    """
    def __init__(self, dim=256, num_classes=10, device=None):
        self.device = device if device else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.dim = dim
        self.num_classes = num_classes
        
        # Initialize the Discrete Decision Engine (Corrected: No 'dim' argument to avoid Runtime Error)
        self.dde = get_canonical_dde()
        
        # Initialize the Spectral Shift Tracker
        self.sst = SpectralShiftTracker()
        
        # Frozen Projection Weights (The 'Static' Manifold)
        # We use a fixed random orthogonal-like quaternionic projection
        self.frozen_projection = torch.randn(dim // 4, dim // 4, 4, device=self.device)
        self.frozen_projection = quaternion_normalize(self.frozen_projection)
        
        # The Geometric Crystal: Persistent State Vector ψ
        # Initialized as a unit spinor on the S3 hypersphere
        self.psi = torch.randn(1, dim // 4, 4, device=self.device)
        self.psi = quaternion_normalize(self.psi)
        
        # Class-specific Phase Anchors (Frozen)
        self.anchors = torch.randn(num_classes, dim // 4, 4, device=self.device)
        self.anchors = quaternion_normalize(self.anchors)

    def run_inference(self, x, labels=None, update_crystal=True):
        """
        Performs inference by measuring the resonance between input and the crystal.
        """
        batch_size = x.shape[0]
        
        # 1. Map input to Quaternionic Basis
        q_in = to_quaternion_basis(x, self.dim).to(self.device) # [B, dim/4, 4]
        
        # 2. Interaction: Hamilton Product with Frozen Projection
        # y = q_in ⊗ W_frozen
        # For simplicity in benchmark, we treat q_in as the operator
        
        # 3. Calculate Spectral Shift η relative to the current Crystal ψ
        # η = (1/π) arg{det(S)}
        # We simulate the scattering matrix S as the transition from ψ to ψ'
        psi_prev = self.psi.clone()
        
        # Apply interaction to evolve ψ
        # ψ_new = mean(q_in) ⊗ ψ_prev
        q_mean = torch.mean(q_in, dim=0, keepdim=True)
        psi_new = quaternion_mul(q_mean, psi_prev)
        psi_new = quaternion_normalize(psi_new)
        
        if update_crystal:
            self.psi.data = psi_new.data

        # 4. Classification via Geodesic Distance to Anchors
        # We measure which class anchor the evolved ψ is most 'in phase' with
        # This is the ZWI equivalent of a forward pass
        
        # Expand psi to match anchors [NumClasses, dim/4, 4]
        psi_expanded = self.psi.expand(self.num_classes, -1, -1)
        
        # Dot product on S3 (Hamiltonian inner product approximation)
        resonance = torch.sum(psi_expanded * self.anchors, dim=(1, 2))
        predictions = torch.argmax(resonance)
        
        # 5. Track Spectral Shift
        eta = self.sst.update(psi_prev, psi_new)
        
        return predictions, eta

    def benchmark(self, data_loader, iterations=100):
        print(f"[ZWI_BENCHMARK] Starting Geometric Crystal Validation on {self.device}")
        print(f"[ZWI_BENCHMARK] Weights: FROZEN | State ψ: PERSISTENT")
        
        start_time = time.time()
        correct = 0
        total = 0
        eta_history = []

        for i, (x, y) in enumerate(data_loader):
            if i >= iterations: break
            
            x = x.to(self.device)
            pred, eta = self.run_inference(x, update_crystal=True)
            
            if pred.item() == y[0].item(): # Assuming batch size 1 for pure trajectory tracking
                correct += 1
            total += 1
            eta_history.append(eta)
            
            if i % 10 == 0:
                print(f"Iteration {i}: η = {eta:.4f} | Accuracy: {100 * correct / total:.2f}%")

        end_time = time.time()
        print(f"--- BENCHMARK COMPLETE ---")
        print(f"Final Accuracy: {100 * correct / total:.2f}%")
        print(f"Mean Spectral Shift: {torch.tensor(eta_history).mean():.6f}")
        print(f"Latency: {(end_time - start_time)/total:.4f}s / sample")

if __name__ == "__main__":
    # Mock Data for Validation
    class MockLoader:
        def __iter__(self):
            for _ in range(100):
                yield torch.randn(1, 256), torch.tensor([torch.randint(0, 10, (1,))])

    tester = ZWIBenchmark(dim=256, num_classes=10)
    tester.benchmark(MockLoader(), iterations=50)
