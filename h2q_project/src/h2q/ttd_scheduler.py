import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from h2q.core.sst import SpectralShiftTracker
from h2q.knot_kernel import H2Q_Knot_Kernel
from h2q.core.interface_registry import get_canonical_dde

class TTDState:
    """Encapsulates the temporal state of the manifold for dilation calculations."""
    def __init__(self, window_size: int = 10):
        self.eta_history = []
        self.window_size = window_size

    def update(self, eta: float):
        self.eta_history.append(eta)
        if len(self.eta_history) > self.window_size:
            self.eta_history.pop(0)

    @property
    def volatility(self) -> float:
        if len(self.eta_history) < 2:
            return 0.0
        tensor_eta = torch.tensor(self.eta_history)
        return torch.std(tensor_eta).item()

class TopologicalTimeDilation(nn.Module):
    """
    TTD Scheduler: Dynamically adjusts Knot Kernel recursion depth based on 
    Spectral Shift (eta) volatility.
    
    Rigid Construction: Depth is clamped between 1 and 8 to respect M4 (16GB) memory limits.
    Elastic Extension: Uses a sigmoidal mapping to translate logical noise into compute time.
    """
    def __init__(
        self,
        knot_kernel: H2Q_Knot_Kernel,
        min_depth: int = 1,
        max_depth: int = 8,
        sensitivity: float = 5.0
    ):
        super().__init__()
        self.knot_kernel = knot_kernel
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sensitivity = sensitivity
        
        # Initialize components using canonical registry to avoid 'dim' kwarg errors
        self.sst = SpectralShiftTracker()
        self.dde = get_canonical_dde() 
        self.state = TTDState(window_size=12)

    def _calculate_dynamic_depth(self, volatility: float) -> int:
        """
        Maps eta-volatility to recursion depth k.
        High volatility (logical complexity) -> Higher k (Time Dilation).
        """
        # Sigmoidal scaling: k = min + (max-min) * sigmoid(alpha * vol)
        scale = 1.0 / (1.0 + math.exp(-self.sensitivity * volatility))
        depth = self.min_depth + (self.max_depth - self.min_depth) * scale
        return int(round(depth))

    def forward(self, manifold: torch.Tensor, mu_e: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            manifold: The 256-dim topological knot (S3 representation).
            mu_e: Environmental drag vector.
        Returns:
            Updated manifold and the depth k utilized.
        """
        # 1. Audit current spectral shift
        # Note: sst.track returns eta based on the manifold's unitary deflection
        eta = self.sst.track(manifold)
        self.state.update(eta)
        
        # 2. Determine recursion depth (Time Dilation Factor)
        k = self._calculate_dynamic_depth(self.state.volatility)
        
        # 3. Execute Recursive Knot Flow
        # We treat each recursion as a geodesic step on the 3-sphere
        current_manifold = manifold
        for _ in range(k):
            # The Knot Kernel performs the SU(2) isomorphic transformation
            current_manifold = self.knot_kernel(current_manifold, mu_e)
            
            # Holomorphic Auditing: Ensure the manifold hasn't collapsed (HDI check)
            # If DDE signals a 'topological tear', we break recursion early to save compute
            decision = self.dde.decide(current_manifold, eta)
            if decision.item() == 0: # 0 = Halt/Stabilize
                break
                
        return current_manifold, k

def mock_hamilton_kernel(x, mu):
    """Fallback kernel for testing symmetry."""
    return x + 0.01 * torch.tanh(mu * x)
