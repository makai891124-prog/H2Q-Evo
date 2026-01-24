import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np

# [STABLE] Spectral Shift Tracker (eta) implementation
# Based on Krein-like trace formula: η = (1/π) arg{det(S)}
class SpectralShiftTracker(nn.Module):
    def __init__(self, window_size: int = 20):
        super().__init__()
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def compute_eta(self, S_matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the spectral shift η from the scattering/transition matrix S.
        S must be a square matrix representing the decision manifold state.
        """
        # Ensure S is complex for determinant calculation in SU(2) space
        if not S_matrix.is_complex():
            # Mapping real to complex representation of SU(2)
            # Assuming S is [N, N], we treat it as a projection of the quaternionic manifold
            S_complex = torch.complex(S_matrix, torch.zeros_like(S_matrix))
        else:
            S_complex = S_matrix

        # det(S) calculation
        # Using slogdet for numerical stability on Mac Mini M4 (MPS/CPU)
        sign, logabsdet = torch.linalg.slogdet(S_complex)
        
        # η = (1/π) * phase(det(S))
        # phase = arg(sign) + imag(logabsdet) -> since logabsdet is real for complex det, 
        # we use the angle of the sign (which is the unit complex phase)
        eta = torch.angle(sign) / torch.pi
        return eta

    def update_volatility(self, eta: torch.Tensor) -> float:
        self.history.append(eta.item())
        if len(self.history) < 2:
            return 0.0
        return float(np.std(list(self.history)))

# [STABLE] Fixed DiscreteDecisionEngine to resolve 'dim' keyword error
class DiscreteDecisionEngine(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        REASONING: The previous error 'unexpected keyword argument dim' 
        suggests a mismatch in the constructor signature. 
        Standardizing to 'input_size' and 'output_size'.
        """
        super().__init__()
        self.projection = nn.Linear(input_size, output_size)
        self.gate = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # Returns the S-matrix (transition probabilities) for the tracker
        logits = self.projection(x)
        s_matrix = self.gate(logits)
        return s_matrix

# [EXPERIMENTAL] Dynamic Drag Feedback Loop (DDFL)
class DynamicDragFeedbackLoop(nn.Module):
    def __init__(self, 
                 latent_dim: int = 256, 
                 initial_mu: float = 0.01, 
                 target_stability: float = 0.05):
        super().__init__()
        # Rigid Construction: Symmetry between Decision Engine and Tracker
        self.engine = DiscreteDecisionEngine(input_size=latent_dim, output_size=latent_dim)
        self.tracker = SpectralShiftTracker(window_size=30)
        
        # CEM Drag Coefficient μ(E)
        self.register_buffer("mu_e", torch.tensor(initial_mu))
        self.target_stability = target_stability
        self.adaptation_rate = 0.001

    def forward(self, x: torch.Tensor):
        """
        Executes the feedback loop: 
        1. Map input to S-matrix
        2. Compute η (Spectral Shift)
        3. Adjust μ(E) based on η volatility
        """
        # 1. Generate Decision Atom (S-matrix)
        s_matrix = self.engine(x)
        
        # 2. Track Spectral Shift
        eta = self.tracker.compute_eta(s_matrix)
        volatility = self.tracker.update_volatility(eta)
        
        # 3. Real-time Drag Adjustment (The Feedback Loop)
        # If volatility > target, increase drag to prevent catastrophic forgetting
        # If volatility < target, decrease drag to allow faster geodesic flow
        diff = volatility - self.target_stability
        new_mu = self.mu_e + (self.adaptation_rate * diff)
        
        # Clamp mu to prevent manifold stagnation or total collapse
        self.mu_e = torch.clamp(torch.tensor(new_mu), 1e-4, 0.5)
        
        return {
            "s_matrix": s_matrix,
            "eta": eta,
            "mu_e": self.mu_e,
            "volatility": volatility
        }

    def get_drag(self):
        return self.mu_e
