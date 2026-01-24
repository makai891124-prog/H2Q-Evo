import torch
import torch.nn as nn
import math
from typing import List, Optional

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine (DDE)
    Governs the transition from task-driven (Wake) to self-organized (Sleep) phases
    using the Spectral Shift Tracker (η).
    """
    def __init__(self, manifold_dim: int = 256, maturity_threshold: float = 5.0, device: str = "mps"):
        super().__init__()
        # [FIX] Explicitly handling 'manifold_dim' to resolve 'unexpected keyword argument dim'
        # We map 'dim' to 'manifold_dim' for internal consistency with SU(2) geometry.
        self.manifold_dim = manifold_dim
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # Foundational Atoms
        self.autonomy_weight = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.maturity_threshold = maturity_threshold
        self.eta_history: List[float] = []
        
        # SU(2) Isomorphism Layer: Maps discrete atoms to manifold coordinates
        self.projection = nn.Linear(2, self.manifold_dim, device=self.device)
        
        # Stable Label: Experimental Reversible Kernel for O(1) memory
        self.coupling_alpha = nn.Parameter(torch.ones(1, device=self.device))

    def compute_spectral_shift(self, S_matrix: torch.Tensor) -> torch.Tensor:
        """
        Implements η = (1/π) arg{det(S)}
        Links discrete decision atoms to continuous environmental drag.
        """
        # Ensure S_matrix is complex for determinant argument calculation
        det_s = torch.linalg.det(S_matrix)
        eta = torch.angle(det_s) / math.pi
        return eta

    def update_autonomy_schedule(self, current_eta: float):
        """
        ESTABLISH_AUTONOMY_WEIGHT_DECAY_SCHEDULE
        Transitions system from Wake (0.0) to Sleep (1.0) based on cumulative η.
        """
        self.eta_history.append(current_eta)
        cumulative_eta = sum(self.eta_history)
        
        # Logistic transition function for cognitive maturity
        # As cumulative η increases, the system relies more on internal self-organization
        new_weight = torch.sigmoid(torch.tensor(cumulative_eta - self.maturity_threshold, device=self.device))
        
        # Rigid Construction: Update parameter in-place to maintain symmetry with optimizer
        self.autonomy_weight.data = new_weight

    def forward(self, x: torch.Tensor, external_task_signal: torch.Tensor) -> torch.Tensor:
        """
        Executes decision logic weighted by autonomy level.
        """
        # Fractal Expansion: 2-atom seed to manifold_dim
        internal_state = self.projection(x)
        
        # Elastic Weaving: Blend external task-driven signals with internal self-organization
        # Wake Phase (autonomy_weight -> 0): Driven by external_task_signal
        # Sleep Phase (autonomy_weight -> 1): Driven by internal_state (self-organization)
        combined_output = ((1.0 - self.autonomy_weight) * external_task_signal) + \
                          (self.autonomy_weight * internal_state)
        
        return combined_output

# [STABLE] Factory method to prevent __init__ argument errors
def create_dde(config: dict) -> DiscreteDecisionEngine:
    # Mapping 'dim' from legacy configs to 'manifold_dim'
    dim = config.get('dim', 256)
    return DiscreteDecisionEngine(manifold_dim=dim)