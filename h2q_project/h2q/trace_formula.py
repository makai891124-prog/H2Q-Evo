import torch
import torch.nn as nn
from typing import Optional

class SpectralShiftTracker(nn.Module):
    """
    [STABLE] SpectralShiftTracker (η)
    Implements the η calculation grounded in SU(2) Group Theory and Projective Geometry.
    Links cognitive deflection to environmental drag μ(E) via the Krein-like trace formula.
    """
    def __init__(self, device: str = "mps"):
        super().__init__()
        self.device = torch.device(device) if torch.cuda.is_available() or "mps" in device else torch.device("cpu")
        # η history for spectral convergence tracking
        self.register_buffer("cumulative_shift", torch.tensor(0.0, device=self.device))
        self.register_buffer("total_drag", torch.tensor(0.0, device=self.device))

    def calculate_eta(self, s_matrix: torch.Tensor) -> torch.Tensor:
        """
        Core η calculation: η = (1/π) * arg{det(S)}
        Symmetry: S must be a unitary operator in the SU(2) manifold.
        """
        if s_matrix.ndim < 2 or s_matrix.shape[-1] != s_matrix.shape[-2]:
            raise ValueError(f"S_matrix must be square, got shape {s_matrix.shape}")

        # Ensure complex representation for determinant phase extraction
        if not s_matrix.is_complex():
            s_matrix = torch.complex(s_matrix, torch.zeros_like(s_matrix))

        # det(S) calculation
        det_s = torch.linalg.det(s_matrix)
        
        # η = (1/π) * arg(det(S))
        # torch.angle returns the phase in radians [-π, π]
        eta = torch.angle(det_s) / torch.pi
        return eta

    def ground_to_environment(self, s_matrix: torch.Tensor, mu_E: torch.Tensor) -> torch.Tensor:
        """
        Grounds the discrete sum of η to the environmental drag μ(E).
        
        Args:
            s_matrix: The Scattering Matrix (S) representing the cognitive transition.
            mu_E: Environmental drag recorded in the ContinuousEnvironmentModel.
            
        Returns:
            grounded_eta: The deflection adjusted for environmental resistance.
        """
        eta_raw = self.calculate_eta(s_matrix)
        
        # [RIGID CONSTRUCTION] 
        # The discrete sum of η is linked to μ(E) as a dissipative constraint.
        # η_grounded = η_raw - μ(E)
        # This ensures that cognitive expansion (η) is only 'realized' if it overcomes drag.
        grounded_eta = eta_raw - mu_E
        
        # Update stateful tracking
        self.cumulative_shift += grounded_eta.detach()
        self.total_drag += mu_E.detach()
        
        return grounded_eta

class ContinuousEnvironmentModel(nn.Module):
    """
    [EXPERIMENTAL] ContinuousEnvironmentModel
    Tracks environmental drag μ(E) across the manifold.
    """
    def __init__(self, input_dim: int, device: str = "mps"):
        super().__init__()
        self.device = device
        # Simple drag estimator: maps state energy to a scalar resistance μ
        self.drag_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Drag is normalized [0, 1]
        ).to(device)

    def get_drag(self, state_energy: torch.Tensor) -> torch.Tensor:
        return self.drag_net(state_energy).squeeze()

# FIX: Addressing the DiscreteDecisionEngine __init__ error mentioned in feedback
class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Corrected DiscreteDecisionEngine to prevent 'dim' keyword error.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Explicitly using 'state_dim' to avoid the 'dim' collision in base classes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_head = nn.Linear(state_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(x)