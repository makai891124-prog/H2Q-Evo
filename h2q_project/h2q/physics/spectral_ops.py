import torch
import torch.nn as nn
import torch.linalg as linalg
from typing import Optional

class SpectralEntropyRegularizer(nn.Module):
    """
    H2Q Spectral Entropy Regularizer [STABLE]
    
    Governed by the Veracity Compact: This implementation uses torch.linalg.svdvals
    which is verified for MPS (Mac Mini M4) compatibility. 
    
    Purpose: Prevents manifold singularity in the 256-dimensional quaternionic manifold
    by penalizing the collapse of the scattering matrix (S) eigenvalues.
    """
    def __init__(self, strength: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.strength = strength
        self.eps = eps

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calculates the spectral loss based on the Krein-like trace formula logic.
        Args:
            S (torch.Tensor): The scattering matrix or transition operator. 
                             Expected shape: [Batch, 256, 256] or [Batch, 1024, 1024] for real-quaternion mapping.
        Returns:
            torch.Tensor: Scalar loss term.
        """
        # Ensure S is at least 2D for linalg operations
        if S.dim() < 2:
            return torch.tensor(0.0, device=S.device)

        # 1. Extract Singular Values (Stable proxy for eigenvalues in non-Hermitian flows)
        # MPS Optimization: svdvals is more memory-efficient than full SVD
        s = linalg.svdvals(S)

        # 2. Log-Determinant Penalty: -log|det(S)| = -sum(log(sigma))
        # This prevents det(S) from approaching zero (manifold collapse)
        log_det_penalty = -torch.sum(torch.log(s + self.eps), dim=-1)

        # 3. Spectral Entropy: H = -sum(p * log(p))
        # Quantifies the 'spread' of cognitive atoms across the manifold
        p = (s**2) / (torch.sum(s**2, dim=-1, keepdim=True) + self.eps)
        entropy = -torch.sum(p * torch.log(p + self.eps), dim=-1)

        # 4. Combine: We want high entropy (distributed info) and high det (non-singular)
        # Loss = Strength * (LogDetPenalty - Entropy)
        total_spectral_loss = self.strength * (log_det_penalty.mean() - entropy.mean())
        
        return total_spectral_loss

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine [REFACTORED]
    
    FIX: Resolved 'unexpected keyword argument dim' by explicitly defining 
    the signature to accept manifold dimensions.
    """
    def __init__(self, input_dim: int = 256, output_dim: int = 256, **kwargs):
        super().__init__()
        # Elastic Extension: Capture 'dim' if passed by legacy callers to prevent Runtime Error
        self.dim = kwargs.get('dim', input_dim)
        self.projection = nn.Linear(self.dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

# EXPERIMENTAL: Quaternionic Symmetry Wrapper
def apply_spectral_shift(S: torch.Tensor, eta_target: float = 0.5) -> torch.Tensor:
    """
    Adjusts the scattering matrix to align with the target Spectral Shift (η).
    η = (1/π) arg{det(S)}
    """
    det_s = torch.linalg.det(S)
    eta_current = (1.0 / 3.14159) * torch.angle(det_s)
    shift_correction = torch.exp(1j * (eta_target - eta_current))
    return S * shift_correction.unsqueeze(-1).unsqueeze(-1)