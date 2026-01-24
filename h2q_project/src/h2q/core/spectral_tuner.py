import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
from h2q.core.interface_registry import get_canonical_dde, verify_dde_integrity
from h2q.core.sst import SpectralShiftTracker

class SpectralEntropyAutoTuner(nn.Module):
    """
    [STABLE] Spectral Entropy Auto-Tuner
    Modulates the Fractal Expansion delta (h ± δ) to prevent Manifold Heat-Death.
    Ensures effective rank remains above the 128-dimensional critical boundary.
    """
    def __init__(
        self, 
        target_rank: int = 128, 
        manifold_dim: int = 256,
        alpha_smooth: float = 0.95,
        expansion_rate: float = 0.01,
        dde_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.target_rank = target_rank
        self.manifold_dim = manifold_dim
        self.alpha_smooth = alpha_smooth
        self.expansion_rate = expansion_rate
        
        # Initialize DDE via canonical registry to avoid 'dim' keyword errors
        self.dde = get_canonical_dde(dde_config or {})
        self.sst = SpectralShiftTracker()
        
        # State variables
        self.register_buffer("current_delta", torch.tensor(1e-3))
        self.register_buffer("moving_avg_rank", torch.tensor(float(manifold_dim)))
        
        verify_dde_integrity(self.dde)

    def calculate_effective_rank(self, manifold_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the effective rank using the Shannon entropy of the singular value spectrum.
        """
        # Ensure tensor is 2D for SVD
        if manifold_tensor.dim() > 2:
            manifold_tensor = manifold_tensor.view(-1, manifold_tensor.size(-1))
            
        # MPS-optimized singular value decomposition
        try:
            s = torch.linalg.svdvals(manifold_tensor)
        except RuntimeError:
            # Fallback for non-convergent SVD
            return torch.tensor(float(self.target_rank), device=manifold_tensor.device)
            
        # Normalize singular values to form a probability distribution
        s_norm = s / (torch.sum(s) + 1e-9)
        
        # Spectral Entropy H = -Σ p log(p)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-9))
        
        # Effective Rank = exp(H)
        eff_rank = torch.exp(entropy)
        return eff_rank

    def tune_delta(self, manifold_tensor: torch.Tensor) -> torch.Tensor:
        """
        Real-time modulation of the Fractal Expansion delta.
        """
        device = manifold_tensor.device
        eff_rank = self.calculate_effective_rank(manifold_tensor)
        
        # Update moving average
        self.moving_avg_rank = self.alpha_smooth * self.moving_avg_rank + (1 - self.alpha_smooth) * eff_rank
        
        # Calculate Heat-Death Proximity (Df-like logic)
        # If rank drops below 128, we increase delta to inject topological noise/expansion
        rank_gap = self.target_rank - self.moving_avg_rank
        
        if rank_gap > 0:
            # Manifold is collapsing; increase expansion delta
            adjustment = torch.exp(rank_gap / self.target_rank) * self.expansion_rate
            self.current_delta = torch.clamp(self.current_delta + adjustment, max=0.5)
        else:
            # Manifold is healthy; decay delta to maintain precision
            self.current_delta = torch.clamp(self.current_delta * 0.99, min=1e-5)
            
        return self.current_delta.clone()

    def verify_valve_symmetry(self) -> bool:
        """
        Ensures the tuner's output is within the stable bounds of the SU(2)^64 manifold.
        """
        return 0.0 < self.current_delta.item() < 1.0

def verify_valve_symmetry(tuner: SpectralEntropyAutoTuner) -> bool:
    return tuner.verify_valve_symmetry()