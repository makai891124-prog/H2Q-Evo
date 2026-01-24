import torch
import torch.nn as nn
from typing import Optional
from .. discrete_decision_engine import get_canonical_dde
from ...quaternion_ops import quaternion_norm

class HolomorphicSparsityGuard(nn.Module):
    """
    Holomorphic Sparsity Guard
    
    Dynamically prunes manifold knots based on the Fueter-analyticity residual.
    Following the Veracity Compact, this module ensures that only knots contributing
    to the analytic geodesic flow (Df=0) or meeting the sparsity noise floor are preserved.
    """
    def __init__(self, noise_floor: float = 1e-7, hallucination_threshold: float = 0.5):
        super().__init__()
        self.noise_floor = noise_floor
        self.hallucination_threshold = hallucination_threshold
        
        # RIGID CONSTRUCTION: Use canonical getter to avoid 'dim' argument error 
        # reported in previous runtime feedback.
        self.dde = get_canonical_dde()

    def compute_fueter_residual(self, q_knots: torch.Tensor) -> torch.Tensor:
        """
        Approximates the Discrete Fueter Operator (Df) across the manifold knots.
        q_knots: Tensor of shape [Batch, Sequence, 4] representing quaternions (1, i, j, k).
        """
        if q_knots.shape[1] < 3:
            return torch.zeros(q_knots.shape[:-1], device=q_knots.device)

        # Central difference approximation of the Fueter derivative along the manifold index
        # In a 256-dim manifold, we treat the sequence index as a proxy for the geodesic parameter.
        dq = (q_knots[:, 2:] - q_knots[:, :-2]) / 2.0
        
        # The Fueter residual is the norm of the variation in the su(2) Lie Algebra
        # For a perfectly analytic knot in a static flow, Df -> 0.
        residual = torch.norm(dq, dim=-1) 
        
        # Pad to maintain symmetry with input shape
        padding = torch.zeros((q_knots.shape[0], 1), device=q_knots.device)
        return torch.cat([padding, residual, padding], dim=1)

    def forward(self, knots: torch.Tensor) -> torch.Tensor:
        """
        Applies the sparsity guard to the manifold knots.
        
        1. Identifies 'topological tears' (residual > hallucination_threshold).
        2. Identifies 'dead knots' (residual < noise_floor) to enforce sparsity.
        """
        # Ensure input is on the correct device (MPS/CPU)
        residual = self.compute_fueter_residual(knots)

        # ELASTIC WEAVING: Dual-threshold pruning
        # Prune if residual is below noise floor (Sparsity) 
        # OR if residual is above hallucination threshold (Veracity/Topological Tear)
        
        # Logic: Keep if (residual >= noise_floor) AND (residual <= hallucination_threshold)
        valid_mask = (residual >= self.noise_floor) & (residual <= self.hallucination_threshold)
        
        # Apply mask to the quaternionic knots
        pruned_knots = knots * valid_mask.unsqueeze(-1).to(knots.dtype)

        # Metacognitive Audit: Log sparsity ratio if in experimental mode
        # sparsity_ratio = 1.0 - (valid_mask.float().mean().item())
        
        return pruned_knots

    def audit_logic_curvature(self, knots: torch.Tensor) -> float:
        """
        Calculates the global logic curvature (mean Fueter residual) for the HDI monitor.
        """
        residual = self.compute_fueter_residual(knots)
        return torch.mean(residual).item()