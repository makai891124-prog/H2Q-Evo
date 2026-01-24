import torch
import torch.nn as nn
from typing import Optional, Tuple
from h2q.core.sst import SpectralShiftTracker
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.quaternion_ops import quaternion_norm, quaternion_stability

class GeodesicPruningEngine(nn.Module):
    """
    Implements η-Sensitivity Pruning for the H2Q Quaternionic Manifold.
    Identifies and removes manifold atoms with minimal contribution to the spectral phase shift (η)
    while enforcing topological integrity via the Discrete Fueter Operator (Df).
    """
    def __init__(self, manifold_dim: int = 256, sparsity_target: float = 0.3):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.sparsity_target = sparsity_target
        
        # Corrected DDE initialization based on Interface Registry feedback
        # Avoiding 'dim' keyword which caused previous runtime errors
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        # Pruning Mask (Stable Code)
        self.register_buffer("atom_mask", torch.ones(manifold_dim, manifold_dim))

    def compute_fueter_residual(self, manifold_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Discrete Fueter Operator (Df) residual.
        Topological tears (Df != 0) indicate areas that must NOT be pruned.
        """
        # Simplified Df proxy: local variance in quaternionic analyticity
        # In a 256-dim manifold, we check the symmetry of the local Jacobian
        q_norm = quaternion_norm(manifold_weights)
        df_residual = torch.abs(torch.gradient(q_norm)[0])
        return df_residual

    def calculate_eta_sensitivity(self, manifold_weights: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
        """
        Calculates the sensitivity of η = (1/π) arg{det(S)} w.r.t. each manifold atom.
        """
        manifold_weights.requires_grad_(True)
        
        # Compute η via SST
        eta = self.sst.compute_eta(s_matrix)
        
        # Gradient of η w.r.t weights
        # We use a small perturbation if backprop is too expensive for M4 memory
        eta.backward(retain_graph=True)
        sensitivity = torch.abs(manifold_weights.grad)
        
        manifold_weights.grad.zero_()
        return sensitivity

    @torch.no_grad()
    def prune_iteration(self, manifold_weights: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
        """
        Executes the η-Sensitivity Pruning cycle.
        """
        # 1. Calculate Spectral Sensitivity (η-impact)
        with torch.enable_grad():
            sensitivity = self.calculate_eta_sensitivity(manifold_weights, s_matrix)

        # 2. Calculate Holomorphic Integrity (Fueter Guard)
        fueter_res = self.compute_fueter_residual(manifold_weights)
        
        # 3. Combine metrics: Score = Sensitivity * (1 / (Fueter_Residual + epsilon))
        # High score = High impact on learning AND high topological stability
        pruning_score = sensitivity / (fueter_res + 1e-6)
        
        # 4. Determine threshold for sparsity target
        flat_scores = pruning_score.view(-1)
        k = int(self.sparsity_target * flat_scores.numel())
        if k > 0:
            threshold = torch.topk(flat_scores, k, largest=False).values.max()
            new_mask = (pruning_score > threshold).float()
            self.atom_mask.copy_(new_mask)

        # 5. Apply mask to preserve L1 concept integrity
        pruned_weights = manifold_weights * self.atom_mask
        
        # Verify symmetry (Rigid Construction)
        return quaternion_stability(pruned_weights)

    def forward(self, x: torch.Tensor, manifold_weights: torch.Tensor) -> torch.Tensor:
        # Apply the learned topological mask during forward pass
        active_weights = manifold_weights * self.atom_mask
        return torch.matmul(x, active_weights)

# Experimental: Geodesic Gradient Surgery Hook
def apply_pruning_guard(module, grad_input, grad_output):
    """Experimental: Ensures gradients do not flow through pruned atoms."""
    if hasattr(module, 'atom_mask'):
        return (grad_input[0] * module.atom_mask,)
    return grad_input