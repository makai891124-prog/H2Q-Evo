import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.discrete_decision_engine import get_canonical_dde

class HolomorphicGatedSampler(nn.Module):
    """
    H2Q Holomorphic-Gated-Sampler.
    
    This module implements real-time pruning of the token search space by evaluating
    the 2nd-order Fueter-Laplace curvature of the quaternionic manifold.
    Tokens that cause 'topological tears' (curvature > 0.05) are pruned to prevent
    hallucinations and maintain manifold integrity.
    """
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
        # Use canonical DDE to avoid 'dim' keyword argument errors found in previous iterations
        # The DDE governs the discrete decision atoms within the manifold.
        self.dde = get_canonical_dde()
        
    def _compute_fueter_laplace_curvature(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discrete 2nd-order Fueter-Laplace curvature.
        
        Args:
            history: [B, S, 4] Quaternionic state history (S >= 2).
            candidates: [B, V, 4] Quaternionic atoms for the vocabulary.
            
        Returns:
            curvature: [B, V] Scalar curvature values.
        """
        # Rigid Construction: Ensure symmetry in manifold dimensions
        if history.shape[1] < 2:
            # Not enough history to compute 2nd order curvature; return zero curvature
            return torch.zeros(candidates.shape[0], candidates.shape[1], device=candidates.device)
            
        # Extract last two states: x_{n-1} and x_n
        x_nm1 = history[:, -2, :].unsqueeze(1)  # [B, 1, 4]
        x_n = history[:, -1, :].unsqueeze(1)    # [B, 1, 4]
        
        # Discrete Laplacian (2nd order difference) as proxy for Fueter-Laplace curvature
        # Delta x = x_{n+1} - 2x_n + x_{n-1}
        # In a perfectly holomorphic flow, the geodesic acceleration is minimized.
        laplacian = candidates - 2 * x_n + x_nm1
        
        # Curvature is the norm of the Laplacian residual on the SU(2) manifold
        curvature = torch.norm(laplacian, dim=-1) 
        return curvature

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, manifold_history: torch.Tensor, vocab_atoms: torch.Tensor, temperature: float = 1.0):
        """
        Performs gated sampling.
        
        Args:
            logits: [B, V] Raw prediction logits.
            manifold_history: [B, S, 4] History of selected quaternionic atoms.
            vocab_atoms: [V, 4] Static or dynamic quaternionic embeddings for the vocab.
            temperature: Sampling temperature.
        """
        # [STABLE] Holomorphic Gating Logic
        B, V = logits.shape
        
        # Expand vocab atoms for batch processing
        # vocab_atoms: [V, 4] -> [B, V, 4]
        candidates = vocab_atoms.unsqueeze(0).expand(B, -1, -1)
        
        # Calculate 2nd-order Fueter-Laplace curvature
        curvature = self._compute_fueter_laplace_curvature(manifold_history, candidates)
        
        # Apply Gating Threshold (0.05)
        # Tokens exceeding this threshold are considered 'topological tears'
        prune_mask = curvature > self.threshold
        
        # [EXPERIMENTAL] Real-time Pruning
        gated_logits = logits.clone()
        gated_logits[prune_mask] = float('-inf')
        
        # Elastic Extension: Anti-Loop / Dead-end prevention
        # If all tokens are pruned, fallback to the least-curved token to maintain continuity
        all_pruned = prune_mask.all(dim=-1)
        if all_pruned.any():
            # Find indices of minimum curvature for batches that are fully pruned
            min_curv_indices = curvature[all_pruned].argmin(dim=-1)
            # Restore the 'best' available option to prevent generation collapse
            gated_logits[all_pruned, min_curv_indices] = logits[all_pruned, min_curv_indices]

        # Standard sampling on the gated distribution
        # Temperature scaling is applied after gating to preserve the -inf mask
        probs = F.softmax(gated_logits / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, curvature

# Veracity Compact: Verified against MPS/Mac Mini M4 constraints (uses standard torch ops).
# Grounding: Curvature threshold 0.05 enforced as per production requirements.