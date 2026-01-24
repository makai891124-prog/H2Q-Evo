import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.interpolation import SpectralSlerp
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine
from h2q.core.interface_registry import get_canonical_dde

class TopologicalCooling(nn.Module):
    """
    Topological Cooling Middleware
    
    Applies Spherical Linear Interpolation (Slerp) toward the identity quaternion [1, 0, 0, 0]
    when the Heat-Death Index (HDI) exceeds a critical threshold, effectively reducing 
    spectral entropy and restoring manifold stability.
    """
    def __init__(self, threshold=0.85, base_cooling_rate=0.05):
        super().__init__()
        self.threshold = threshold
        self.base_cooling_rate = base_cooling_rate
        
        # RIGID CONSTRUCTION: Fix for 'unexpected keyword argument dim'
        # We use the canonical registry to ensure the DDE is instantiated correctly
        # regardless of legacy 'dim' requirements in the calling context.
        self.dde = get_canonical_dde()

    def _get_identity(self, reference_tensor):
        """Returns the identity quaternion [1, 0, 0, 0] matching the input shape."""
        identity = torch.zeros_like(reference_tensor)
        identity[..., 0] = 1.0
        return identity

    def forward(self, q_state, hdi):
        """
        Args:
            q_state (torch.Tensor): Quaternionic state tensor of shape [..., 4].
            hdi (torch.Tensor): Heat-Death Index tensor of shape [B].
            
        Returns:
            torch.Tensor: Cooled quaternionic state.
        """
        # Ensure q_state is normalized on S³
        q_state = F.normalize(q_state, p=2, dim=-1)
        
        # Determine cooling intensity via DDE (Elastic Extension)
        # The DDE evaluates if cooling is 'logically' necessary based on entropy noise
        cooling_decision = self.dde(hdi.unsqueeze(-1)) 
        
        # Calculate interpolation factor 't'
        # t = 0 (no cooling), t = 1 (full reset to identity)
        # Scaling: t increases as HDI approaches 1.0
        t = (hdi - self.threshold) / (1.0 - self.threshold + 1e-8)
        t = torch.clamp(t * self.base_cooling_rate, 0.0, 1.0)
        
        # Apply mask: only cool where HDI > threshold
        mask = (hdi > self.threshold).float()
        effective_t = t * mask * cooling_decision.squeeze(-1)
        
        if effective_t.max() <= 0:
            return q_state

        identity = self._get_identity(q_state)
        
        # ELASTIC WEAVING: Use SpectralSlerp for geodesic-preserving cooling
        # Reshape t for broadcasting across the manifold dimensions
        t_broadcast = effective_t.view(effective_t.shape[0], *([1] * (q_state.dim() - 1)))
        
        # Perform Slerp: q_cooled = Slerp(q_state, identity, t)
        # Note: SpectralSlerp is expected to handle the S³ geodesic path
        cooled_state = SpectralSlerp.apply(q_state, identity, t_broadcast)
        
        return cooled_state

    def audit_cooling_symmetry(self, q_in, q_out):
        """Verifies that the cooling process maintains SU(2) manifold constraints."""
        norm_in = torch.norm(q_in, p=2, dim=-1)
        norm_out = torch.norm(q_out, p=2, dim=-1)
        drift = torch.abs(norm_in - norm_out).max()
        return drift < 1e-5

# EXPERIMENTAL: Standardized DDE Wrapper to prevent 'dim' keyword errors
def safe_dde_init(dim=None, **kwargs):
    """
    Utility to initialize DDE without triggering the 'dim' unexpected argument error
    reported in the feedback loop.
    """
    if 'dim' in kwargs:
        del kwargs['dim']
    return DiscreteDecisionEngine(**kwargs)