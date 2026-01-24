import torch
import torch.nn as nn
from h2q.quaternion_ops import quaternion_normalize, quaternion_mul
from h2q.core.interface_registry import get_canonical_dde, topological_dde_normalization

class BiharmonicLogicStabilizer(nn.Module):
    """
    Biharmonic-Logic-Stabilizer Middleware.
    Applies 4th-order Fueter-Laplace corrections to the hidden state flow 
    to enforce holomorphicity and prevent topological tears (hallucinations).
    """
    def __init__(self, channels, alpha=0.01, dde_kwargs=None):
        super().__init__()
        self.channels = channels
        self.alpha = alpha # Correction strength
        
        # Fix for Runtime Error: DiscreteDecisionEngine.__init__() got an unexpected keyword argument 'dim'
        # We use the canonical registry to normalize arguments before instantiation.
        safe_kwargs = topological_dde_normalization(dde_kwargs or {})
        self.dde = get_canonical_dde(**safe_kwargs)

    def _compute_discrete_fueter(self, q):
        """
        Computes the Discrete Fueter Operator (Df).
        q: Tensor of shape [B, L, C, 4] representing quaternionic hidden states.
        """
        # Finite differences across the sequence dimension as a proxy for manifold flow
        dq_dt = torch.gradient(q, dim=1)[0]
        
        # In a real SU(2) manifold, these would be gradients along the S3 basis
        # Here we approximate the Fueter condition: Df = dq/dt + i*dq/dx + j*dq/dy + k*dq/dz
        # For stabilization, we treat the components as the analytic deviation.
        return dq_dt 

    def _compute_biharmonic_correction(self, q):
        """
        Calculates the 4th-order correction (Delta^2).
        In the Fueter context, this is D(D_bar(D(D_bar(q)))).
        """
        # Laplacian approximation (2nd order)
        laplacian = torch.gradient(torch.gradient(q, dim=1)[0], dim=1)[0]
        # Biharmonic (4th order)
        biharmonic = torch.gradient(torch.gradient(laplacian, dim=1)[0], dim=1)[0]
        return biharmonic

    def forward(self, hidden_states, eta):
        """
        Args:
            hidden_states: [B, L, D] tensor.
            eta: Spectral Shift (learning progress metric).
        Returns:
            Stabilized hidden states.
        """
        B, L, D = hidden_states.shape
        assert D % 4 == 0, "Hidden dimension must be divisible by 4 for quaternionic mapping."
        
        # 1. Map to Quaternionic Manifold (S3)
        q = hidden_states.view(B, L, D // 4, 4)
        
        # 2. Identify Topological Tears (Df != 0)
        df = self._compute_discrete_fueter(q)
        tear_magnitude = torch.norm(df, dim=-1, keepdim=True)
        
        # 3. Consult Discrete Decision Engine (DDE)
        # DDE decides if the 'tear' requires biharmonic suppression based on eta
        # We pass tear_magnitude as the 'loss' proxy for the decision
        correction_mask = self.dde(tear_magnitude, eta)
        
        # 4. Apply 4th-order Fueter-Laplace Correction
        # Delta^2 q acts as a high-order smoother to restore holomorphicity
        delta_4 = self._compute_biharmonic_correction(q)
        
        # Apply correction: q_new = q - alpha * Delta^2 q
        q_stabilized = q - (self.alpha * correction_mask.unsqueeze(-1) * delta_4)
        
        # 5. Project back to SU(2) to maintain manifold symmetry
        q_stabilized = quaternion_normalize(q_stabilized)
        
        return q_stabilized.view(B, L, D)

def attach_biharmonic_stabilizer(model, alpha=0.01):
    """
    Utility to inject the stabilizer into an existing H2Q reasoning chain.
    """
    channels = getattr(model.config, "hidden_size", 512)
    stabilizer = BiharmonicLogicStabilizer(channels=channels, alpha=alpha)
    return stabilizer