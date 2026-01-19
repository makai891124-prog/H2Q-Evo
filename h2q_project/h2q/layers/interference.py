import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiscreteDecisionEngine(nn.Module):
    """
    Corrected Decision Engine for H2Q.
    The previous version failed due to 'dim' keyword mismatch.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.SiLU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

class CPIGating(nn.Module):
    """
    Constructive Phase Interference Gating (CPIG).
    Replaces Dot-Product Attention with SU(2) Spinor Interference.
    
    Logic: 
    1. Map inputs to Quaternionic Spinors (256-dim -> 128 complex pairs).
    2. Calculate the Berry Phase difference between Query and Key spinors.
    3. Route information via constructive interference patterns.
    """
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for complex spinor mapping."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Spinor Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Spectral Shift Tracker (η) components
        self.register_buffer("eta", torch.tensor(0.0))
        
        # Fixed DiscreteDecisionEngine call
        self.decision_engine = DiscreteDecisionEngine(latent_dim=dim)

    def _to_spinors(self, x: torch.Tensor):
        # Reshape to complex spinors (B, N, H, D/2, 2)
        # Representing SU(2) elements as pairs of complex numbers
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim // 2, 2)
        return torch.view_as_complex(x)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        device = x.device

        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Transform to Spinor Space
        # q_s, k_s are complex tensors representing SU(2) states
        q_s = self._to_spinors(q)
        k_s = self._to_spinors(k)
        
        # 3. Calculate Berry Phase Interference
        # Interference I = <psi_q | psi_k>
        # We use the inner product of spinors to determine phase alignment
        interference = torch.einsum("bnhd,bmhd->bnhm", q_s, k_s.conj())
        
        # Extract Phase (Berry Phase approximation in discrete manifold)
        phase = torch.angle(interference)
        
        # Constructive Interference Weighting: cos^2(theta/2)
        # This maps to the probability of state transition in SU(2)
        weights = torch.cos(phase / 2) ** 2
        weights = weights / (math.sqrt(self.head_dim) + 1e-6)
        weights = F.softmax(weights, dim=-1)

        # 4. Information Routing
        # Apply interference weights to Values
        v_res = v.view(B, N, self.num_heads, self.head_dim)
        out = torch.einsum("bnhm,bmhd->bnhd", weights, v_res)
        out = out.reshape(B, N, C)

        # 5. Spectral Shift Tracking (η)
        # η = (1/π) arg{det(S)} - simplified for runtime tracking
        with torch.no_grad():
            # Trace-based approximation of the spectral shift
            s_matrix = weights.mean(dim=1) # Average over heads
            self.eta = torch.angle(torch.linalg.det(s_matrix + 1e-6 * torch.eye(s_matrix.size(-1), device=device))).mean() / math.pi

        # 6. Decision Gating
        gate = self.decision_engine(out)
        return out * gate

    def get_spectral_shift(self):
        return self.eta.item()