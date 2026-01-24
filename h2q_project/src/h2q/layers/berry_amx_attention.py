import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from h2q.group_ops import HamiltonProductAMX
from h2q.core.sst import SpectralShiftTracker

class BerryAMXAttention(nn.Module):
    """
    Berry-Phase-Attention Kernel optimized for M4 AMX registers.
    Replaces standard Softmax attention with Spinor Interference Patterns.
    
    Mathematical Foundation:
    Instead of scalar dot-products, we compute the SU(2) overlap between quaternionic spinors.
    The attention weight is the Berry Phase (geometric phase) accumulated during the 
    geodesic transport between Query and Key atoms.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % 4 == 0, "Embedding dimension must be a multiple of 4 (Quaternionic Atom constraint)"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize DDE using LatentConfig to avoid 'dim' keyword error
        config = LatentConfig()
        # Manually setting attributes if config doesn't support them in __init__
        config.latent_dim = embed_dim
        self.dde = DiscreteDecisionEngine(config)
        
        self.sst = SpectralShiftTracker()
        self.amx_engine = HamiltonProductAMX()
        
        # Quaternionic projections (W, I, J, K components)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        """
        Forward pass utilizing Spinor Interference.
        """
        B, L, E = x.shape
        
        # 1. Project to Quaternionic Space
        q = self._split_heads(self.q_proj(x))  # [B, H, L, D]
        k = self._split_heads(self.k_proj(x))  # [B, H, L, D]
        v = self._split_heads(self.v_proj(x))  # [B, H, L, D]

        # 2. AMX-Optimized Spinor Interference
        # We treat the head_dim as a collection of 4-atom quaternions
        # Interference (I) = HamiltonProduct(Q, K_conjugate)
        # This replaces QK^T
        
        # Reshape for Hamilton Product: [B*H, L, D/4, 4]
        q_quat = q.reshape(-1, L, self.head_dim // 4, 4)
        k_quat = k.reshape(-1, L, self.head_dim // 4, 4)
        
        # Compute interference pattern using M4 AMX registers
        # interference shape: [B*H, L, L, 4] (Pairwise quaternionic overlap)
        interference = self.amx_engine.apply_hamilton_product(q_quat, k_quat.transpose(1, 2))

        # 3. Berry Phase Extraction
        # The 'attention weight' is the normalized real component (Berry Phase cosine)
        # plus the imaginary vector components representing the rotation axis.
        # We use the Discrete Decision Engine to modulate the phase stability.
        
        # Calculate norm for SU(2) normalization
        norm = torch.norm(interference, dim=-1, keepdim=True) + 1e-6
        spinor_weights = interference / norm
        
        # Apply DDE to select optimal phase-shift atoms
        # This replaces the Softmax operation
        decision_mask = self.dde.forward(spinor_weights.mean(dim=-2))
        spinor_weights = spinor_weights * decision_mask.unsqueeze(2)

        # 4. Value Aggregation via Geodesic Flow
        # Y = SpinorWeights * V (Quaternionic multiplication)
        v_quat = v.reshape(-1, L, self.head_dim // 4, 4)
        
        # [B*H, L, L, 4] x [B*H, L, D/4, 4] -> [B*H, L, D/4, 4]
        # Using AMX for the final aggregation
        context_quat = self.amx_engine.apply_hamilton_product(spinor_weights, v_quat)
        
        # 5. Reconstruct and Project
        context = context_quat.view(B, self.num_heads, L, self.head_dim)
        context = context.transpose(1, 2).reshape(B, L, E)
        
        output = self.out_proj(context)
        
        # 6. Veracity Audit (Spectral Shift)
        # Track the entropy of the interference pattern to prevent manifold collapse
        self.sst.update(interference)
        
        return output

    def get_veracity_metrics(self):
        return {
            "spectral_shift": self.sst.get_eta(),
            "dde_entropy": self.dde.get_entropy() if hasattr(self.dde, 'get_entropy') else 0.0
        }
