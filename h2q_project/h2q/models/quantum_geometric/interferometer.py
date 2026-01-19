import torch
import torch.nn as nn
import torch.nn.functional as F

class BerryPhaseInterferometer(nn.Module):
    """
    [EXPERIMENTAL] Berry Phase Cross-Modality Interferometer
    Replaces standard contrastive loss with SU(2) geometric phase alignment.
    
    Logic: Maps Vision and Text embeddings to the SU(2) manifold as spinors,
    calculates the Pancharatnam-Berry phase between them, and minimizes the 
    Spectral Shift (eta) to ensure semantic isomorphism.
    """
    def __init__(self, embedding_dim: int = 256, n_spinors: int = 128):
        super().__init__()
        self.dim = embedding_dim
        self.n_spinors = n_spinors
        # Ensure symmetry: 256 real dims -> 128 complex pairs (spinors)
        assert embedding_dim == n_spinors * 2, "Embedding dim must be 2 * n_spinors for SU(2) mapping."
        
        # Metric scaling for environmental drag mu(E)
        self.mu_e = nn.Parameter(torch.ones(1) * 0.01)

    def _to_spinors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps R^256 to SU(2) spinors in C^(128x2).
        Input: (Batch, 256)
        Output: (Batch, 128, 2) complex
        """
        # Reshape to (Batch, 128, 2)
        x = x.view(-1, self.n_spinors, 2)
        # Normalize to unit sphere to satisfy SU(2) constraint |alpha|^2 + |beta|^2 = 1
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8
        return x / norm

    def forward(self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the geometric overlap and Spectral Shift η.
        """
        device = vision_embeds.device
        
        # 1. Project to SU(2) Spinors
        # We treat the 2-dim real vector as a complex scalar for simplicity in phase calc
        # or as a spinor [a, b]. Here we use the spinor overlap approach.
        psi_v = self._to_spinors(vision_embeds) # (B, 128, 2)
        psi_t = self._to_spinors(text_embeds)   # (B, 128, 2)

        # 2. Compute Complex Overlap <psi_v | psi_t>
        # Using real-space equivalent of complex inner product
        # <a,b|c,d> = (ac + bd) + i(ad - bc)
        real_part = (psi_v[..., 0] * psi_t[..., 0] + psi_v[..., 1] * psi_t[..., 1])
        imag_part = (psi_v[..., 0] * psi_t[..., 1] - psi_v[..., 1] * psi_t[..., 0])
        
        # 3. Calculate Berry Phase (Geometric Alignment)
        # phi = arg(<psi_v|psi_t>)
        phases = torch.atan2(imag_part, real_part + 1e-8)

        # 4. Derive Spectral Shift (η)
        # η = (1/π) arg{det(S)}. For diagonalized overlap, det(S) is product of overlaps.
        # arg(det(S)) = sum(arg(overlaps))
        eta = (1.0 / torch.pi) * torch.sum(phases, dim=-1)

        # 5. Loss: Minimize deflection (eta) and maximize overlap magnitude
        # We want the phase to be 0 (isomorphic) and magnitude to be 1.
        alignment_loss = torch.mean(eta**2)
        magnitude_loss = 1.0 - torch.mean(real_part)
        
        # Apply environmental drag factor
        total_loss = alignment_loss + (self.mu_e * magnitude_loss)
        
        return total_loss

class DiscreteDecisionEngine(nn.Module):
    """
    FIX: Resolved 'unexpected keyword argument dim' by explicitly defining __init__.
    """
    def __init__(self, dim: int = 256, **kwargs):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.gate(x))
