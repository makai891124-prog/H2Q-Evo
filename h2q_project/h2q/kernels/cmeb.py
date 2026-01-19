import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.utils.mps_compat import mps_safe_det
from h2q.quaternion_ops import quaternion_normalize

class CrossModalEntropyBalancer(nn.Module):
    """
    CMEB: Synchronizes the Heat-Death Index (Spectral Entropy) between 
    Vision (YCbCr) and Text (Byte-stream) manifolds.
    
    Governed by Rigid Construction: Symmetry between modalities is enforced
    via additive reversible coupling conditioned on the entropy gap.
    """
    def __init__(self, dim=256, epsilon=1e-6, fractal_delta=0.01):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.h_delta = fractal_delta
        
        # Transformation kernels for reversible coupling
        self.phi = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.psi = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def calculate_hdi(self, manifold_tensor):
        """
        Calculates the Heat-Death Index (Spectral Entropy).
        HDI = -sum(p * log(p)) where p are normalized singular values.
        """
        # Ensure we are working with a 2D matrix for SVD
        # manifold_tensor shape: [B, 256] -> treat as [B, 16, 16] for spectral analysis
        b = manifold_tensor.shape[0]
        matrix = manifold_tensor.view(b, 16, 16)
        
        # SVD is stable on MPS for small matrices
        s = torch.linalg.svdvals(matrix)
        
        # Normalize singular values to form a probability distribution
        p = s / (torch.sum(s, dim=-1, keepdim=True) + self.epsilon)
        entropy = -torch.sum(p * torch.log(p + self.epsilon), dim=-1)
        
        # Normalize by max entropy (log of dimension)
        hdi = entropy / torch.log(torch.tensor(16.0, device=manifold_tensor.device))
        return hdi

    def calculate_eta(self, manifold_tensor):
        """
        Spectral Shift Tracker (η) via Krein-like trace formula.
        η = (1/π) arg{det(S)}
        """
        b = manifold_tensor.shape[0]
        # Map to complex SU(2) representation (simplified 2x2 block)
        # Using the first 4 elements as a quaternion [a, b, c, d] -> [[a+bi, c+di], [-c+di, a-bi]]
        q = manifold_tensor[:, :4]
        
        # Construct 2x2 complex matrix
        real_part = torch.stack([
            torch.stack([q[:, 0], q[:, 2]], dim=-1),
            torch.stack([-q[:, 2], q[:, 0]], dim=-1)
        ], dim=-2)
        
        imag_part = torch.stack([
            torch.stack([q[:, 1], q[:, 3]], dim=-1),
            torch.stack([q[:, 3], -q[:, 1]], dim=-1)
        ], dim=-2)
        
        s_matrix = torch.complex(real_part, imag_part)
        
        # Use MPS safe determinant
        det_s = mps_safe_det(s_matrix)
        eta = (1.0 / torch.pi) * torch.angle(det_s)
        return eta

    def inject_fractal_noise(self, x, h):
        """
        Fractal Noise Injection (h ± δ) to prevent manifold collapse.
        """
        noise = torch.randn_like(x) * self.h_delta * h.unsqueeze(-1)
        return x + noise

    def forward(self, vision_manifold, text_manifold):
        """
        Synchronizes manifolds using a reversible additive coupling layer
        modulated by the entropy gradient between modalities.
        """
        # 1. Compute Heat-Death Indices
        hdi_v = self.calculate_hdi(vision_manifold) # [B]
        hdi_t = self.calculate_hdi(text_manifold)   # [B]
        
        # 2. Calculate Entropy Gap
        # If gap is positive, vision is more 'ordered' (lower entropy) than text
        gap = hdi_t - hdi_v
        
        # 3. Reversible Coupling Step (O(1) Memory Logic)
        # v_next = v + phi(t) * scale
        # t_next = t + psi(v_next) * scale
        
        # Scale coupling by the entropy gap to force synchronization
        sync_scale = torch.sigmoid(gap).unsqueeze(-1)
        
        v_mid = vision_manifold + self.phi(text_manifold) * sync_scale
        t_out = text_manifold + self.psi(v_mid) * (1.0 - sync_scale)
        v_out = v_mid

        # 4. Dimensional Integrity Check (Fractal Injection)
        # If HDI falls below critical threshold (0.2), inject noise
        critical_mask = (hdi_v < 0.2) | (hdi_t < 0.2)
        if critical_mask.any():
            v_out = self.inject_fractal_noise(v_out, 1.0 - hdi_v)
            t_out = self.inject_fractal_noise(t_out, 1.0 - hdi_t)

        # 5. Verify Symmetry (Rigid Construction)
        v_out = quaternion_normalize(v_out)
        t_out = quaternion_normalize(t_out)

        return v_out, t_out, {
            "hdi_vision": hdi_v.mean().item(),
            "hdi_text": hdi_t.mean().item(),
            "spectral_shift_v": self.calculate_eta(v_out).mean().item()
        }

    def inverse(self, v_out, t_out, sync_scale):
        """
        Reconstructs inputs from outputs (Manual Reversible Kernel).
        """
        # t = t_out - psi(v_out) * (1 - scale)
        # v = v_out - phi(t) * scale
        t_mid = t_out - self.psi(v_out) * (1.0 - sync_scale)
        v_in = v_out - self.phi(t_mid) * sync_scale
        return v_in, t_mid
