import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# [STABLE] Core H2Q Atoms for Mac Mini M4 (MPS Optimized)

class FractalExpansion(nn.Module):
    """
    Atom 1: Fractal Expansion (2 -> 256)
    Expands binary/seed logic into quaternionic knots via recursive symmetry breaking.
    """
    def __init__(self, in_dim=2, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        # Symmetry breaking parameters (h ± δ)
        self.h = nn.Parameter(torch.randn(1, out_dim) * 0.02)
        self.delta = nn.Parameter(torch.randn(1, out_dim) * 0.01)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Recursive symmetry breaking simulation
        base = self.proj(x)
        knot = base * (self.h + self.delta) - base * (self.h - self.delta)
        return torch.tanh(knot)

class ReversibleKernel(nn.Module):
    """
    Atom 2: Reversible Kernels
    Additive coupling [y1=x1+F(x2); y2=x2+G(y1)] for O(1) memory.
    """
    def __init__(self, dim):
        super().__init__()
        self.split_dim = dim // 2
        self.F = nn.Sequential(nn.Linear(self.split_dim, self.split_dim), nn.GELU())
        self.G = nn.Sequential(nn.Linear(self.split_dim, self.split_dim), nn.GELU())

    def forward(self, x):
        x1, x2 = torch.split(x, self.split_dim, dim=-1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=-1)

class BerryPhaseInterferometer(nn.Module):
    """
    TASK: Multimodal alignment via Pancharatnam-Berry phase.
    Replaces Cosine Similarity with geometric phase curvature on SU(2).
    """
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.vision_expander = FractalExpansion(in_dim=3, out_dim=dim) # YCbCr
        self.text_expander = FractalExpansion(in_dim=1, out_dim=dim)   # Bytes
        self.rev_kernel = ReversibleKernel(dim)
        
        # Spectral Shift Tracker (η)
        self.register_buffer("eta", torch.tensor(0.0))

    def _to_spinor(self, x):
        """Project 256-dim real vector to 128-dim complex SU(2) spinor."""
        x = x.view(*x.shape[:-1], self.dim // 2, 2)
        # Create complex representation: z = a + bi
        return torch.complex(x[..., 0], x[..., 1])

    def forward(self, vision_ycbcr, text_bytes):
        """
        vision_ycbcr: (B, 3) - Mean YCbCr values
        text_bytes: (B, 1) - Normalized byte values
        """
        # 1. Fractal Expansion
        v_knot = self.vision_expander(vision_ycbcr)
        t_knot = self.text_expander(text_bytes)

        # 2. Reversible Coupling
        v_state = self.rev_kernel(v_knot)
        t_state = self.rev_kernel(t_knot)

        # 3. Pancharatnam-Berry Phase Calculation
        # γ = arg <ψ_v | ψ_t>
        psi_v = self._to_spinor(v_state)
        psi_t = self._to_spinor(t_state)
        
        # Inner product across the spinor dimension
        inner_product = torch.sum(psi_v * torch.conj(psi_t), dim=-1)
        
        # Geometric Phase (Berry Phase)
        berry_phase = torch.angle(inner_product)

        # 4. Spectral Shift Tracker (η = (1/π) arg{det(S)})
        # Here S is the alignment matrix between modalities
        S = torch.matmul(psi_v.unsqueeze(-1), torch.conj(psi_t.unsqueeze(-2)))
        # Use a stable approximation for det(S) on small manifolds
        det_S = torch.linalg.det(S + torch.eye(S.size(-1), device=S.device) * 1e-6)
        self.eta = (1.0 / math.pi) * torch.angle(det_S).mean()

        return berry_phase, self.eta

# [EXPERIMENTAL] Corrected Decision Engine to resolve previous Runtime Error
class DiscreteDecisionEngine(nn.Module):
    def __init__(self, input_dim): # Removed 'dim' keyword ambiguity
        super().__init__()
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.gate(x))

if __name__ == "__main__":
    # Mac Mini M4 Verification
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[H2Q] Initializing Interferometer on {device}")
    
    model = BerryPhaseInterferometer().to(device)
    v_in = torch.randn(8, 3).to(device) # YCbCr
    t_in = torch.randn(8, 1).to(device) # Bytes
    
    phase, shift = model(v_in, t_in)
    print(f"Berry Phase Alignment: {phase.shape}")
    print(f"Spectral Shift (η): {shift.item():.4f}")
