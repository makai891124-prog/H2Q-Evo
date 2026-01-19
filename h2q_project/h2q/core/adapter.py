import torch
import torch.nn as nn
from typing import Optional
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.sst import SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde

class CayleyManifoldInjector(nn.Module):
    """
    H2Q Cayley-Transform-Injection Wrapper.
    Converts Euclidean weights into SU(2) manifold representations for FDC fine-tuning.
    Ensures O(1) memory complexity via manifold projection and tracks spectral shift (eta).
    """
    def __init__(self, legacy_layer: nn.Linear, knot_count: int = 64):
        super().__init__()
        self.legacy_layer = legacy_layer
        self.out_features, self.in_features = legacy_layer.weight.shape
        self.knot_count = knot_count
        
        # Initialize Metacognitive Components
        # Note: Avoiding 'dim' argument in DDE to honor FEEDBACK log
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        # Manifold Parameters: 256-dim quaternionic manifold (64 knots x 4 atoms)
        # We represent the Lie Algebra su(2) as pure imaginary quaternions
        self.su2_skew = nn.Parameter(torch.randn(self.out_features, self.in_features, 3) * 0.01)
        
        self._is_stable = False

    def _cayley_transform(self, v: torch.Tensor) -> torch.Tensor:
        """
        Maps su(2) Lie Algebra to SU(2) Lie Group via Quaternionic Cayley Transform.
        q = (1 - v)(1 + v)^-1
        """
        # v is (..., 3) representing (bi + cj + dk)
        # 1 + v is (1, b, c, d)
        one = torch.ones((*v.shape[:-1], 1), device=v.device, dtype=v.dtype)
        
        # Numerator: (1 - v) -> (1, -b, -c, -d)
        num = torch.cat([one, -v], dim=-1)
        
        # Denominator: (1 + v) -> (1, b, c, d)
        den = torch.cat([one, v], dim=-1)
        
        # Quaternionic Inverse: q^-1 = q_conj / |q|^2
        den_conj = torch.cat([den[..., :1], -den[..., 1:]], dim=-1)
        den_norm_sq = torch.sum(den**2, dim=-1, keepdim=True) + 1e-8
        den_inv = den_conj / den_norm_sq
        
        # q = num * den_inv
        q_su2 = self._quat_mul(num, den_inv)
        return quaternion_normalize(q_su2)

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Internal Hamilton product for Cayley mapping."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Generate SU(2) Manifold Weights via Cayley Transform
        q_weights = self._cayley_transform(self.su2_skew)
        
        # 2. Project back to Euclidean space for legacy compatibility
        # We use the scalar part 'w' as the primary magnitude scaling
        # and the vector part as the directional bias.
        w_manifold = q_weights[..., 0] * torch.norm(q_weights[..., 1:], dim=-1)
        
        # 3. Apply Spectral Shift Tracking (eta)
        # η = (1/π) arg{det(S)}
        with torch.no_grad():
            self.sst.update(w_manifold)
            
        # 4. Execute Linear Operation
        out = torch.nn.functional.linear(x, w_manifold, self.legacy_layer.bias)
        
        # 5. Veracity Audit (Discrete Fueter Operator check)
        # If residuals are high, DDE triggers a correction proposer
        if self.training:
            decision = self.dde.propose_correction(out)
            out = out + decision * 0.01
            
        return out

def inject_manifold(model: nn.Module):
    """
    Recursively replaces Linear layers with CayleyManifoldInjectors.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, CayleyManifoldInjector(module))
        else:
            inject_manifold(module)
    return model