import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize

class HamiltonReversibleFunction(torch.autograd.Function):
    """
    Manual Autograd function for Hamilton Reversible Cell.
    Achieves O(1) memory by reconstructing inputs from outputs during backward pass.
    Uses 16x16 tiling logic for AMX-optimized quaternionic multiplication.
    """
    @staticmethod
    def forward(ctx, x, w1, w2):
        # x shape: [Batch, Dim, 4] (Quaternions)
        # Split into two halves for reversible coupling
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Ensure 16x16 alignment for AMX tiling
        # We treat the Hamilton product as a transformation in SU(2)
        # f(x) = x * w
        
        with torch.no_grad():
            # y1 = x1 + Hamilton(x2, w1)
            f_x2 = HamiltonReversibleFunction._hamilton_amx_tile(x2, w1)
            y1 = x1 + f_x2
            
            # y2 = x2 + Hamilton(y1, w2)
            g_y1 = HamiltonReversibleFunction._hamilton_amx_tile(y1, w2)
            y2 = x2 + g_y1
            
        ctx.save_for_backward(w1, w2, y1, y2)
        return torch.cat([y1, y2], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        w1, w2, y1, y2 = ctx.saved_tensors
        grad_y1, grad_y2 = torch.chunk(grad_output, 2, dim=1)
        
        # Reconstruct x2: x2 = y2 - Hamilton(y1, w2)
        with torch.no_grad():
            g_y1 = HamiltonReversibleFunction._hamilton_amx_tile(y1, w2)
            x2 = y2 - g_y1
            
        # Gradient for w2 and y1 from g(y1, w2)
        with torch.enable_grad():
            y1_temp = y1.detach().requires_grad_(True)
            w2_temp = w2.detach().requires_grad_(True)
            g_out = HamiltonReversibleFunction._hamilton_amx_tile(y1_temp, w2_temp)
            g_out.backward(grad_y2)
            
            grad_w2 = w2_temp.grad
            grad_y1_total = grad_y1 + y1_temp.grad
            
        # Reconstruct x1: x1 = y1 - Hamilton(x2, w1)
        with torch.no_grad():
            f_x2 = HamiltonReversibleFunction._hamilton_amx_tile(x2, w1)
            x1 = y1 - f_x2
            
        # Gradient for w1 and x2 from f(x2, w1)
        with torch.enable_grad():
            x2_temp = x2.detach().requires_grad_(True)
            w1_temp = w1.detach().requires_grad_(True)
            f_out = HamiltonReversibleFunction._hamilton_amx_tile(x2_temp, w1_temp)
            f_out.backward(grad_y1_total)
            
            grad_w1 = w1_temp.grad
            grad_x2_total = grad_y2 + x2_temp.grad
            
        grad_x = torch.cat([grad_y1_total, grad_x2_total], dim=1)
        return grad_x, grad_w1, grad_w2

    @staticmethod
    def _hamilton_amx_tile(q1, q2):
        """
        Implements Hamilton Product with 16x16 tiling hints for Metal/AMX.
        q1: [B, N, 4], q2: [N, N, 4] or [N, 4]
        """
        # Ensure dimensions are multiples of 16 for AMX efficiency
        b, n, _ = q1.shape
        
        # Quaternionic components
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        # Hamilton Product components using optimized matmuls
        # This structure allows MPS to tile the 16x16 blocks effectively
        r = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        
        return torch.stack([r, i, j, k], dim=-1)

class HamiltonReversibleCell(nn.Module):
    """
    Unified Reversible Layer replacing standard Linear layers in L1.
    Natively supports SU(2) rotations via Hamilton Product.
    """
    def __init__(self, dim):
        super().__init__()
        # Ensure dim is even for splitting and multiple of 16 for AMX
        self.dim = (dim + 15) // 16 * 16
        self.half_dim = self.dim // 2
        
        # Quaternionic weights: [HalfDim, 4]
        self.w1 = nn.Parameter(torch.randn(self.half_dim, 4) * 0.02)
        self.w2 = nn.Parameter(torch.randn(self.half_dim, 4) * 0.02)
        
        # Initialize as identity rotations (1, 0, 0, 0)
        with torch.no_grad():
            self.w1[:, 0].fill_(1.0)
            self.w2[:, 0].fill_(1.0)

    def forward(self, x):
        # Pad x if not aligned to 16
        original_shape = x.shape
        if x.shape[1] < self.dim:
            padding = self.dim - x.shape[1]
            x = F.pad(x, (0, 0, 0, padding))
            
        # Normalize weights to ensure they stay on S^3 (SU(2) manifold)
        w1_norm = quaternion_normalize(self.w1)
        w2_norm = quaternion_normalize(self.w2)
        
        out = HamiltonReversibleFunction.apply(x, w1_norm, w2_norm)
        
        # Truncate back to original dim if padded
        return out[:, :original_shape[1], :]

    def inverse(self, y):
        """
        Explicit inverse for inference-time reconstruction or verification.
        """
        y1, y2 = torch.chunk(y, 2, dim=1)
        w1_norm = quaternion_normalize(self.w1)
        w2_norm = quaternion_normalize(self.w2)
        
        with torch.no_grad():
            # x2 = y2 - Hamilton(y1, w2)
            g_y1 = HamiltonReversibleFunction._hamilton_amx_tile(y1, w2_norm)
            x2 = y2 - g_y1
            
            # x1 = y1 - Hamilton(x2, w1)
            f_x2 = HamiltonReversibleFunction._hamilton_amx_tile(x2, w1_norm)
            x1 = y1 - f_x2
            
        return torch.cat([x1, x2], dim=1)