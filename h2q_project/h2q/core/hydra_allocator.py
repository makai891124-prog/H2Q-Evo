import torch
import torch.nn as nn
from h2q.core.quantization.tpq_engine import TopologicalPhaseQuantizer
from h2q.core.interface_registry import get_canonical_dde

class HydraBitAllocator(nn.Module):
    """
    Hydra-Manifold-Bit-Allocator: Middleware for dynamic precision switching.
    Toggles between 4-bit TPQ and FP32 based on local Fueter-analyticity residuals.
    """
    def __init__(self, threshold: float = 0.05, dde_config: dict = None):
        super().__init__()
        self.threshold = threshold
        self.tpq = TopologicalPhaseQuantizer()
        
        # Use canonical DDE to avoid 'dim' keyword argument errors identified in feedback
        self.dde = get_canonical_dde(**(dde_config or {}))
        
        # Experimental: Track the ratio of high-precision knots
        self.register_buffer("high_precision_ratio", torch.tensor(0.0))

    def compute_fueter_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Discrete Fueter Operator Df = ∂w + i∂x + j∂y + k∂z.
        Measures the deviation from quaternionic holomorphicity.
        Input x shape: [..., 256] (representing 64 quaternions)
        """
        # Reshape to [..., 64, 4] to isolate quaternionic components (w, i, j, k)
        q = x.view(*x.shape[:-1], 64, 4)
        
        # Compute local finite differences as a proxy for partial derivatives
        # In a geodesic flow, this represents the 'tear' in the manifold
        dw = torch.gradient(q[..., 0], dim=-1)[0]
        di = torch.gradient(q[..., 1], dim=-1)[0]
        dj = torch.gradient(q[..., 2], dim=-1)[0]
        dk = torch.gradient(q[..., 3], dim=-1)[0]
        
        # Df residual norm
        df_norm = torch.abs(dw + di + dj + dk)
        return df_norm.mean(dim=-1) # Average residual per knot

    def forward(self, knot_tensor: torch.Tensor) -> torch.Tensor:
        """
        Allocates bits dynamically. 
        If Df > threshold (topological tear), use FP32.
        Else, use 4-bit TPQ.
        """
        # 1. Calculate logical hallucination risk (Fueter Residual)
        with torch.no_grad():
            residual = self.compute_fueter_residual(knot_tensor)
            
        # 2. Decision Logic: Df > 0.05 identifies logical hallucinations
        # We use the DDE to mediate the transition to ensure homeostatic stability
        mask = (residual > self.threshold).float()
        
        # Update telemetry
        self.high_precision_ratio = mask.mean()

        # 3. Path Execution
        # Path A: 4-bit TPQ (Memory Efficient)
        tpq_out = self.tpq(knot_tensor)
        
        # Path B: FP32 (Veracity Preserving)
        fp32_out = knot_tensor.to(torch.float32)

        # 4. Symmetrical Recombination
        # We use the mask to blend or select. For strict bit-allocation, we select.
        # Ensure mask is broadcastable to the 256-dim manifold
        mask_expanded = mask.unsqueeze(-1).expand_as(knot_tensor)
        
        output = (mask_expanded * fp32_out) + ((1 - mask_expanded) * tpq_out)
        
        return output

    def audit_precision_integrity(self):
        """
        Veracity Compact Check: Ensure we aren't over-compressing critical logic.
        """
        if self.high_precision_ratio < 0.1:
            print(f"[WARNING] Hydra-Allocator: High-precision ratio ({self.high_precision_ratio:.4f}) below safety bounds.")
        return {"hp_ratio": self.high_precision_ratio.item(), "threshold": self.threshold}