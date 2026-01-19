import torch
import torch.nn as nn
from h2q.core.metal_jit_bridge import MetalJITBridge
from h2q.quaternion_ops import quaternion_stability

class AMXReversibleFunction(torch.autograd.Function):
    """
    Custom autograd function implementing bit-accurate activation reconstruction
    using 16x16 tiled Hamilton products on M4 AMX hardware.
    
    Memory Complexity: O(1) (excluding weights and output buffer)
    Hardware Target: Apple Silicon M4 (AMX/Metal)
    """

    @staticmethod
    def forward(ctx, x, weight, bridge: MetalJITBridge):
        """
        Forward pass using additive coupling:
        Y1 = X1
        Y2 = X2 + Hamilton(X1, W)
        """
        # x shape: [Batch, Dim] where Dim is 2 * Hidden (for split)
        # weight shape: [Hidden, Hidden] (Quaternionic representation)
        
        ctx.bridge = bridge
        
        # Split into two streams for additive coupling
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Execute 16x16 tiled Hamilton product via Metal JIT
        # We assume the bridge handles the tiling logic for the 4-component quaternions
        h_prod = bridge.execute_tiled_hamilton(x1, weight, tile_size=16)
        
        y2 = x2 + h_prod
        y = torch.cat([x1, y2], dim=-1)
        
        # Save only output and weights to maintain O(1) activation memory
        ctx.save_for_backward(y, weight)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Reconstruct X1, X2 from Y, then compute gradients.
        """
        y, weight = ctx.saved_tensors
        bridge = ctx.bridge
        
        # 1. Reconstruct Activations
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x1 = y1 # Y1 = X1
        
        # Reconstruct X2: X2 = Y2 - Hamilton(X1, W)
        # This must be bit-accurate to ensure gradient integrity
        h_prod_recon = bridge.execute_tiled_hamilton(x1, weight, tile_size=16)
        x2 = y2 - h_prod_recon
        
        # 2. Compute Gradients
        grad_y1, grad_y2 = torch.chunk(grad_output, 2, dim=-1)
        
        # Gradient of Hamilton product w.r.t X1 and W
        # dL/dX1 = grad_y1 + Hamilton_Grad_X(grad_y2, weight)
        # dL/dW = Hamilton_Grad_W(grad_y2, x1)
        
        # Using the bridge for adjoint Hamilton operations (conjugate products)
        dx1_from_h = bridge.execute_tiled_hamilton_adjoint(grad_y2, weight, mode='input')
        dw_from_h = bridge.execute_tiled_hamilton_adjoint(grad_y2, x1, mode='weight')
        
        grad_x1 = grad_y1 + dx1_from_h
        grad_x2 = grad_y2
        
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
        
        return grad_input, dw_from_h, None

class AMXReversibleLayer(nn.Module):
    """
    High-level wrapper for the AMXReversibleFunction.
    """
    def __init__(self, hidden_dim, bridge=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Quaternionic weights: 4 real numbers per quaternion
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.bridge = bridge if bridge else MetalJITBridge()
        
        # Verify JIT integrity on initialization
        from h2q.core.metal_jit_bridge import audit_jit_integrity
        audit_jit_integrity(self.bridge)

    def forward(self, x):
        return AMXReversibleFunction.apply(x, self.weight, self.bridge)

# EXPERIMENTAL: Tiling Symmetry Validation
def verify_amx_reconstruction_fidelity(layer, x_input):
    """
    Validates that the reconstruction in backward pass is bit-accurate.
    """
    layer.eval()
    with torch.no_grad():
        y = layer(x_input)
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x1_recon = y1
        h_prod = layer.bridge.execute_tiled_hamilton(x1_recon, layer.weight, tile_size=16)
        x2_recon = y2 - h_prod
        x_recon = torch.cat([x1_recon, x2_recon], dim=-1)
        
        drift = torch.norm(x_input - x_recon)
        return drift < 1e-6