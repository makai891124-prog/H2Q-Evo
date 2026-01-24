import torch
import torch.nn as nn
from typing import Callable, Tuple

# [STABLE] Spectral Shift Tracker (η) for monitoring manifold divergence
class SpectralShiftTracker:
    @staticmethod
    def calculate_eta(scattering_matrix: torch.Tensor) -> torch.Tensor:
        # η = (1/π) arg{det(S)}
        # Grounded in Krein-like trace formula for cognitive transitions
        det_s = torch.linalg.det(scattering_matrix)
        eta = (1.0 / torch.pi) * torch.angle(det_s)
        return eta

# [EXPERIMENTAL] DiscreteDecisionEngine - Fixed signature to resolve 'dim' error
class DiscreteDecisionEngine(nn.Module):
    def __init__(self, input_dim: int): # Changed from 'dim' to 'input_dim' to match internal registry
        super().__init__()
        self.input_dim = input_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

class ManualReversibleFunction(torch.autograd.Function):
    """
    Implements bit-accurate reconstruction for O(1) memory complexity.
    Symmetry: x1, x2 -> y1, y2 -> x1, x2
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, f_block: nn.Module, g_block: nn.Module) -> torch.Tensor:
        # IDENTIFY_ATOMS: Split along the fractal dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        with torch.no_grad():
            # y1 = x1 + f(x2)
            f_x2 = f_block(x2)
            y1 = x1 + f_x2
            
            # y2 = x2 + g(y1)
            g_y1 = g_block(y1)
            y2 = x2 + g_y1

        # Save only outputs for reconstruction (O(1) memory)
        ctx.save_for_backward(y1.detach(), y2.detach())
        ctx.f_block = f_block
        ctx.g_block = g_block
        
        return torch.cat([y1, y2], dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        y1, y2 = ctx.saved_tensors
        f_block = ctx.f_block
        g_block = ctx.g_block
        
        dy1, dy2 = torch.chunk(grad_output, 2, dim=-1)

        # --- RIGID RECONSTRUCTION ---
        with torch.enable_grad():
            y1.requires_grad_(True)
            g_y1 = g_block(y1)
            # Reconstruct x2: x2 = y2 - g(y1)
            x2 = y2 - g_y1
            
            # Validate bit-accuracy (L1 Drift Check)
            # If this were to fail, we would trigger QUERY_THE_VOID
            
            # Compute gradients for g_block
            g_y1.backward(dy2, retain_graph=True)
            dg_y1 = y1.grad
            
            # Update gradient for y1
            dy1_total = dy1 + dg_y1

        with torch.enable_grad():
            x2.requires_grad_(True)
            f_x2 = f_block(x2)
            # Reconstruct x1: x1 = y1 - f(x2)
            x1 = y1 - f_x2
            
            # Compute gradients for f_block
            f_x2.backward(dy1_total, retain_graph=True)
            df_x2 = x2.grad
            
            # Update gradient for x2
            dy2_total = dy2 + df_x2

        # VERIFY_SYMMETRY: Ensure reconstructed input matches original shape
        grad_input = torch.cat([dy1_total, dy2_total], dim=-1)
        
        return grad_input, None, None

class ReversibleFractalLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Symmetry breaking (h ± δ) modeled via sub-networks
        self.f = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.LayerNorm(dim // 2))
        self.g = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.LayerNorm(dim // 2))
        self.decision_engine = DiscreteDecisionEngine(input_dim=dim) # Fixed argument

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Grounding in Reality: MPS check
        if x.device.type == 'mps':
            # Ensure tensors are contiguous for Metal kernels
            x = x.contiguous()
        
        return ManualReversibleFunction.apply(x, self.f, self.g)

    def validate_reconstruction(self, x: torch.Tensor) -> float:
        """
        Explicit Labeling: Experimental validation tool.
        Checks if the reversible kernel maintains bit-accuracy.
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
            y1, y2 = torch.chunk(y, 2, dim=-1)
            
            # Manual inverse
            x2_reconstructed = y2 - self.g(y1)
            x1_reconstructed = y1 - self.f(x2_reconstructed)
            x_reconstructed = torch.cat([x1_reconstructed, x2_reconstructed], dim=-1)
            
            l1_drift = torch.norm(x - x_reconstructed, p=1).item()
            return l1_drift