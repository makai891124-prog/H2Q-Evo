import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TPQv2STE(torch.autograd.Function):
    """
    Straight-Through Estimator for 4-bit Phase Quantization on SU(2).
    Maps quaternionic phase to 16 discrete levels while preserving gradients.
    """
    @staticmethod
    def forward(ctx, q, levels=16):
        # q shape: (..., 4) representing (w, x, y, z)
        # Ensure unit quaternion for manifold integrity
        q = F.normalize(q, p=2, dim=-1)
        
        w = q[..., 0].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        v = q[..., 1:]
        
        # Extract phase theta in [0, pi]
        theta = torch.acos(w)
        
        # Quantize theta to 4-bit (16 levels)
        step = math.pi / (levels - 1)
        theta_q = torch.round(theta / step) * step
        
        # Reconstruct quantized quaternion
        # sin(theta_q) * (v / |v|)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        
        w_q = torch.cos(theta_q).unsqueeze(-1)
        v_q = torch.sin(theta_q).unsqueeze(-1) * (v / v_norm)
        
        q_q = torch.cat([w_q, v_q], dim=-1)
        
        # Save for backward if needed, though STE usually passes grad directly
        return q_q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: Identity mapping for gradients
        return grad_output, None

class TPQv2Kernel(nn.Module):
    """
    Topological Phase Quantizer (v2) with QAT support.
    Implements 4-bit phase quantization within the SU(2) manifold.
    """
    def __init__(self, levels=16):
        super().__init__()
        self.levels = levels
        self.tracker = SpectralShiftTracker()

    def forward(self, q):
        """
        Args:
            q: Quaternionic tensor of shape (..., 4)
        Returns:
            q_quant: Quantized quaternionic tensor
        """
        if self.training:
            # Apply STE for Quantization-Aware Training
            q_quant = TPQv2STE.apply(q, self.levels)
        else:
            # Standard inference quantization
            with torch.no_grad():
                q_quant = TPQv2STE.apply(q, self.levels)
        
        # Calculate Spectral Shift (eta) to monitor information persistence
        # S is treated as the transition matrix between continuous and quantized states
        # In this context, we approximate S via the inner product of q and q_quant
        S = torch.matmul(q.transpose(-1, -2), q_quant)
        eta = self.tracker.compute_eta(S)
        
        return q_quant

class SpectralShiftTracker(nn.Module):
    """Monitoring tool for cognitive transitions in the manifold."""
    def compute_eta(self, S):
        # η = (1/π) arg{det(S)}
        # Using MPS-safe complex determinant approximation
        # S is expected to be (..., 4, 4) or (..., 2, 2) in complex representation
        # Here we use the trace-based approximation for SU(2) stability
        det_s = torch.linalg.det(S.to(torch.complex64)) if S.shape[-1] == S.shape[-2] else torch.tensor(1.0)
        eta = (1.0 / math.pi) * torch.angle(det_s)
        return eta

class DiscreteDecisionEngine(nn.Module):
    """
    Updated to match H2Q Global Interface Registry.
    Corrected __init__ signature to avoid 'dim' keyword error.
    """
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.proj = nn.Linear(input_features, output_features)

    def forward(self, x):
        return torch.softmax(self.proj(x), dim=-1)

class ReversibleTPQ(torch.autograd.Function):
    """
    Legacy support for reversible operations within the TPQ pipeline.
    """
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x * weight

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        return grad_output * weight, grad_output * x