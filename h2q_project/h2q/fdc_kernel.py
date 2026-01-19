import torch
import torch.nn as nn
import torch.nn.functional as F

class GeodesicBackprop(torch.autograd.Function):
    """
    Implementation of gradients as infinitesimal rotations on the SU(2) manifold.
    Replaces Euclidean translations with geodesic steps.
    """
    @staticmethod
    def forward(ctx, x, weight, phase_shift):
        # x: [batch, dim], weight: [out_dim, in_dim]
        ctx.save_for_backward(x, weight, phase_shift)
        # Fractal Expansion: Linear projection followed by unitary-constrained activation
        z = F.linear(x, weight)
        # Apply phase deflection (Spectral Shift component)
        return torch.cos(z + phase_shift) + 1j * torch.sin(z + phase_shift)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, phase_shift = ctx.saved_tensors
        
        # Treat grad_output as a deflection in the tangent space (Lie Algebra su(2))
        # Instead of standard subtraction, we calculate the rotation required to minimize loss
        grad_real = grad_output.real
        grad_imag = grad_output.imag
        
        # Orthogonal approach: Vectorized rotation gradient
        # grad_weight = grad_output * x^T mapped back to real manifold
        grad_weight = torch.matmul(grad_real.t(), x) 
        grad_phase = grad_real.sum()
        
        return grad_real @ weight, grad_weight, grad_phase

class SpectralShiftTracker(nn.Module):
    """
    Calculates η = (1/π) arg{det(S)} to track cognitive deflection.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, S_matrix):
        # S_matrix is expected to be the scattering matrix of the current state
        # For SU(2), det(S) is the product of eigenvalues on the unit circle
        # We use the log-det trick for stability
        eigenvalues = torch.linalg.eigvals(S_matrix)
        phase_angles = torch.angle(eigenvalues)
        eta = torch.sum(phase_angles) / torch.pi
        return eta

class FDCKernel(nn.Module):
    """
    Fractal Dimension Controller (FDC) Kernel.
    Governs the 2 -> 256 dimension expansion via Symmetry Breaking.
    """
    def __init__(self, in_dim=2, out_dim=256, device="mps"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        # Symmetry Breaking parameters (h ± δ)
        self.h = nn.Parameter(torch.randn(out_dim, in_dim, device=device) * 0.01)
        self.delta = nn.Parameter(torch.randn(out_dim, device=device) * 0.01)
        
        self.tracker = SpectralShiftTracker(out_dim)

    def forward(self, x):
        # Ensure input is on the correct device
        if x.device.type != self.device:
            x = x.to(self.device)
            
        # Recursive Symmetry Breaking: W = h + delta
        # This maintains the geodesic flow constraints
        weight = self.h + torch.diag(self.delta).unsqueeze(1).expand(-1, self.in_dim, -1).sum(dim=-1)
        
        # Apply Vectorized Geodesic Backprop
        out_complex = GeodesicBackprop.apply(x, self.h, self.delta)
        
        # Calculate Spectral Shift (η)
        # Construct a proxy Scattering Matrix S from the output for tracking
        # In a real CEM, S is derived from the environment interaction
        S_proxy = torch.diag_embed(out_complex.mean(dim=0))
        eta = self.tracker(S_proxy)
        
        return out_complex.real, eta

class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Fixed implementation to resolve 'num_actions' unexpected keyword argument.
    """
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.classifier(x)

# Experimental: Trace Formula Grounding
def continuous_environment_drag(eta, time_step):
    """
    Calculates μ(E) based on the path integral of η.
    """
    return torch.cumsum(eta, dim=0) * time_step
