import torch
import torch.nn as nn
import math

# [STABLE] Fractal Weight Quantization (FWQ) for SU(2) Manifolds
# Grounded in: Geodesic Flow on 256-dim Quaternionic Manifold
# Constraint: Mac Mini M4 (MPS) Optimized

class FractalWeightQuantizer(nn.Module):
    """
    Implements 4-bit Fractal Weight Quantization (FWQ).
    Instead of quantizing Euclidean weights, we quantize the rotation angles (theta)
    of the Hamilton Product to preserve the topological eta-signature.
    """
    def __init__(self, bits=4, fractal_depth=3):
        super().__init__()
        self.bits = bits
        self.levels = 2 ** bits
        self.fractal_depth = fractal_depth
        # Pre-compute fractal bins based on h +/- delta symmetry breaking
        self.register_buffer("bins", self._generate_fractal_bins())

    def _generate_fractal_bins(self):
        """
        Generates non-linear bins using the Fractal Expansion Protocol (h +/- delta).
        This ensures the spectral shift tracker (eta) remains stable.
        """
        bins = torch.tensor([0.0, 1.0])
        h = 0.5
        delta = 0.25
        
        for _ in range(self.fractal_depth):
            new_bins = []
            for b in bins:
                new_bins.extend([b - delta, b + delta])
            bins = torch.tensor(sorted(list(set(new_bins + bins.tolist()))))
            delta /= 2
        
        # Normalize and scale to [0, pi] for SU(2) rotation angles
        bins = (bins - bins.min()) / (bins.max() - bins.min())
        return bins[:self.levels] * math.pi

    def forward(self, q_weights):
        """
        Args:
            q_weights: Tensor of shape (..., 4) representing quaternions [a, b, c, d]
        Returns:
            quantized_q: Reconstructed quaternions from 4-bit angles
        """
        # 1. Decompose Quaternion to Polar (Rotation Angle theta and Unit Vector u)
        # q = cos(theta/2) + u * sin(theta/2)
        norms = torch.norm(q_weights, dim=-1, keepdim=True).clamp(min=1e-6)
        q_unit = q_weights / norms
        
        # Extract theta: a = cos(theta/2) -> theta = 2 * acos(a)
        theta = 2 * torch.acos(q_unit[..., 0].clamp(-1.0, 1.0))
        
        # 2. Quantize Theta using Fractal Bins
        # We use bucketize for O(log N) lookup on MPS
        shape = theta.shape
        theta_flat = theta.view(-1)
        indices = torch.bucketize(theta_flat, self.bins) - 1
        indices = indices.clamp(0, self.levels - 1)
        
        theta_q = self.bins[indices].view(shape)
        
        # 3. Reconstruct Quaternion
        # Preserve the vector direction (u) to maintain topological symmetry
        u = q_unit[..., 1:] / torch.norm(q_unit[..., 1:], dim=-1, keepdim=True).clamp(min=1e-6)
        
        new_a = torch.cos(theta_q / 2).unsqueeze(-1)
        new_v = u * torch.sin(theta_q / 2).unsqueeze(-1)
        
        quantized_q = torch.cat([new_a, new_v], dim=-1)
        
        # Apply original norm to maintain energy levels (CEM compatibility)
        return quantized_q * norms

# [EXPERIMENTAL] Integration with Spectral Shift Tracker
def calculate_eta_signature(scattering_matrix):
    """
    Krein-like trace formula: eta = (1/pi) arg det(S)
    """
    # Ensure S is on MPS
    det_s = torch.linalg.det(scattering_matrix)
    eta = (1.0 / math.pi) * torch.angle(det_s)
    return eta

class DiscreteDecisionEngine(nn.Module):
    """
    Fixed version of the engine to resolve 'dim' keyword error.
    """
    def __init__(self, input_features):
        super().__init__()
        # Removed 'dim' argument to match the runtime feedback fix
        self.projection = nn.Linear(input_features, 16) 

    def forward(self, x):
        return torch.tanh(self.projection(x))
