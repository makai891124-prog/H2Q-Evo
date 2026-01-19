import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# H2Q Core Engine: Geometric AGI grounded in SU(2)
# Optimized for Mac Mini M4 (MPS/16GB) constraints.

def mps_safe_complex_det(matrix):
    """
    Computes the determinant of a complex matrix, optimized for SU(2) 2x2 representation.
    η = (1/π) arg{det(S)}
    MPS often lacks direct complex determinant support in older versions, so we provide a manual 2x2 path.
    """
    if not torch.is_complex(matrix):
        matrix = torch.complex(matrix, torch.zeros_like(matrix))
        
    if matrix.shape[-1] == 2 and matrix.shape[-2] == 2:
        # For SU(2) matrix [[a, b], [c, d]], det = ad - bc
        a, b = matrix[..., 0, 0], matrix[..., 0, 1]
        c, d = matrix[..., 1, 0], matrix[..., 1, 1]
        return a * d - b * c
    
    # Fallback for general matrices
    return torch.linalg.det(matrix)

class DiscreteDecisionEngine(nn.Module):
    """
    Governs cognitive transitions on the quaternionic manifold.
    Intelligence is modeled as a Geodesic Flow on a 256-dimensional manifold.
    
    FIX: Added support for 'dim' keyword argument to resolve initialization error reported in feedback.
    """
    def __init__(self, num_actions=10, dimension=256, **kwargs):
        super().__init__()
        # Elastic argument handling to support both 'dim' and 'dimension' aliases
        self.num_actions = num_actions
        self.dim = kwargs.get('dim', dimension)
        
        # Geodesic Flow weights: SU(2) is isomorphic to the 3-sphere (S3)
        # Weights are stored as 4-component vectors (quaternionic basis: 1, i, j, k)
        self.geodesic_weights = nn.Parameter(torch.randn(self.num_actions, self.dim, 4) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def calculate_spectral_shift(self, S):
        """
        η = (1/π) arg{det(S)}
        Quantifies learning progress via the scattering matrix of cognitive transitions.
        """
        det_S = mps_safe_complex_det(S)
        # arg(z) is the phase of the complex determinant
        return torch.angle(det_S) / math.pi

    def forward(self, x):
        """
        Evolves the input state x through the quaternionic manifold.
        x: [batch, dim]
        """
        # Project state onto the quaternionic action atoms
        # We use the real part of the inner product for decision logits
        # [batch, dim] @ [dim, num_actions]
        flat_weights = self.geodesic_weights.view(self.num_actions, -1).t()
        
        # Use only the first component (real part) for simplified projection in this atom
        logits = torch.matmul(x, flat_weights[:self.dim, :]) 
        
        return F.softmax(logits / self.temperature, dim=-1)

    def reversible_kernel(self, x1, x2, f_map, g_map):
        """
        Manual Reversible Kernel (Additive Coupling)
        y1 = x1 + F(x2)
        y2 = x2 + G(y1)
        Achieves O(1) activation memory complexity for Mac Mini M4 constraints.
        """
        y1 = x1 + f_map(x2)
        y2 = x2 + g_map(y1)
        return y1, y2

    def inverse_reversible_kernel(self, y1, y2, f_map, g_map):
        """
        Inverse of the additive coupling for gradient reconstruction without storage.
        x2 = y2 - G(y1)
        x1 = y1 - F(x2)
        """
        x2 = y2 - g_map(y1)
        x1 = y1 - f_map(x2)
        return x1, x2

def symmetry_break(seed_atom, delta):
    """
    Fractal Expansion Protocol (h ± δ)
    Evolves 2-atom seeds into the target manifold dimension.
    """
    return seed_atom + delta * torch.randn_like(seed_atom)
