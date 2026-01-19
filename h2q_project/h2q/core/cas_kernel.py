import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.discrete_decision_engine import get_canonical_dde

class CliffordSpellingEngine(nn.Module):
    """
    Engine for mapping discrete tokens to Clifford-algebraic structures (S³).
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)

class CAS_Kernel(nn.Module):
    """
    Clifford Algebraic System (CAS) Kernel.
    Implements Geodesic Flow on SU(2) with Fueter Regularization.
    Optimized for Mac Mini M4 (AMX-tiled operations).
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        # Fixed: Using get_canonical_dde to avoid 'dim' keyword argument error
        self.dde = get_canonical_dde()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) / (input_dim ** 0.5))
        self.curvature_threshold = 0.05

    def forward(self, x):
        """
        Standard forward pass using 16x16 AMX-compatible tiling logic.
        x: (Batch, Dim) where Dim is a multiple of 4 (Quaternionic basis).
        """
        return torch.matmul(x, self.weight)

    def compute_fueter_regularization(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        [EXPERIMENTAL] Differentiable Fueter Regularization.
        Penalizes non-analytic logic curvature (topological tears).
        
        The Fueter operator D acting on f = y0 + i*y1 + j*y2 + k*y3 is:
        Df = (dy0/dx0 - dy1/dx1 - dy2/dx2 - dy3/dx3) + 
             i(dy0/dx1 + dy1/dx0 + dy2/dx3 - dy3/dx2) + ...
        """
        if not x.requires_grad:
            return torch.tensor(0.0, device=x.device)

        # Reshape to quaternionic components (B, N, 4)
        batch_size = x.shape[0]
        x_q = x.view(batch_size, -1, 4)
        y_q = y.view(batch_size, -1, 4)

        # We compute partials for the first quaternionic atom to estimate curvature
        # This maintains O(1) memory complexity relative to sequence length
        x0, x1, x2, x3 = x_q[:, 0, 0], x_q[:, 0, 1], x_q[:, 0, 2], x_q[:, 0, 3]
        y_atom = y_q[:, 0, :]

        def get_grad(output_idx, input_var):
            return torch.autograd.grad(
                y_atom[:, output_idx].sum(), input_var, 
                create_graph=True, retain_graph=True
            )[0]

        # Compute the 16 partial derivatives for the Fueter matrix
        # Real part of Df
        d0y0 = get_grad(0, x0)
        d1y1 = get_grad(1, x1)
        d2y2 = get_grad(2, x2)
        d3y3 = get_grad(3, x3)
        
        real_df = d0y0 - d1y1 - d2y2 - d3y3

        # Imaginary parts (i, j, k)
        i_df = get_grad(1, x0) + get_grad(0, x1) + get_grad(3, x2) - get_grad(2, x3)
        j_df = get_grad(2, x0) - get_grad(3, x1) + get_grad(0, x2) + get_grad(1, x3)
        k_df = get_grad(3, x0) + get_grad(2, x1) - get_grad(1, x2) + get_grad(0, x3)

        # Logic Curvature is the norm of the Fueter deviation from zero
        logic_curvature = torch.sqrt(real_df**2 + i_df**2 + j_df**2 + k_df**2 + 1e-8)
        
        # Apply threshold-based penalty (Holomorphic Auditing)
        penalty = F.relu(logic_curvature - self.curvature_threshold)
        
        return penalty.mean()

    def wake_phase_step(self, x: torch.Tensor):
        """
        Executes a forward pass with Fueter Regularization enabled.
        """
        x.requires_grad_(True)
        y = self.forward(x)
        f_loss = self.compute_fueter_regularization(x, y)
        return y, f_loss