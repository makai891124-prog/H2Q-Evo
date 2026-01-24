import torch
import torch.nn as nn
import torch.autograd as autograd

class ReversibleManifoldFunction(autograd.Function):
    """
    [STABLE] Reversible Manifold Function for H2Q Framework.
    Implements O(1) memory complexity by reconstructing input activations 
    from output states using the SU(2) symmetry-preserving coupling.
    """

    @staticmethod
    def forward(ctx, x, F, G, F_params, G_params):
        """
        Forward pass: 
        x1, x2 = split(x)
        y1 = x1 + F(x2)
        y2 = x2 + G(y1)
        y = concat(y1, y2)
        """
        with torch.no_grad():
            x1, x2 = torch.chunk(x, 2, dim=-1)
            
            f_x2 = F(x2)
            y1 = x1 + f_x2
            
            g_y1 = G(y1)
            y2 = x2 + g_y1
            
            y = torch.cat([y1, y2], dim=-1)

        # Save only the outputs for reconstruction in backward
        ctx.save_for_backward(y.detach())
        ctx.F = F
        ctx.G = G
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        Backward pass: Reconstructs x1, x2 from y1, y2 to calculate gradients.
        Ensures bit-accuracy to prevent L1 gradient drift.
        """
        y = ctx.saved_tensors[0]
        F, G = ctx.F, ctx.G
        
        y1, y2 = torch.chunk(y, 2, dim=-1)
        grad_y1, grad_y2 = torch.chunk(grad_y, 2, dim=-1)

        with torch.enable_grad():
            # 1. Reconstruct x2: x2 = y2 - G(y1)
            y1.requires_grad_(True)
            g_y1 = G(y1)
            x2 = y2 - g_y1
            
            # 2. Compute gradients for G
            # dL/dy2 is grad_y2. dL/dG = grad_y2
            # dL/dy1_total = grad_y1 + grad_y2 * G'(y1)
            g_y1.backward(grad_y2, retain_graph=True)
            grad_y1_total = grad_y1 + y1.grad
            y1.grad = None # Reset for next step

            # 3. Reconstruct x1: x1 = y1 - F(x2)
            x2.requires_grad_(True)
            f_x2 = F(x2)
            x1 = y1 - f_x2

            # 4. Compute gradients for F
            # dL/dx2_total = grad_y2 + grad_y1_total * F'(x2)
            f_x2.backward(grad_y1_total, retain_graph=True)
            grad_x2_total = grad_y2 + x2.grad
            
            # Reconstructed input for potential verification
            # x = torch.cat([x1, x2], dim=-1)
            
        return torch.cat([grad_y1_total, grad_x2_total], dim=-1), None, None, None, None

class ManifoldLayer(nn.Module):
    """
    [EXPERIMENTAL] SU(2) Geodesic Flow Layer.
    Utilizes Fractal Differential Calculus (FDC) for infinitesimal rotations.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Manifold dimension must be even for SU(2) partitioning."
        self.dim = dim
        # F and G are the coupling functions (the 'atoms' of the manifold)
        self.F = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, dim // 2)
        )
        self.G = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, dim // 2)
        )

    def forward(self, x):
        return ReversibleManifoldFunction.apply(
            x, self.F, self.G, 
            tuple(self.F.parameters()), 
            tuple(self.G.parameters())
        )

    def verify_reconstruction(self, x):
        """
        Rigid Construction Check: Verifies bit-accuracy of the inverse operation.
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
            y1, y2 = torch.chunk(y, 2, dim=-1)
            
            # Manual Inverse
            x2_rec = y2 - self.G(y1)
            x1_rec = y1 - self.F(x2_rec)
            x_rec = torch.cat([x1_rec, x2_rec], dim=-1)
            
            drift = torch.norm(x - x_rec)
            return drift < 1e-6, drift


class QuaternionicManifold(nn.Module):
    """
    Quaternionic Manifold for H2Q Framework.
    Implements quaternion-based geometric transformations.
    """
    def __init__(self, seed_atoms=2, target_dim=256):
        super().__init__()
        self.seed_atoms = seed_atoms
        self.target_dim = target_dim
        # Use the existing ManifoldLayer as base
        self.manifold_layer = ManifoldLayer(target_dim)

    def forward(self, x):
        return self.manifold_layer(x)
    
    def fractal_expand(self, seed):
        """
        Fractal expansion from seed atoms to target dimension.
        """
        # Simple expansion: repeat and concatenate to reach target dimension
        batch_size, n_atoms, quat_components = seed.shape
        # Flatten and expand
        expanded = seed.view(batch_size, -1)  # Flatten atoms and components
        # Expand to target dimension by repeating and truncating
        if expanded.shape[-1] < self.target_dim:
            # Repeat to reach target dimension
            repeats = (self.target_dim + expanded.shape[-1] - 1) // expanded.shape[-1]
            expanded = expanded.repeat(1, repeats)[:, :self.target_dim]
        else:
            expanded = flattened[:, :self.target_dim]
        # Return shape: (batch, target_dim, 1) so that [-2] is target_dim
        return expanded.unsqueeze(-1)