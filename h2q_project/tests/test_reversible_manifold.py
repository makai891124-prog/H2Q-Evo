import torch
import torch.nn as nn
from torch.autograd import Function
import unittest

# [STABLE] Reversible Kernel for O(1) Memory Complexity
class ReversibleManifoldFunction(Function):
    """
    Implements a reversible coupling layer for the 256-dimensional manifold.
    Reconstructs input states from outputs during backpropagation to save memory.
    """
    @staticmethod
    def forward(ctx, x, f_block, g_block):
        # Rigid Construction: Split 256-dim manifold into two 128-dim atoms
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        with torch.no_grad():
            # y1 = x1 + f(x2)
            f_x2 = f_block(x2)
            y1 = x1 + f_x2
            # y2 = x2 + g(y1)
            g_y1 = g_block(y1)
            y2 = x2 + g_y1
            
        ctx.save_for_backward(y1.detach(), y2.detach())
        ctx.f_block = f_block
        ctx.g_block = g_block
        return torch.cat([y1, y2], dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        y1, y2 = ctx.saved_tensors
        f_block = ctx.f_block
        g_block = ctx.g_block
        
        grad_y1, grad_y2 = torch.chunk(grad_output, 2, dim=-1)
        
        with torch.enable_grad():
            y1.requires_grad = True
            g_y1 = g_block(y1)
            # Reconstruct x2: x2 = y2 - g(y1)
            x2 = y2 - g_y1
            
            # Gradient of g_block
            g_y1.backward(grad_y2, retain_graph=True)
            grad_x2 = grad_y2 + y1.grad
            y1.grad = None
            
            x2.requires_grad = True
            f_x2 = f_block(x2)
            # Reconstruct x1: x1 = y1 - f(x2)
            x1 = y1 - f_x2
            
            # Gradient of f_block
            f_x2.backward(grad_y1, retain_graph=True)
            grad_x1 = grad_y1
            grad_x2 = grad_x2 + x2.grad
            
        return torch.cat([grad_x1, grad_x2], dim=-1), None, None

class ReversibleModule(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        hidden = dim // 2
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.g = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x):
        return ReversibleManifoldFunction.apply(x, self.f, self.g)

class TestReversibleDrift(unittest.TestCase):
    """
    Bit-accurate unit tests to detect L1 gradient drift during manifold reconstruction.
    Target: Mac Mini M4 (MPS/16GB).
    """
    def setUp(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dim = 256
        self.batch_size = 4
        self.tol = 1e-6 # Strict tolerance for bit-accuracy

    def test_gradient_drift(self):
        # 1. Initialize identical states
        x = torch.randn(self.batch_size, self.dim, device=self.device, requires_grad=True)
        model = ReversibleModule(self.dim).to(self.device)
        
        # 2. Standard Autograd Pass (Reference)
        x_ref = x.detach().clone().requires_grad_(True)
        x1, x2 = torch.chunk(x_ref, 2, dim=-1)
        y1_ref = x1 + model.f(x2)
        y2_ref = x2 + model.g(y1_ref)
        y_ref = torch.cat([y1_ref, y2_ref], dim=-1)
        
        loss_ref = y_ref.sum()
        loss_ref.backward()
        grad_ref = x_ref.grad.clone()

        # 3. Reversible Pass (Experimental)
        x_rev = x.detach().clone().requires_grad_(True)
        y_rev = model(x_rev)
        loss_rev = y_rev.sum()
        loss_rev.backward()
        grad_rev = x_rev.grad.clone()

        # 4. Measure L1 Drift (Spectral Shift η)
        drift = torch.abs(grad_ref - grad_rev).mean().item()
        print(f"\n[METRIC] L1 Gradient Drift: {drift:.2e}")

        # 5. Veracity Compact Check
        self.assertLess(drift, self.tol, f"Gradient drift {drift} exceeds tolerance {self.tol}. Manifold reconstruction is unstable.")

    def test_reconstruction_integrity(self):
        # Verify that x can be perfectly reconstructed from y
        x = torch.randn(1, self.dim, device=self.device)
        model = ReversibleModule(self.dim).to(self.device)
        
        with torch.no_grad():
            y = model(x)
            y1, y2 = torch.chunk(y, 2, dim=-1)
            
            # Manual Inversion Logic
            x2_rec = y2 - model.g(y1)
            x1_rec = y1 - model.f(x2_rec)
            x_rec = torch.cat([x1_rec, x2_rec], dim=-1)
            
            recon_error = torch.norm(x - x_rec).item()
            print(f"[METRIC] Reconstruction Error: {recon_error:.2e}")
            self.assertLess(recon_error, 1e-5)

if __name__ == "__main__":
    unittest.main()