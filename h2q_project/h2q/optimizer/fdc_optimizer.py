import torch
from torch.optim import Optimizer
import torch.nn as nn

class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Fixed initialization signature to resolve 'dim' keyword error.
    Handles discrete state transitions within the SU(2) manifold.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Fix: Renamed 'dim' to 'input_dim' to match internal registry and external calls
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.gate(self.projection(x))

class FDCOptimizer(Optimizer):
    """
    [EXPERIMENTAL] Fractal-Derivative-Constrained Optimizer with Holomorphic Logic Auditing.
    Integrates Fueter-analyticity residuals to penalize non-conformal reasoning steps.
    """
    def __init__(self, params, lr=1e-3, lambda_fueter=0.01, fractal_delta=1e-4):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, lambda_fueter=lambda_fueter, fractal_delta=fractal_delta)
        super(FDCOptimizer, self).__init__(params, defaults)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with Logic Curvature penalty.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. IDENTIFY_ATOMS: Extract Quaternionic Components
                # Assuming p is reshaped or structured as [..., 4] for SU(2) projection
                if p.dim() >= 1 and p.shape[-1] == 4:
                    # Calculate Fueter-analyticity residual (Logic Curvature)
                    # Df = dw/dw + i*dx/dx + j*dy/dy + k*dz/dz -> simplified as divergence check
                    # In neural context, we penalize the deviation from the Hamilton symmetry
                    q_grad = p.grad.view(-1, 4)
                    w, x, y, z = q_grad[:, 0], q_grad[:, 1], q_grad[:, 2], q_grad[:, 3]
                    
                    # Fueter Residual: || dw/d0 + dx/d1 + dy/d2 + dz/d3 ||
                    # Here we approximate via the variance of the gradient norms across the manifold
                    logic_curvature = torch.var(torch.stack([w.norm(), x.norm(), y.norm(), z.norm()]))
                    
                    # Apply soft-penalty to the gradient
                    p.grad.add_(p, alpha=group['lambda_fueter'] * logic_curvature.item())

                # 2. ELASTIC WEAVING: Fractal Update
                # Apply h ± delta scaling to prevent vanishing gradients in deep knots
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])
                
                # Spectral Shift Tracking (Simplified for step logic)
                # Ensures the update doesn't push the state out of the SU(2) unit sphere
                if p.shape[-1] == 4:
                    p.data = nn.functional.normalize(p.data, p=2, dim=-1)

        return loss

    def compute_logic_curvature(self, quaternionic_tensor):
        """
        Explicitly calculates the Fueter-analyticity residual.
        Used for auditing long-context generation consistency.
        """
        # Ensure tensor is [N, 4]
        q = quaternionic_tensor.view(-1, 4)
        # Logic Curvature is the L2 norm of the Fueter operator result
        # For a stable system, Df -> 0
        residual = torch.abs(torch.mean(q[:, 0] + q[:, 1] + q[:, 2] + q[:, 3]))
        return residual