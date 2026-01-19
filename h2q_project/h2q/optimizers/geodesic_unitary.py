import torch
from torch.optim import Optimizer
import math

class GeodesicUnitaryOptimizer(Optimizer):
    """
    FDC-Optim: Geodesic Unitary Optimizer.
    
    Treats gradients as elements of the tangent space (Lie Algebra su(n))
    and updates weights via the exponential map to ensure they remain on the 
    SU(n) manifold (Unitarity preservation) without weight clipping.
    
    Irreducible Atom: Manifold Mapping (Gradients -> Skew-Hermitian Generators).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(GeodesicUnitaryOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step using geodesic flow on SU(n)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute adaptive learning rate (Adam-style scaling in tangent space)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Effective gradient in Euclidean space
                eff_grad = exp_avg / denom

                # --- RIGID CONSTRUCTION: SU(n) Projection ---
                # To preserve unitarity, we map the Euclidean gradient to the Lie Algebra su(n).
                # The generator Omega must be skew-Hermitian: Omega = G*W^H - W*G^H
                # For real-valued weights (SO(n)), this is Omega = G*W^T - W*G^T
                
                # Ensure p is at least 2D for matrix operations
                if p.dim() < 2:
                    # Fallback for biases/scalars: standard Euclidean update
                    p.add_(eff_grad, alpha=-step_size)
                    continue

                # Reshape to 2D if necessary (e.g., for Conv layers)
                original_shape = p.shape
                w = p.view(original_shape[0], -1)
                g = eff_grad.view(original_shape[0], -1)

                # Construct the skew-symmetric generator (Lie Algebra element)
                # Omega = g @ w.T - w @ g.T
                omega = torch.matmul(g, w.t()) - torch.matmul(w, g.t())
                
                # --- ELASTIC WEAVING: Exponential Map ---
                # Update: W_new = expm(-lr * omega) @ W_old
                # matrix_exp is stable on MPS (Mac Mini M4)
                update_matrix = torch.matrix_exp(-step_size * omega)
                
                # Apply rotation
                new_w = torch.matmul(update_matrix, w)
                
                # Restore original shape
                p.copy_(new_w.view(original_shape))

        return loss

# EXPERIMENTAL: Spectral Shift Tracker Integration
def calculate_spectral_shift(S_matrix):
    """
    Implements η = (1/π) arg{det(S)}
    Tracks cognitive deflection against environmental drag.
    """
    sign, logdet = torch.linalg.slogdet(S_matrix)
    # η is the phase of the determinant
    eta = torch.angle(sign) / math.pi
    return eta
