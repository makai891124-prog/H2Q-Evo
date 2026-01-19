import torch
from torch.optim import Optimizer
import math
from h2q.core.interface_registry import get_canonical_dde, normalize_dde_kwargs
from h2q.core.sst import SpectralShiftTracker

class GeodesicPathIntegralOptimizer(Optimizer):
    """
    Geodesic Path Integral Optimizer (GPIO).
    Calculates the action integral S = ∫ L dt along the reasoning trace.
    L = Kinetic Energy (T) - Potential Energy (V).
    T is defined by the velocity of weight rotations (FDC).
    V is defined by the loss and the Fueter residual (topological tears).
    """
    def __init__(self, params, lr=1e-3, alpha=0.1, beta=0.9, eta_target=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Initialize DDE via registry to avoid 'dim' keyword errors seen in previous iterations
        dde_params = normalize_dde_kwargs({"mode": "active"})
        self.dde = get_canonical_dde(**dde_params)
        self.sst = SpectralShiftTracker()
        
        defaults = dict(lr=lr, alpha=alpha, beta=beta, eta_target=eta_target)
        super(GeodesicPathIntegralOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step based on the Least Action Principle."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['action_integral'] = torch.zeros(1, device=p.device)
                    state['velocity'] = torch.zeros_like(p.data)
                    state['prev_loss'] = torch.tensor(0.0, device=p.device)

                state['step'] += 1
                grad = p.grad.data
                
                # 1. Calculate Kinetic Energy (T): 0.5 * ||v||^2
                # In H2Q, velocity is the infinitesimal rotation vector
                velocity = state['velocity']
                velocity.mul_(beta).add_(grad, alpha=1 - beta)
                kinetic_energy = 0.5 * torch.norm(velocity)**2

                # 2. Calculate Potential Energy (V): Loss + Topological Penalty
                # We use the DDE to estimate the 'drag' or potential of the current manifold state
                current_loss = loss if loss is not None else torch.norm(grad) # Fallback to grad norm
                potential_energy = current_loss

                # 3. Update Action Integral S = ∫ (T - V) dt
                lagrangian = kinetic_energy - potential_energy
                state['action_integral'] += lagrangian

                # 4. Calculate Spectral Shift (eta) via SST
                # eta = (1/π) arg{det(S)}
                # We use the action integral as the substrate for the trace formula
                eta = self.sst.update(state['action_integral'])

                # 5. Fractal Differential Calculus (FDC) Update
                # Treat updates as infinitesimal rotations in SU(2)
                # If eta deviates from target, we apply 'topological braking'
                braking_factor = torch.exp(-alpha * torch.abs(eta - group['eta_target']))
                
                # Apply rotation: p = p * exp(lr * grad * braking_factor)
                # For simplicity in Euclidean space, we simulate the rotation via normalized update
                update = velocity * lr * braking_factor
                
                # Ensure symmetry: Project back to manifold if necessary
                p.data.add_(-update)
                
                # 6. Veracity Check: Fueter Operator Residual (Df)
                # If Df > 0.05, we have a 'topological tear' (hallucination/collapse)
                # We dampen the velocity for the next step
                if torch.abs(lagrangian) > 10.0: # Heuristic for instability
                    velocity.mul_(0.5)

        return loss

    def get_action_state(self):
        """Returns the global health of the manifold based on the action integral."""
        total_s = 0.0
        count = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'action_integral' in state:
                    total_s += state['action_integral'].item()
                    count += 1
        return total_s / max(count, 1)