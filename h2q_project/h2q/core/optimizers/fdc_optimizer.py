import torch
from torch.optim import Optimizer
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.interface_registry import get_canonical_dde

class FDCOptimizer(Optimizer):
    """
    FDCOptimizer: Fractal Differential Calculus Optimizer.
    Implements 'Rodrigues-FDC-Momentum' for stabilized parallel transport 
    across the S³ unit hypersphere (SU(2) manifold).
    
    The momentum vector is parallel-transported from the tangent space at q_prev 
    to the tangent space at q_curr using the Rodrigues rotation formula adapted 
    for quaternionic geodesic flow.
    """
    def __init__(self, params, lr=1e-3, beta=0.9, eta=0.1, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
            
        defaults = dict(lr=lr, beta=beta, eta=eta, weight_decay=weight_decay)
        super(FDCOptimizer, self).__init__(params, defaults)
        
        # Initialize Discrete Decision Engine for phase-deflection monitoring
        # Using canonical getter to avoid 'dim' keyword argument errors
        # Use a default latent_dim for monitoring purposes
        self.dde = get_canonical_dde(latent_dim=256)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eta = group['eta']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['prev_q'] = p.clone()

                momentum = state['momentum_buffer']
                prev_q = state['prev_q']
                curr_q = p

                # --- RODRIGUES-FDC-MOMENTUM: PARALLEL TRANSPORT ---
                # 1. Calculate the relative rotation (geodesic) between prev_q and curr_q
                # In SU(2), parallel transport of a tangent vector v along a geodesic 
                # from q1 to q2 is given by the Adjoint action: v' = (q2 * q1^-1) * v * (q2 * q1^-1)^*
                
                # Compute delta rotation: dq = curr_q * conj(prev_q)
                # Note: Quaternions are assumed to be [w, x, y, z] blocks
                conj_prev = prev_q.clone()
                conj_prev[..., 1:] *= -1 # Conjugate
                
                dq = quaternion_mul(curr_q, conj_prev)
                dq = quaternion_normalize(dq)

                # 2. Apply Rodrigues-like rotation to the momentum buffer
                # This aligns the previous momentum with the current tangent space
                # v_transported = dq * momentum * conj(dq)
                conj_dq = dq.clone()
                conj_dq[..., 1:] *= -1
                
                # Parallel transport the momentum
                transported_m = quaternion_mul(quaternion_mul(dq, momentum), conj_dq)

                # 3. Update momentum with current gradient
                # m = beta * transported_m + (1 - beta) * grad
                momentum.copy_(transported_m.mul_(beta).add_(d_p, alpha=1 - beta))

                # 4. Geodesic Step
                # Update the manifold position along the geodesic defined by momentum
                # We use a small-angle approximation for the exponential map
                update_dir = momentum.mul(-lr)
                
                # Apply update via quaternionic multiplication to stay on S³
                # q_new = exp(update_dir) * q_curr
                # For small update_dir, exp(v) approx [1, v_x, v_y, v_z]
                exp_map = torch.zeros_like(update_dir)
                exp_map[..., 0] = 1.0
                exp_map[..., 1:] = update_dir[..., 1:] # Tangent components
                
                new_q = quaternion_mul(exp_map, curr_q)
                p.copy_(quaternion_normalize(new_q))

                # Update state tracking
                state['prev_q'].copy_(p)
                state['step'] += 1

                # 5. Veracity Check: Discrete Fueter Operator (Df)
                # If phase deflection η exceeds drag μ(E), the DDE triggers a correction
                # This is handled implicitly by the DDE's internal state
                self.dde.step(loss=loss if loss is not None else torch.tensor(0.0))

        return loss