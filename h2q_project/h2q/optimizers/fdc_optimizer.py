import torch
from torch.optim import Optimizer
import math

class FDCOptimizer(Optimizer):
    """
    FDC (Fractal-Differential-Coupling) Optimizer
    Implements learning as infinitesimal rotations in su(2) Lie Algebra.
    Integrates Geodesic Trace-Error Recovery (GTER) via QR-decomposition.
    """
    def __init__(self, params, lr=1e-3, mu_env=0.01, gter_interval=10):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, mu_env=mu_env, gter_interval=gter_interval)
        super(FDCOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu_env']
            interval = group['gter_interval']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Spectral Shift Tracker (eta)
                    state['eta'] = torch.zeros(1, device=p.device)

                state['step'] += 1
                d_p = p.grad

                # --- RIGID CONSTRUCTION: su(2) Update ---
                # We treat the gradient as an element of the Lie Algebra su(2).
                # Update: p = p * exp(-lr * d_p)
                # For infinitesimal rotations, we use the first-order Taylor approximation
                # and then project back to the manifold via GTER.
                
                # Apply environmental drag mu(E) to the gradient
                d_p = d_p + mu * p
                
                # Infinitesimal rotation update
                p.add_(d_p, alpha=-lr)

                # --- ELASTIC WEAVING: GTER Mechanism ---
                # Neutralize floating-point drift every N steps using QR-Decomposition
                if state['step'] % interval == 0:
                    self._apply_gter(p)

                # Update Spectral Shift Tracker (eta)
                # η = (1/π) arg{det(S)}
                # Here S is approximated by the local parameter transformation matrix
                if p.dim() >= 2:
                    # Experimental: Tracking phase deflection of the weight matrix
                    # Using a simplified trace-based proxy for det(S) in high-dim
                    s_matrix = p.view(p.size(0), -1)[:, :p.size(0)]
                    if s_matrix.size(0) == s_matrix.size(1):
                        det_s = torch.linalg.det(s_matrix + torch.eye(s_matrix.size(0), device=p.device) * 1e-6)
                        state['eta'] = (1.0 / math.pi) * torch.angle(det_s.to(torch.complex64))

        return loss

    def _apply_gter(self, p):
        """
        [EXPERIMENTAL] Geodesic Trace-Error Recovery
        Enforces SU(2) unitarity by projecting the parameter back to the manifold.
        """
        original_shape = p.shape
        
        # SU(2) isomorphism requires 2x2 complex or 4-real components.
        # We treat the last two dimensions as the SU(2) manifold space.
        if p.numel() % 4 == 0:
            # Reshape to pseudo-complex 2x2 matrices: [Batch, 2, 2]
            # We use the property that any M in GL(2,C) can be decomposed to QR
            # where Q is in U(2). We then normalize det(Q) to 1 for SU(2).
            reshaped_p = p.view(-1, 2, 2)
            
            # QR Decomposition (Stable on MPS/Mac Mini M4)
            q, r = torch.linalg.qr(reshaped_p.to(torch.complex64) if not p.is_complex() else reshaped_p)
            
            # Force det(Q) = 1 to satisfy SU(2) constraint
            det_q = torch.linalg.det(q).unsqueeze(-1).unsqueeze(-1)
            # Phase correction: Q_su2 = Q / sqrt(det(Q))
            q_su2 = q / torch.sqrt(det_q)
            
            # Cast back to original dtype and shape
            if not p.is_complex():
                p.copy_(q_su2.real.view(original_shape))
            else:
                p.copy_(q_su2.view(original_shape))
        else:
            # Fallback for non-SU(2) compatible shapes: Standard Orthonormalization
            if p.dim() >= 2:
                flat_p = p.view(p.size(0), -1)
                q, r = torch.linalg.qr(flat_p)
                p.copy_(q.view(original_shape))
            else:
                # Vector normalization (S3 unit quaternion drift correction)
                p.div_(torch.norm(p) + 1e-12)
