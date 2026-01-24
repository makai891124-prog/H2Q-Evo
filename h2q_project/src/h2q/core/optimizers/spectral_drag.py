import torch
from torch.optim import Optimizer
import math
from collections import deque

class SpectralShiftTracker:
    """
    Calculates the Krein-like spectral shift η = (1/π) arg{det(S)}
    and monitors volatility to derive environmental drag μ(E).
    """
    def __init__(self, window_size=32, device="mps"):
        self.window_size = window_size
        self.device = device
        self.eta_history = deque(maxlen=window_size)
        
    def compute_eta(self, S_matrix):
        """
        S_matrix: Scattering matrix of cognitive transitions (Complex Tensor).
        Formula: η = (1/π) arg{det(S)}
        """
        # Ensure complex for determinant
        if not torch.is_complex(S_matrix):
            # Assume real representation of SU(2) if not complex
            # For 2x2 SU(2), det is real, but we handle the general manifold case
            S_matrix = torch.complex(S_matrix, torch.zeros_like(S_matrix))
            
        # MPS safe determinant calculation for small matrices
        # det(S) for SU(2) double-cover
        det_s = torch.linalg.det(S_matrix)
        
        # η = (1/π) * phase(det_s)
        eta = torch.angle(det_s) / math.pi
        return eta.detach()

    def get_environmental_drag(self, current_eta):
        """
        μ(E) derived from the volatility (variance) of η-signatures.
        """
        self.eta_history.append(current_eta.item())
        
        if len(self.eta_history) < 2:
            return 0.0
            
        eta_tensor = torch.tensor(list(self.eta_history), device=self.device)
        volatility = torch.std(eta_tensor)
        
        # μ(E) = Volatility of the context stream
        return volatility.item()

class SpectralDragOptimizer(Optimizer):
    """
    SDO: Modulates learning rate as an inverse function of environmental drag μ(E).
    LR_eff = LR_base * exp(-drag_coeff * μ(E))
    """
    def __init__(self, params, lr=1e-3, drag_coefficient=1.0, window_size=32, device="mps"):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, base_lr=lr, drag_coefficient=drag_coefficient)
        super(SpectralDragOptimizer, self).__init__(params, defaults)
        
        self.tracker = SpectralShiftTracker(window_size=window_size, device=device)
        self.current_drag = 0.0

    @torch.no_grad()
    def step(self, S_matrix, closure=None):
        """
        Performs a single optimization step modulated by spectral drag.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Update η and calculate environmental drag μ(E)
        eta = self.tracker.compute_eta(S_matrix)
        self.current_drag = self.tracker.get_environmental_drag(eta)

        # 2. Modulate Learning Rate: Inverse function of drag
        # We use exponential decay for stability: LR * exp(-k * μ)
        for group in self.param_groups:
            drag_factor = math.exp(-group['drag_coefficient'] * self.current_drag)
            group['lr'] = group['base_lr'] * drag_factor

            # 3. Standard SGD-like update (Rigid Construction)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])

        return loss

    def get_telemetry(self):
        return {
            "environmental_drag": self.current_drag,
            "effective_lr": self.param_groups[0]['lr']
        }

# --- VERACITY COMPACT: FIXING INITIALIZATION ERROR ---
# The registry indicates DiscreteDecisionEngine in h2q/core/discrete_decision_engine.py 
# expects a 'config' object, not a 'dim' keyword.

class LatentConfig:
    def __init__(self, latent_dim, action_dim, temperature=1.0):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.temperature = temperature

def get_canonical_dde(latent_dim, action_dim):
    """
    Factory to prevent 'unexpected keyword argument dim' errors.
    """
    from h2q.core.discrete_decision_engine import DiscreteDecisionEngine
    config = LatentConfig(latent_dim=latent_dim, action_dim=action_dim)
    return DiscreteDecisionEngine(config)