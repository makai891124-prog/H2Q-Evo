import torch
from torch.optim.lr_scheduler import _LRScheduler
from h2q.core.interface_registry import SpectralShiftTracker

class SpectralDragScheduler(_LRScheduler):
    """
    Spectral-Drag-Scheduler: An adaptive learning rate controller for the H2Q architecture.
    Modulates step-size inversely to the environmental drag μ(E) derived from 
    SpectralShiftTracker (η) history.
    
    Formula:
    μ(E)_t = β * μ(E)_{t-1} + (1 - β) * |η_t|
    LR_t = LR_initial / (1 + α * μ(E)_t)
    """
    def __init__(self, optimizer, tracker: SpectralShiftTracker, alpha=0.5, beta=0.9, last_epoch=-1, verbose=False):
        if not isinstance(tracker, SpectralShiftTracker):
            raise TypeError(f"Expected h2q.core.interface_registry.SpectralShiftTracker, got {type(tracker)}")
        
        self.tracker = tracker
        self.alpha = alpha  # Sensitivity to drag
        self.beta = beta    # Momentum for drag EMA
        self.mu_e = 0.0     # Environmental Drag
        self.verbose = verbose
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculates the learning rate based on current environmental drag μ(E)."""
        if self.last_epoch == 0:
            return self.base_lrs
        
        # Inverse modulation: Higher drag results in smaller steps to preserve manifold stability
        drag_factor = 1.0 / (1.0 + self.alpha * self.mu_e)
        
        return [base_lr * drag_factor for base_lr in self.base_lrs]

    def step(self, scattering_matrix=None):
        """
        Updates the scheduler state.
        Args:
            scattering_matrix (torch.Tensor): The S matrix from the current Geodesic Flow step.
        """
        if scattering_matrix is not None:
            # 1. Extract Spectral Shift η = (1/π) arg{det(S)}
            # We use the tracker's canonical implementation to ensure symmetry
            eta = self.tracker.compute_eta(scattering_matrix)
            
            # 2. Update Environmental Drag μ(E) 
            # Drag is modeled as the magnitude of phase deflection in the manifold
            current_drag = torch.abs(torch.as_tensor(eta, device='cpu')).item()
            self.mu_e = (self.beta * self.mu_e) + (1.0 - self.beta) * current_drag
            
            if self.verbose:
                print(f"[Spectral-Drag] η: {eta:.4f} | μ(E): {self.mu_e:.4f} | Factor: {1.0/(1.0 + self.alpha * self.mu_e):.4f}")

        super().step()

    def get_drag_telemetry(self):
        """Returns current drag metrics for Holomorphic Auditing."""
        return {
            "environmental_drag": self.mu_e,
            "last_eta": self.tracker.compute_eta(torch.eye(2)) # Placeholder for last known
        }
