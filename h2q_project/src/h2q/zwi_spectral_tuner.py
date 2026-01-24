import torch
import torch.nn as nn
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker
from h2q.core.optimization.fdc_optimizer import FDCOptimizer

class ZeroWeightSpectralTuner(nn.Module):
    """
    Zero-Weight Intelligence (ZWI) Phase Tuner.
    Adapts a frozen 256-D quaternionic manifold to new logical domains 
    by optimizing the spectral phase-shift (eta) using Fractal Differential Calculus (FDC).
    """
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        # Rigid Construction: Freeze all manifold weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Elastic Extension: Learnable spectral phase shift (phi)
        # This modulates the geodesic flow without altering the underlying crystal structure.
        self.phi = nn.Parameter(torch.zeros(1, device='mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Initialize SST for monitoring cognitive progress
        self.sst = SpectralShiftTracker()
        
        # Initialize DDE using canonical registry to avoid 'dim' keyword errors
        # The DDE governs the discrete transitions between manifold knots.
        self.dde = get_canonical_dde()
        
        # FDC Optimizer: Specifically designed for non-Euclidean phase optimization
        self.optimizer = FDCOptimizer([self.phi], lr=learning_rate)

    def forward(self, x: torch.Tensor):
        """
        Forward pass modulating the frozen manifold with the learnable phase phi.
        """
        # Calculate base spectral shift from the frozen manifold
        with torch.no_grad():
            # Assuming model has a method to compute current spectral density S
            # If not, we use the SST to estimate it from the input stream
            base_eta = self.sst.update(x)

        # Apply the learned phase shift: eta_total = base_eta + phi
        effective_eta = base_eta + self.phi
        
        # Inject the effective_eta into the model's decision engine
        # This steers the geodesic flow along the manifold knots
        output = self.model(x, eta_override=effective_eta)
        
        return output, effective_eta

    def train_zwi_step(self, x: torch.Tensor, target: torch.Tensor, task_loss_fn: callable):
        """
        Performs a single ZWI adaptation step.
        """
        self.optimizer.zero_grad()
        
        output, eta = self.forward(x)
        
        # Task Loss: Standard domain-specific error
        loss_task = task_loss_fn(output, target)
        
        # Spectral Entropy Loss: Encourages the phase to find 'stable' geodesics
        # (1/pi) arg{det(S)} alignment
        loss_spectral = torch.abs(torch.sin(torch.pi * eta))
        
        total_loss = loss_task + 0.1 * loss_spectral
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "eta": eta.item(),
            "phi": self.phi.item()
        }

def run_zwi_benchmark(model, dataloader, task_loss_fn):
    """
    Utility to run the ZWI Phase Tuner on a specific dataset.
    """
    tuner = ZeroWeightSpectralTuner(model)
    results = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        metrics = tuner.train_zwi_step(data, target, task_loss_fn)
        results.append(metrics)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss: {metrics['loss']:.4f} | Eta: {metrics['eta']:.4f}")
            
    return results
