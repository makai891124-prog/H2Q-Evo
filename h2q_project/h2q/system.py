import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Verifying imports from H2Q Global Interface Registry
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor
from h2q.core.trainers.sleep_healer import H2QSleepHealer
from h2q.core.optimizers.fdc_optimizer import FDCOptimizer
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class ConceptDecoder(nn.Module):
    """Experimental: Decodes SU(2) knots into logical atoms."""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection = nn.Linear(latent_dim, 512) # Symmetry expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(x))

class AutonomousSystem(nn.Module):
    """
    The Unified Homeostatic Loop controller.
    Manages the transition between Wake (SGD) and Sleep (HJB Healing)
    based on the Heat-Death Index (HDI).
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        
        # Fix for Runtime Error: DiscreteDecisionEngine.__init__() got an unexpected keyword argument 'dim'
        # Using canonical factory method to handle kwarg normalization
        self.dde = get_canonical_dde(config=config)
        
        self.sst = SpectralShiftTracker()
        self.monitor = ManifoldHeatDeathMonitor(model)
        self.healer = H2QSleepHealer(model)
        self.optimizer = FDCOptimizer(model.parameters(), lr=config.get("lr", 1e-4))
        
        self.hdi_threshold = config.get("hdi_threshold", 0.75)
        self.recovery_threshold = config.get("recovery_threshold", 0.20)
        self.phase = "WAKE"

    def compute_hdi(self) -> float:
        """Calculates the Heat-Death Index via Manifold Entropy."""
        # Telemetry from the monitor mapping logic curvature
        stats = self.monitor.audit_manifold()
        return stats.get("heat_death_index", 0.0)

    def homeostatic_step(self, data_batch: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Executes one iteration of the Homeostatic Loop.
        Toggles between Wake-phase SGD and Sleep-phase HJB Geodesic Healing.
        """
        hdi = self.compute_hdi()
        metrics = {"hdi": hdi, "phase": self.phase}

        # Phase Transition Logic
        if self.phase == "WAKE" and hdi > self.hdi_threshold:
            self.phase = "SLEEP"
            print(f"[H2Q_SYSTEM] HDI Critical ({hdi:.4f}). Transitioning to SLEEP phase.")
        elif self.phase == "SLEEP" and hdi < self.recovery_threshold:
            self.phase = "WAKE"
            print(f"[H2Q_SYSTEM] Manifold Healed ({hdi:.4f}). Transitioning to WAKE phase.")

        # Execution
        if self.phase == "WAKE":
            if data_batch is not None:
                self.optimizer.zero_grad()
                output = self.model(data_batch)
                loss = self.model.compute_loss(output, data_batch)
                loss.backward()
                self.optimizer.step()
                
                # Track spectral shift η
                eta = self.sst.update(self.model)
                metrics["loss"] = loss.item()
                metrics["eta"] = eta
        else:
            # Sleep Phase: HJB Geodesic Healing
            # No data_batch required; operates on internal manifold curvature
            healing_stats = self.healer.heal_system()
            metrics.update(healing_stats)

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard inference pass."""
        return self.model(x)
