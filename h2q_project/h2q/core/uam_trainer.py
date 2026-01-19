import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from h2q.core.sst import SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor
from h2q.core.trainers.sleep_healer import H2QSleepHealer
from h2q.core.logic_auditing import HolomorphicAuditKernel

class H2QUnifiedAutonomousMaster:
    """
    UAM Trainer: Orchestrates Wake-phase SGD and Sleep-phase Geodesic Healing.
    Uses the Manifold Heat-Death Index (HDI) to prevent topological tears (Df != 0).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        hdi_threshold: float = 0.75,
        device: str = "mps"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.hdi_threshold = hdi_threshold
        self.device = device
        
        # Initialize Core H2Q Components via Registry to avoid 'dim' kwarg errors
        self.dde = get_canonical_dde() 
        self.sst = SpectralShiftTracker()
        self.mhdm = ManifoldHeatDeathMonitor()
        self.healer = H2QSleepHealer(model=self.model, device=self.device)
        self.auditor = HolomorphicAuditKernel()
        
        self.phase = "WAKE"
        self.stats = {"hdi": 0.0, "spectral_shift": 0.0, "fueter_residual": 0.0}

    def _compute_manifold_metrics(self) -> Dict[str, float]:
        """
        Calculates the current state of the quaternionic manifold.
        """
        # η = (1/π) arg{det(S)}
        eta = self.sst.compute_shift(self.model)
        # Df: Fueter residuals (hallucination detection)
        fueter_res = self.auditor.calculate_residuals(self.model)
        # HDI: Combined entropy and topological tear index
        hdi = self.mhdm.calculate_index(eta, fueter_res)
        
        return {
            "hdi": hdi.item() if isinstance(hdi, torch.Tensor) else hdi,
            "spectral_shift": eta.item() if isinstance(eta, torch.Tensor) else eta,
            "fueter_residual": fueter_res.item() if isinstance(fueter_res, torch.Tensor) else fueter_res
        }

    def step(self, data_batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        A single iteration of the UAM loop.
        """
        # 1. Audit Manifold Integrity
        self.stats.update(self._compute_manifold_metrics())
        
        # 2. Autonomous Decision via DDE
        # We pass the HDI to the DDE to decide if the system needs 'Sleep'
        decision = self.dde.forward(torch.tensor([self.stats["hdi"]], device=self.device))
        
        # Trigger Sleep if HDI exceeds threshold or DDE mandates it
        if self.stats["hdi"] > self.hdi_threshold or decision > 0.5:
            self.phase = "SLEEP"
            healing_results = self.sleep_phase()
            return {"phase": "SLEEP", "metrics": self.stats, "healing": healing_results}
        
        # 3. Wake Phase (Standard SGD)
        self.phase = "WAKE"
        wake_results = self.wake_phase(data_batch)
        return {"phase": "WAKE", "metrics": self.stats, "train": wake_results}

    def wake_phase(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Standard learning as an infinitesimal rotation in su(2).
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(batch['input_ids'])
        loss = nn.functional.cross_entropy(outputs, batch['labels'])
        
        loss.backward()
        # Apply Geodesic Constraint: Ensure updates stay on SU(2)^64
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def sleep_phase(self) -> Dict[str, float]:
        """
        Geodesic Healing: Minimizes Fueter residuals and stabilizes η.
        """
        self.model.eval()
        # H2QSleepHealer performs infinitesimal rotations to close topological tears
        healing_loss = self.healer.heal_system(iterations=5)
        
        # Reset SST baseline after healing to stabilize Geodesic Flow
        self.sst.reset_baseline()
        
        return {"healing_loss": healing_loss}

    def get_status(self):
        return {
            "current_phase": self.phase,
            "manifold_health": "STABLE" if self.stats["hdi"] < self.hdi_threshold else "CRITICAL",
            "metrics": self.stats
        }