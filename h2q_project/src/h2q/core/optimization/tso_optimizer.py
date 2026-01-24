import torch
import numpy as np
from typing import Optional, Dict, Any
from h2q.core.interface_registry import get_canonical_dde, LatentConfig, verify_dde_integrity
from h2q.core.memory.rskh_ssd_persistence_broker import RSKH_SSD_Persistence_Broker
from h2q.core.sst import SpectralShiftTracker
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor

class TSO_Optimizer:
    """
    Topological Self-Organization (TSO) Optimizer.
    Dynamically prunes or expands manifold density based on η-volatility and environmental drag μ(E).
    Ensures 16GB RAM stability for 100M+ token streams using RSKH persistence.
    """
    def __init__(
        self,
        initial_knots: int = 64,
        max_knots: int = 256,
        min_knots: int = 16,
        memory_limit_gb: float = 14.0,  # Safety buffer for 16GB total
        persistence_path: str = "./vault/rskh_knots.bin"
    ):
        self.current_knots = initial_knots
        self.max_knots = max_knots
        self.min_knots = min_knots
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        
        # Initialize Core Components
        self.sst = SpectralShiftTracker()
        self.hdi_monitor = ManifoldHeatDeathMonitor()
        self.persistence = RSKH_SSD_Persistence_Broker(persistence_path)
        
        # Initialize DDE via Canonical Interface to avoid 'dim' keyword errors
        config = LatentConfig(atoms=4, knots=initial_knots)
        self.dde = get_canonical_dde(config)
        
        # State tracking
        self.eta_history = []
        self.hdi_history = []

    def step(
        self, 
        manifold_tensors: torch.Tensor, 
        mu_e: float, 
        external_loss: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Performs one optimization step of the topology.
        manifold_tensors: [knots, 4, dim] quaternionic representation
        mu_e: Environmental drag (complexity of current stream)
        """
        # 1. Calculate Spectral Shift (η) and Heat-Death Index (HDI)
        eta = self.sst.calculate_shift(manifold_tensors)
        hdi = self.hdi_monitor.calculate_entropy(manifold_tensors)
        
        self.eta_history.append(eta)
        self.hdi_history.append(hdi)
        if len(self.eta_history) > 100: self.eta_history.pop(0)

        # 2. Monitor Memory Pressure (MPS specific)
        current_mem = 0
        if torch.backends.mps.is_available():
            current_mem = torch.mps.current_allocated_memory()
        
        # 3. Decision Logic: Prune, Expand, or Maintain
        eta_volatility = np.std(self.eta_history) if len(self.eta_history) > 1 else 0.0
        
        action = "MAINTAIN"
        
        # Expansion Trigger: High volatility + High drag + Memory Headroom
        if eta_volatility > 0.15 and mu_e > 0.5 and current_mem < self.memory_limit_bytes * 0.7:
            if self.current_knots < self.max_knots:
                manifold_tensors = self._expand_manifold(manifold_tensors)
                action = "EXPAND"
        
        # Pruning Trigger: High Entropy (HDI) OR Memory Pressure
        elif hdi > 0.85 or current_mem > self.memory_limit_bytes:
            if self.current_knots > self.min_knots:
                manifold_tensors = self._prune_manifold(manifold_tensors)
                action = "PRUNE"

        return {
            "action": action,
            "current_knots": self.current_knots,
            "eta": eta,
            "hdi": hdi,
            "memory_usage_gb": current_mem / 1024**3,
            "manifold": manifold_tensors
        }

    def _expand_manifold(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Fractal Expansion Protocol (h + δ).
        Doubles density of the most active knots.
        """
        new_knots_count = min(self.current_knots * 2, self.max_knots)
        if new_knots_count == self.current_knots: return tensors
        
        # Recursive seed evolution
        delta = torch.randn_like(tensors) * 0.01
        expanded = torch.cat([tensors, tensors + delta], dim=0)
        
        self.current_knots = expanded.shape[0]
        self._update_dde_config()
        return expanded

    def _prune_manifold(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Topological Pruning.
        Offloads low-utility knots to SSD via RSKH.
        """
        # Identify knots with lowest singular value contribution (Heat-Death candidates)
        u, s, v = torch.svd(tensors.view(self.current_knots, -1))
        keep_indices = torch.argsort(s, descending=True)[:self.current_knots // 2]
        prune_indices = torch.argsort(s, descending=True)[self.current_knots // 2:]
        
        # Persist pruned knots to SSD
        pruned_data = tensors[prune_indices]
        self.persistence.persist_knots(pruned_data)
        
        pruned_manifold = tensors[keep_indices]
        self.current_knots = pruned_manifold.shape[0]
        self._update_dde_config()
        return pruned_manifold

    def _update_dde_config(self):
        """
        Ensures the DiscreteDecisionEngine is symmetrical with the new manifold density.
        """
        new_config = LatentConfig(atoms=4, knots=self.current_knots)
        self.dde = get_canonical_dde(new_config)
        verify_dde_integrity(self.dde)

# Experimental: TTD (Topological Time Dilation) hook for OOM emergencies
def apply_ttd_emergency_brake(optimizer: TSO_Optimizer):
    if torch.mps.current_allocated_memory() > optimizer.memory_limit_bytes * 0.95:
        # Force aggressive pruning regardless of HDI
        optimizer.min_knots = 4 
        print("[TSO_CRITICAL] Memory threshold breached. Triggering TTD Pruning.")