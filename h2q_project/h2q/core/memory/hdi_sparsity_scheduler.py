import torch
import psutil
import logging
from typing import Dict, List, Optional
from h2q.core.memory.rskh_ssd_paging import RSKH_SSD_Paging_System
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.interface_registry import normalize_dde_kwargs

class HDISparsityScheduler:
    """
    Real-time memory scheduler for H2Q AGI.
    Dynamically pages manifold knots to SSD based on Heat-Death Index (HDI) 
    to maintain the 16GB RAM ceiling on M4 Silicon.
    """
    def __init__(
        self,
        paging_system: RSKH_SSD_Paging_System,
        hdi_monitor: ManifoldHeatDeathMonitor,
        memory_threshold_gb: float = 13.5,  # Safety margin for 16GB total
        eviction_batch_size: int = 4
    ):
        self.paging_system = paging_system
        self.hdi_monitor = hdi_monitor
        self.memory_threshold = memory_threshold_gb * 1024 * 1024 * 1024
        self.eviction_batch_size = eviction_batch_size
        
        # Initialize DDE using canonical registry to avoid 'dim' keyword errors
        dde_params = normalize_dde_kwargs({"strategy": "entropy_minimax"})
        self.dde = get_canonical_dde(**dde_params)
        
        self.logger = logging.getLogger("H2Q.HDIScheduler")

    def check_memory_pressure(self) -> bool:
        """Returns True if memory usage exceeds the safety threshold."""
        current_mem = psutil.virtual_memory().used
        return current_mem > self.memory_threshold

    def evaluate_knot_utility(self, knot_id: str, hdi_value: float, spectral_shift: float) -> torch.Tensor:
        """
        Uses the Discrete Decision Engine to determine if a knot should be paged.
        High HDI + Low Spectral Shift = High Paging Priority.
        """
        # Construct state vector for DDE
        state = torch.tensor([hdi_value, spectral_shift], dtype=torch.float32)
        # DDE decides: 0 = Keep in RAM, 1 = Page to SSD
        decision = self.dde.decide(state)
        return decision

    def step(self, active_knots: Dict[str, torch.Tensor]):
        """
        Execution cycle for the sparsity scheduler.
        """
        if not self.check_memory_pressure():
            return

        self.logger.info("Memory pressure detected. Initiating HDI-based paging.")
        
        # Retrieve HDI telemetry for all active knots
        hdi_map = self.hdi_monitor.get_current_hdi_map()
        
        # Rank knots by HDI (Heat-Death Index)
        # Knots with high HDI are 'stagnant' and candidates for SSD paging
        candidates = sorted(
            hdi_map.items(), 
            key=lambda x: x[1], 
            reverse=True
        )

        paged_count = 0
        for knot_id, hdi_val in candidates:
            if paged_count >= self.eviction_batch_size:
                break

            # Verify with DDE before eviction
            # Note: spectral_shift (eta) is retrieved from the monitor
            eta = self.hdi_monitor.get_knot_spectral_shift(knot_id)
            
            if self.evaluate_knot_utility(knot_id, hdi_val, eta) > 0.5:
                self.logger.debug(f"Paging knot {knot_id} to SSD (HDI: {hdi_val:.4f})")
                success = self.paging_system.page_to_ssd(knot_id, active_knots[knot_id])
                
                if success:
                    # Remove from active RAM
                    del active_knots[knot_id]
                    paged_count += 1

        if paged_count > 0:
            # Trigger MPS cache clearing to ensure RAM is actually freed
            if torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
            self.logger.info(f"Successfully paged {paged_count} knots to SSD.")

    def force_vacuum(self, active_knots: Dict[str, torch.Tensor]):
        """Emergency deallocation of highest HDI knots regardless of DDE."""
        hdi_map = self.hdi_monitor.get_current_hdi_map()
        critical_candidates = sorted(hdi_map.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(min(len(critical_candidates), self.eviction_batch_size * 2)):
            knot_id = critical_candidates[i][0]
            if knot_id in active_knots:
                self.paging_system.page_to_ssd(knot_id, active_knots[knot_id])
                del active_knots[knot_id]
        
        torch.backends.mps.empty_cache()