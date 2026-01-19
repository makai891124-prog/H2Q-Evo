import torch
import psutil
import os
import logging
from typing import Dict, Any, List

# H2Q Interface Imports
from h2q.core.memory.rskh_ssd_paging import RSKH_SSD_Paging_System
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor
from h2q.core.quantization.tpq_engine import TopologicalPhaseQuantizer
from h2q.core.interface_registry import get_canonical_dde

class SpectralPagingService:
    """
    Monitors Manifold Heat Death Index (HDI) and manages memory pressure by 
    paging 4-bit TPQ-quantized knots to SSD when RAM usage exceeds the 14GB threshold.
    Optimized for Mac Mini M4 (16GB).
    """
    def __init__(self, threshold_gb: float = 14.0, ssd_path: str = "./vault/spectral_paging"):
        self.threshold = threshold_gb * (1024 ** 3)  # Convert to bytes
        self.ssd_path = ssd_path
        os.makedirs(self.ssd_path, exist_ok=True)

        # Initialize H2Q Components
        # Note: get_canonical_dde() avoids the 'dim' keyword error identified in feedback
        self.dde = get_canonical_dde()
        self.paging_system = RSKH_SSD_Paging_System(vault_path=self.ssd_path)
        self.hdi_monitor = ManifoldHeatDeathMonitor()
        self.quantizer = TopologicalPhaseQuantizer(bits=4)
        
        self.logger = logging.getLogger("SpectralPagingService")
        self.logger.info(f"SpectralPagingService initialized with {threshold_gb}GB threshold.")

    def check_memory_pressure(self) -> bool:
        """Returns True if system RAM usage exceeds the threshold."""
        mem = psutil.virtual_memory()
        return mem.used > self.threshold

    def monitor_and_swap(self, active_knots: Dict[str, torch.Tensor]) -> List[str]:
        """
        Evaluates active knots, identifies high-HDI candidates, and pages them to SSD.
        """
        if not self.check_memory_pressure():
            return []

        self.logger.warning("Memory pressure detected. Initiating Spectral Swap.")
        
        # 1. Calculate HDI for all active knots to prioritize swapping
        hdi_scores = {}
        for knot_id, tensor in active_knots.items():
            # HDI represents 'environmental drag' or 'stagnation' in the manifold
            hdi_scores[knot_id] = self.hdi_monitor.calculate_hdi(tensor)

        # 2. Sort knots by HDI (highest HDI = most 'dead' = best candidate for paging)
        sorted_knots = sorted(hdi_scores.items(), key=lambda x: x[1], reverse=True)
        
        swapped_ids = []
        current_mem_used = psutil.virtual_memory().used

        for knot_id, score in sorted_knots:
            if current_mem_used <= self.threshold * 0.9: # Stop when we reach 90% of threshold
                break

            tensor = active_knots[knot_id]
            
            # 3. Apply 4-bit Topological Phase Quantization (TPQ)
            # This maintains manifold integrity (η) while reducing footprint
            quantized_knot = self.quantizer.quantize(tensor)
            
            # 4. Page to SSD via Recursive Sub-Knot Hashing (RSKH)
            success = self.paging_system.page_out(knot_id, quantized_knot)
            
            if success:
                swapped_ids.append(knot_id)
                # Update memory tracking (approximate)
                current_mem_used -= tensor.nbytes
                self.logger.info(f"Paged knot {knot_id} to SSD (HDI: {score:.4f})")

        return swapped_ids

    def retrieve_knot(self, knot_id: str) -> torch.Tensor:
        """
        Restores a knot from SSD and dequantizes it back to the manifold.
        """
        quantized_knot = self.paging_system.page_in(knot_id)
        if quantized_knot is None:
            raise KeyError(f"Knot {knot_id} not found in SSD vault.")
            
        # Dequantize back to full precision for active reasoning
        return self.quantizer.dequantize(quantized_knot)

# Experimental: Automated Paging Hook
def apply_paging_guard(trainer: Any, service: SpectralPagingService):
    """
    Injects the SpectralPagingService into a training/inference loop.
    """
    def hook():
        if service.check_memory_pressure():
            # Access trainer's internal manifold state
            swapped = service.monitor_and_swap(trainer.manifold.active_knots)
            for sid in swapped:
                del trainer.manifold.active_knots[sid]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
    return hook