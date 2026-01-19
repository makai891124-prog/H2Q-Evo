import torch
import psutil
import logging
from typing import Dict, List, Optional
from h2q.core.memory.rskh_ssd_paging import RSKH_SSD_Paging_System, KnotMetadata
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

# [STABLE] Geodesic Paging Dispatcher for M4 Silicon
# Logic: Automates the migration of low-η (frozen) knots to NVMe when Unified Memory pressure > 14GB.

class GeodesicPagingDispatcher:
    def __init__(
        self, 
        paging_system: RSKH_SSD_Paging_System, 
        memory_threshold_gb: float = 14.0,
        frozen_eta_threshold: float = 0.15
    ):
        """
        Initializes the dispatcher with a specific RSKH SSD backend.
        
        Args:
            paging_system: The RSKH SSD Paging System instance.
            memory_threshold_gb: Threshold in GB to trigger paging (Default 14GB for 16GB M4).
            frozen_eta_threshold: η value below which a knot is considered 'frozen'.
        """
        self.paging_system = paging_system
        self.threshold_bytes = memory_threshold_gb * 1024**3
        self.frozen_eta_threshold = frozen_eta_threshold
        
        # Use canonical DDE to avoid 'dim' keyword argument errors identified in feedback
        self.dde = get_canonical_dde()
        
        # Registry of active knots in Unified Memory: {knot_id: (tensor, sst_instance)}
        self.active_registry: Dict[str, tuple] = {}
        
        logging.info(f"[M24-CW] GeodesicPagingDispatcher initialized. Threshold: {memory_threshold_gb}GB")

    def register_knot(self, knot_id: str, tensor: torch.Tensor, sst: SpectralShiftTracker):
        """Registers a new knot for potential paging."""
        self.active_registry[knot_id] = (tensor, sst)

    def check_memory_pressure(self) -> bool:
        """Checks if current Unified Memory usage exceeds the safety threshold."""
        mem = psutil.virtual_memory()
        return mem.used > self.threshold_bytes

    def get_frozen_knots(self) -> List[str]:
        """
        Identifies knots with η (Spectral Shift) below the frozen threshold.
        Returns a list of knot_ids sorted by η (ascending).
        """
        frozen = []
        for knot_id, (tensor, sst) in self.active_registry.items():
            # η = (1/π) arg{det(S)}
            current_eta = sst.get_current_eta() if hasattr(sst, 'get_current_eta') else 0.0
            if current_eta < self.frozen_eta_threshold:
                frozen.append((knot_id, current_eta))
        
        # Sort by η: lowest η (most frozen) first
        frozen.sort(key=lambda x: x[1])
        return [x[0] for x in frozen]

    def dispatch_cycle(self):
        """
        Executes a single monitoring and paging cycle.
        If pressure is high, evicts frozen knots until pressure is relieved or no frozen knots remain.
        """
        if not self.check_memory_pressure():
            return

        logging.warning("[M24-CW] High Memory Pressure Detected (>14GB). Initiating Geodesic Paging.")
        
        frozen_ids = self.get_frozen_knots()
        
        for knot_id in frozen_ids:
            if not self.check_memory_pressure():
                logging.info("[M24-CW] Memory pressure stabilized. Stopping dispatch.")
                break
            
            tensor, sst = self.active_registry.pop(knot_id)
            eta = sst.get_current_eta() if hasattr(sst, 'get_current_eta') else 0.0
            
            # Construct Metadata for RSKH Persistence
            metadata = KnotMetadata(
                knot_id=knot_id,
                eta=eta,
                timestamp=torch.cuda.Event() if torch.backends.mps.is_available() else None
            )
            
            # Execute Paging to NVMe
            success = self.paging_system.page_out(knot_id, tensor, metadata)
            
            if success:
                logging.info(f"[M24-CW] Paged out knot {knot_id} (η={eta:.4f}) to NVMe.")
                # Explicitly clear tensor from memory
                del tensor
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            else:
                logging.error(f"[M24-CW] Failed to page out knot {knot_id}.")

    def retrieve_knot(self, knot_id: str) -> Optional[torch.Tensor]:
        """Retrieves a knot from NVMe back into Unified Memory."""
        tensor = self.paging_system.page_in(knot_id)
        if tensor is not None:
            # Note: SST state should be managed by the caller or a persistent SST registry
            logging.info(f"[M24-CW] Knot {knot_id} retrieved from NVMe.")
            return tensor
        return None