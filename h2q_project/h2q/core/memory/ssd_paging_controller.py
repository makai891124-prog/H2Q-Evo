import os
import time
import psutil
import torch
from typing import Dict, Optional, List
from dataclasses import dataclass

# Internal H2Q Imports based on Registry
from h2q.core.memory.rskh_vault import RSKHVault
from h2q.core.sst import SpectralShiftTracker
from h2q.core.discrete_decision_engine import get_canonical_dde

@dataclass
class KnotMetadata:
    knot_id: str
    last_access: float
    hdi: float  # Heat Death Index (0.0 = Active, 1.0 = Dormant)
    in_ram: bool = True
    file_path: Optional[str] = None

class SSDPagingController:
    """
    LRU-HDI (Least Recently Used - Heat Death Index) Paging Controller.
    Enforces a 14GB RSS limit for Mac Mini M4 (16GB) constraints.
    """
    def __init__(self, 
                 vault_path: str = "data/vault/knots", 
                 rss_threshold_gb: float = 14.0):
        self.vault_path = vault_path
        os.makedirs(self.vault_path, exist_ok=True)
        
        self.rss_threshold = rss_threshold_gb * 1024**3
        self.registry: Dict[str, KnotMetadata] = {}
        self.vault = RSKHVault() # Registry: h2q.core.memory.rskh_vault
        
        # Initialize DDE without 'dim' to avoid previous runtime error
        self.dde = get_canonical_dde()
        
        self.process = psutil.Process(os.getpid())

    def update_knot_telemetry(self, knot_id: str, eta: float):
        """
        Updates access time and calculates HDI based on Spectral Shift (eta).
        HDI = 1.0 - tanh(eta). High eta (learning) = Low HDI (hot).
        """
        hdi = 1.0 - torch.tanh(torch.tensor(eta)).item()
        if knot_id in self.registry:
            self.registry[knot_id].last_access = time.time()
            self.registry[knot_id].hdi = hdi
        else:
            self.registry[knot_id] = KnotMetadata(
                knot_id=knot_id, 
                last_access=time.time(), 
                hdi=hdi
            )

    def check_memory_pressure(self) -> bool:
        """Returns True if RSS exceeds threshold."""
        return self.process.memory_info().rss > self.rss_threshold

    def _calculate_eviction_score(self, meta: KnotMetadata) -> float:
        """
        Score = (Time Since Access) * HDI.
        Higher score = Higher priority for eviction to SSD.
        """
        idle_time = time.time() - meta.last_access
        return idle_time * (meta.hdi + 1e-6)

    def enforce_hygiene(self, active_knots: Dict[str, torch.Tensor]):
        """
        Main loop to offload dormant knots if memory pressure is high.
        """
        if not self.check_memory_pressure():
            return

        # Identify candidates currently in RAM
        candidates = [k for k, v in self.registry.items() if v.in_ram and k in active_knots]
        if not candidates:
            return

        # Sort by LRU-HDI score descending
        candidates.sort(key=lambda k: self._calculate_eviction_score(self.registry[k]), reverse=True)

        while self.check_memory_pressure() and candidates:
            target_id = candidates.pop(0)
            self._offload_to_ssd(target_id, active_knots[target_id])
            del active_knots[target_id] # Remove from RAM

    def _offload_to_ssd(self, knot_id: str, tensor: torch.Tensor):
        """Serializes knot to NVMe via RSKH Vault."""
        path = os.path.join(self.vault_path, f"{knot_id}.h2q")
        torch.save(tensor, path)
        
        self.registry[knot_id].in_ram = False
        self.registry[knot_id].file_path = path
        
        # Log for Holomorphic Auditing
        print(f"[LRU-HDI] Offloaded knot {knot_id} to SSD (HDI: {self.registry[knot_id].hdi:.4f})")

    def fetch_knot(self, knot_id: str) -> torch.Tensor:
        """Retrieves knot from SSD if not in RAM."""
        meta = self.registry.get(knot_id)
        if not meta:
            raise KeyError(f"Knot {knot_id} not found in registry.")

        if not meta.in_ram:
            if not meta.file_path or not os.path.exists(meta.file_path):
                raise FileNotFoundError(f"Knot {knot_id} data lost on disk.")
            
            tensor = torch.load(meta.file_path)
            meta.in_ram = True
            meta.last_access = time.time()
            return tensor
        
        return None # Already in RAM, handled by caller

    def audit_paging_integrity(self):
        """Veracity Compact: Ensure no topological tears in the paging registry."""
        for knot_id, meta in self.registry.items():
            if not meta.in_ram and (not meta.file_path or not os.path.exists(meta.file_path)):
                raise RuntimeError(f"Topological Tear: Knot {knot_id} registered as SSD but file missing.")
