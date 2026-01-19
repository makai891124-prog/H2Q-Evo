import os
import torch
import psutil
import pickle
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Internal H2Q Imports based on Registry
from h2q.persistence.rskh import RSKH, SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde

@dataclass
class KnotMetadata:
    rskh_hash: str
    eta_history: List[float]
    last_access: float
    is_on_disk: bool = False
    file_path: Optional[str] = None

class SpectralSwapManager:
    """
    Unified persistence controller for the H2Q Manifold.
    Uses η-volatility (Spectral Shift variance) to determine swap candidates.
    Optimized for Mac Mini M4 (16GB RAM) constraints.
    """
    def __init__(
        self,
        storage_dir: str = "vault/spectral_swap",
        ram_threshold_pct: float = 85.0,
        volatility_window: int = 10,
        swap_threshold_eta: float = 0.05
    ):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.ram_threshold_pct = ram_threshold_pct
        self.volatility_window = volatility_window
        self.swap_threshold_eta = swap_threshold_eta
        
        # Initialize Core Components
        self.sst = SpectralShiftTracker()
        self.dde = get_canonical_dde() # Avoids 'dim' keyword error
        self.rskh_engine = RSKH(self.dde, self.sst)
        
        # Registry of managed knots
        self.registry: Dict[str, KnotMetadata] = {}
        
        # Experimental: Tracking swap latency for Elastic Extension
        self.swap_metrics = {"out_count": 0, "in_count": 0, "avg_latency": 0.0}

    def _calculate_volatility(self, eta_history: List[float]) -> float:
        """Calculates the variance of spectral shift over the window."""
        if len(eta_history) < 2:
            return 1.0 # High volatility by default for new knots
        return float(np.std(eta_history[-self.volatility_window:]))

    def register_knot(self, knot_tensor: torch.Tensor, eta: float):
        """Generates RSKH signature and updates metadata."""
        # Ensure tensor is on CPU for hashing/persistence to save MPS memory
        knot_cpu = knot_tensor.detach().cpu()
        rskh_hash = self.rskh_engine.compute_hash(knot_cpu)
        
        if rskh_hash not in self.registry:
            self.registry[rskh_hash] = KnotMetadata(
                rskh_hash=rskh_hash,
                eta_history=[eta],
                last_access=time.time()
            )
        else:
            self.registry[rskh_hash].eta_history.append(eta)
            self.registry[rskh_hash].last_access = time.time()
            
        return rskh_hash

    def check_memory_pressure(self) -> bool:
        """Telemetry via psutil to monitor 16GB ceiling."""
        mem = psutil.virtual_memory()
        return mem.percent > self.ram_threshold_pct

    def perform_spectral_swap(self, active_knots: Dict[str, torch.Tensor]):
        """
        Identifies low-η-volatility knots and offloads them to SSD.
        """
        if not self.check_memory_pressure():
            return active_knots

        candidates = []
        for rskh_hash, meta in self.registry.items():
            if not meta.is_on_disk and rskh_hash in active_knots:
                volatility = self._calculate_volatility(meta.eta_history)
                if volatility < self.swap_threshold_eta:
                    candidates.append((rskh_hash, volatility))

        # Sort by lowest volatility (most stable knots first)
        candidates.sort(key=lambda x: x[1])

        for rskh_hash, _ in candidates:
            if not self.check_memory_pressure():
                break
            
            self._swap_out(rskh_hash, active_knots.pop(rskh_hash))
            
        return active_knots

    def _swap_out(self, rskh_hash: str, tensor: torch.Tensor):
        """Serializes knot to SSD."""
        start_time = time.time()
        file_path = os.path.join(self.storage_dir, f"{rskh_hash}.h2q")
        
        with open(file_path, 'wb') as f:
            pickle.dump(tensor.cpu(), f)
            
        meta = self.registry[rskh_hash]
        meta.is_on_disk = True
        meta.file_path = file_path
        
        self.swap_metrics["out_count"] += 1
        self.swap_metrics["avg_latency"] = (self.swap_metrics["avg_latency"] + (time.time() - start_time)) / 2

    def swap_in(self, rskh_hash: str) -> torch.Tensor:
        """Retrieves knot from SSD and restores to active manifold."""
        if rskh_hash not in self.registry or not self.registry[rskh_hash].is_on_disk:
            raise KeyError(f"Knot {rskh_hash} not found in persistence vault.")

        start_time = time.time()
        meta = self.registry[rskh_hash]
        
        with open(meta.file_path, 'rb') as f:
            tensor = pickle.load(f)
            
        meta.is_on_disk = False
        meta.last_access = time.time()
        
        # Cleanup file to maintain SSD symmetry
        if os.path.exists(meta.file_path):
            os.remove(meta.file_path)
            
        self.swap_metrics["in_count"] += 1
        return tensor

    def audit_persistence_integrity(self) -> bool:
        """Verifies that all registered knots are either in RAM or on Disk (No Topological Tears)."""
        for rskh_hash, meta in self.registry.items():
            if meta.is_on_disk and not os.path.exists(meta.file_path):
                return False
        return True
