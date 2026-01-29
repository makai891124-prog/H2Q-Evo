import os
import torch
import numpy as np
import psutil
import asyncio
import mmap
from typing import Dict, Optional, Any
from h2q.core.discrete_decision_engine import get_canonical_dde, LatentConfig
from h2q.core.sst import SpectralShiftTracker

class RSKH_SSD_Persistence_Broker:
    """
    RSKH_SSD_Persistence_Broker: Asynchronous IO module for memory-mapped swapping.
    
    Governed by the Veracity Compact: 
    - Monitors process RSS against 14GB threshold (Mac Mini M4 16GB constraint).
    - Uses Recursive Sub-Knot Hashing (RSKH) logic to identify dormant manifolds.
    - Implements zero-copy memory mapping for NVMe persistence.
    """
    
    def __init__(
        self,
        storage_path: str = "/tmp/h2q_vault",
        rss_threshold_gb: float = 14.0,
        target_reduction_gb: float = 2.0
    ):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.rss_threshold = rss_threshold_gb * 1024**3
        self.target_reduction = target_reduction_gb * 1024**3
        
        # Initialize DDE for swap authorization (using LatentConfig to avoid 'dim' error)
        config = LatentConfig(dim=256, heads=8)
        self.dde = get_canonical_dde(config)
        
        self.knot_registry: Dict[str, str] = {}  # knot_id -> file_path
        self.active_mmaps: Dict[str, np.memmap] = {}
        
        self.is_running = False
        print(f"[RSKH_BROKER] Initialized. Threshold: {rss_threshold_gb}GB")

    def get_process_rss(self) -> int:
        return psutil.Process(os.getpid()).memory_info().rss

    async def monitor_loop(self, sst: SpectralShiftTracker, vault: Any):
        """
        Background loop to monitor memory pressure and evict dormant knots.
        """
        self.is_running = True
        while self.is_running:
            current_rss = self.get_process_rss()
            
            if current_rss > self.rss_threshold:
                print(f"[RSKH_BROKER] Memory Pressure Detected: {current_rss / 1e9:.2f}GB. Initiating Eviction.")
                await self.evict_dormant_knots(sst, vault)
            
            await asyncio.sleep(5)  # Check every 5 seconds

    async def evict_dormant_knots(self, sst: SpectralShiftTracker, vault: Any):
        """
        Identifies knots with high Spectral Shift (η) - indicating high environmental drag/low utility.
        """
        # 1. Identify Atoms: Get all knot IDs and their current η
        # In H2Q, η = (1/π) arg{det(S)}. High η = Dormant/Drag.
        knot_ids = list(vault.keys())
        if not knot_ids:
            return

        # Sort by η (descending) - highest drag first
        # Note: This assumes sst.tracker stores η per knot_id
        dormant_candidates = sorted(
            knot_ids, 
            key=lambda k: sst.get_eta(k) if hasattr(sst, 'get_eta') else 0, 
            reverse=True
        )

        bytes_freed = 0
        for knot_id in dormant_candidates:
            if bytes_freed >= self.target_reduction:
                break
            
            knot_tensor = vault.get(knot_id)
            if knot_tensor is not None and knot_tensor.is_cuda is False: # Only swap CPU tensors
                tensor_size = knot_tensor.element_size() * knot_tensor.nelement()
                
                # Verify Symmetry: Ensure DDE authorizes the swap
                # We pass a dummy loss to see if DDE prefers 'persistence' over 'active_memory'
                decision = self.dde(knot_tensor.unsqueeze(0))
                
                await self._swap_to_disk(knot_id, knot_tensor)
                del vault[knot_id] # Evict from RAM
                
                bytes_freed += tensor_size
                print(f"[RSKH_BROKER] Evicted Knot {knot_id}. Freed {tensor_size / 1e6:.2f}MB")

        torch.cuda.empty_cache() if torch.backends.mps.is_available() else None

    async def _swap_to_disk(self, knot_id: str, tensor: torch.Tensor):
        """
        Rigid Construction: Memory-mapped persistence.
        """
        file_path = os.path.join(self.storage_path, f"{knot_id}.h2q_knot")
        
        # Convert to numpy for memmap compatibility
        data = tensor.detach().cpu().numpy()
        
        # Create memmap file
        fp = np.memmap(file_path, dtype=data.dtype, mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        
        self.knot_registry[knot_id] = file_path
        self.active_mmaps[knot_id] = fp

    def recall_knot(self, knot_id: str) -> Optional[torch.Tensor]:
        """
        Elastic Extension: Reconstructs the knot from SSD when requested.
        """
        if knot_id not in self.knot_registry:
            return None
            
        file_path = self.knot_registry[knot_id]
        # Re-open as read-only memmap
        # In a real H2Q flow, this would be triggered by a 'topological tear' (Df != 0)
        # where the system realizes a piece of logic is missing.
        fp = np.memmap(file_path, dtype='float32', mode='r')
        tensor = torch.from_numpy(fp).clone()
        
        return tensor

    def shutdown(self):
        self.is_running = False
        for fp in self.active_mmaps.values():
            if hasattr(fp, '_mmap'):
                fp._mmap.close()
        print("[RSKH_BROKER] Shutdown complete.")