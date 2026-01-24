import os
import time
import psutil
import torch
from h2q.core.memory.rskh_ssd_paging import RSKH_SSD_Paging_System, KnotMetadata
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class SpectralSwapDaemon:
    """
    RSKH SSD-Paging Daemon: Automates the offloading of 'frozen' manifold knots 
    to NVMe when process RSS exceeds the safety threshold (14GB for Mac Mini M4).
    """
    def __init__(self, threshold_gb=14.0, swap_dir="/tmp/h2q_rskh_vault"):
        self.threshold = threshold_gb * 1024**3  # Convert GB to Bytes
        self.paging_system = RSKH_SSD_Paging_System(swap_dir=swap_dir)
        self.sst = SpectralShiftTracker()
        
        # FIX: Use get_canonical_dde to avoid 'unexpected keyword argument dim' error
        # The registry handles the correct instantiation parameters for the current environment.
        self.dde = get_canonical_dde()
        
        self.process = psutil.Process(os.get_pid())
        self.is_active = True

    def get_current_rss(self):
        """Returns current Resident Set Size in bytes."""
        return self.process.memory_info().rss

    def identify_frozen_knots(self, manifold_registry):
        """
        Identifies knots with low Spectral Shift (η), indicating they are 
        topologically 'frozen' and suitable for SSD offloading.
        """
        frozen_candidates = []
        for knot_id, knot_tensor in manifold_registry.items():
            # η = (1/π) arg{det(S)} calculated via SST
            eta = self.sst.calculate_shift(knot_tensor)
            
            # Threshold: η < 0.05 indicates low environmental interaction/drag
            if eta < 0.05:
                frozen_candidates.append((knot_id, eta))
        
        # Sort by η ascending (least active first)
        return sorted(frozen_candidates, key=lambda x: x[1])

    def run_swap_cycle(self, manifold_registry):
        """
        Executes a single monitoring and paging cycle.
        Returns True if paging occurred, False otherwise.
        """
        current_rss = self.get_current_rss()
        
        if current_rss < self.threshold:
            return False

        print(f"[RSKH-DAEMON] Memory Pressure Detected: {current_rss/1e9:.2f}GB / {self.threshold/1e9:.2f}GB")
        
        frozen_knots = self.identify_frozen_knots(manifold_registry)
        
        if not frozen_knots:
            print("[RSKH-DAEMON] Warning: High RSS but no frozen knots identified. Manifold Heat-Death imminent.")
            return False

        paged_count = 0
        for knot_id, eta in frozen_knots:
            if self.get_current_rss() < (self.threshold * 0.9): # Target 90% of threshold
                break

            knot_data = manifold_registry[knot_id]
            metadata = KnotMetadata(
                knot_id=knot_id, 
                eta=eta, 
                timestamp=time.time(),
                device=str(knot_data.device)
            )

            # Offload to NVMe via RSKH Paging System
            self.paging_system.page_out(knot_id, knot_data, metadata)
            
            # Remove from active manifold to free RSS
            del manifold_registry[knot_id]
            paged_count += 1

        # Explicitly clear MPS cache for Mac Mini M4 constraints
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"[RSKH-DAEMON] Paging Complete. Offloaded {paged_count} knots to SSD.")
        return True

    def monitor_loop(self, manifold_registry, interval=5.0):
        """Continuous background monitoring loop."""
        while self.is_active:
            try:
                self.run_swap_cycle(manifold_registry)
            except Exception as e:
                print(f"[RSKH-DAEMON] Error in swap cycle: {e}")
            time.sleep(interval)

def deploy_paging_daemon(manifold_registry, threshold_gb=14.0):
    """Factory function to initialize and return the daemon."""
    daemon = SpectralSwapDaemon(threshold_gb=threshold_gb)
    return daemon