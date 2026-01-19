import torch
import psutil
import logging
from typing import Dict, Any, Optional
from h2q.core.memory.rskh_ssd_paging import RSKH_SSD_Paging_System, apply_spectral_paging_policy
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class Unified_Homeostatic_Orchestrator:
    """
    The Unified Homeostatic Orchestrator manages the balance between cognitive performance 
    and hardware constraints (Mac Mini M4 / 16GB RAM).
    
    It integrates the RSKH SSD Paging System to automate the offloading of 'frozen' 
    manifold knots when memory pressure exceeds the 14GB threshold.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_threshold_gb = self.config.get("memory_threshold_gb", 14.0)
        
        # Initialize the Paging System for NVMe offloading
        self.paging_system = RSKH_SSD_Paging_System(
            cache_dir=self.config.get("ssd_cache_path", "./vault/ssd_paging"),
            max_ram_usage_gb=self.memory_threshold_gb
        )
        
        # Initialize the Spectral Shift Tracker for monitoring manifold activity
        self.sst = SpectralShiftTracker()
        
        # FIX: Using get_canonical_dde to avoid 'dim' keyword argument error
        # The DDE governs the decision to offload vs. compress
        self.dde = get_canonical_dde(alpha=0.5, epsilon=0.1)
        
        logging.info(f"[Orchestrator] Initialized with {self.memory_threshold_gb}GB RAM threshold.")

    def get_current_memory_usage_gb(self) -> float:
        """Returns the current RAM usage of the process in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)

    def homeostatic_memory_guard(self, manifold_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Monitors RAM and triggers RSKH offloading if usage > 14GB.
        Identifies 'frozen' knots based on Spectral Shift (eta) and access frequency.
        """
        current_usage = self.get_current_memory_usage_gb()
        status = {"action": "none", "usage_gb": current_usage}

        if current_usage > self.memory_threshold_gb:
            logging.warning(f"[Homeostasis] Memory Pressure Detected: {current_usage:.2f}GB > {self.memory_threshold_gb}GB")
            
            # Apply spectral paging policy to identify and offload low-activity knots
            # Knots with low eta (cognitive deflection) are considered 'frozen'
            paging_results = apply_spectral_paging_policy(
                manifold=manifold_state,
                paging_system=self.paging_system,
                sst_tracker=self.sst
            )
            
            status.update({
                "action": "offload_to_ssd",
                "knots_offloaded": paging_results.get("count", 0),
                "memory_reclaimed_gb": paging_results.get("reclaimed_gb", 0.0)
            })
            
            logging.info(f"[Homeostasis] Offloaded {status['knots_offloaded']} knots to NVMe.")
        
        return status

    def step(self, manifold: Any):
        """
        Standard execution step for the orchestrator.
        Ensures logical veracity and manifold stability.
        """
        # 1. Check Memory Homeostasis
        # Assuming manifold has a state dictionary of knots
        if hasattr(manifold, 'get_knot_state'):
            self.homeostatic_memory_guard(manifold.get_knot_state())
        
        # 2. Verify Manifold Stability (Placeholder for HDI logic)
        # hdi = manifold.calculate_hdi()
        # if hdi > threshold: manifold.inject_fractal_noise()
        
        pass

def get_orchestrator(config: Optional[Dict[str, Any]] = None) -> Unified_Homeostatic_Orchestrator:
    return Unified_Homeostatic_Orchestrator(config)