import torch
import psutil
import os
import time
import shutil
import logging
from typing import Dict, Optional

# Configure logging for memory telemetry
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("H2Q-MemoryManager")

class DynamicAMXMemorySwapper:
    """
    M4-optimized buffer manager for H2Q SU(2) manifold segments.
    Utilizes psutil telemetry to offload dormant tensors to system storage
    to maintain the 16GB Unified Memory constraint during high-pressure Wake phases.
    """
    def __init__(
        self, 
        critical_threshold_pct: float = 85.0, 
        safe_threshold_pct: float = 65.0, 
        swap_dir: str = "./.h2q_swap_cache"
    ):
        self.critical_threshold = critical_threshold_pct
        self.safe_threshold = safe_threshold_pct
        self.swap_dir = swap_dir
        
        # Registry: {name: {"tensor": Optional[torch.Tensor], "path": str, "last_used": float, "on_disk": bool}}
        self.registry: Dict[str, Dict] = {}
        
        if not os.path.exists(self.swap_dir):
            os.makedirs(self.swap_dir)
            
        logger.info(f"DynamicAMXMemorySwapper initialized. Swap directory: {self.swap_dir}")

    def register_manifold(self, name: str, tensor: torch.Tensor):
        """
        Registers a new SU(2) manifold segment into the swapper.
        Initially keeps the tensor on the current device (usually MPS).
        """
        swap_path = os.path.join(self.swap_dir, f"{name}_{id(tensor)}.pt")
        self.registry[name] = {
            "tensor": tensor,
            "path": swap_path,
            "last_used": time.time(),
            "on_disk": False
        }

    def access(self, name: str) -> torch.Tensor:
        """
        Retrieves a manifold segment. If it was offloaded to disk, it is reloaded to MPS.
        """
        if name not in self.registry:
            raise KeyError(f"Manifold segment '{name}' is not registered in the swapper.")

        entry = self.registry[name]
        entry["last_used"] = time.time()

        if entry["on_disk"]:
            logger.info(f"Reloading dormant segment '{name}' from disk to MPS...")
            # Load to CPU first, then move to MPS to ensure clean allocation
            loaded_tensor = torch.load(entry["path"], weights_only=True).to("mps")
            entry["tensor"] = loaded_tensor
            entry["on_disk"] = False
            
            # Clean up the swap file
            if os.path.exists(entry["path"]):
                os.remove(entry["path"])

        return entry["tensor"]

    def synchronize_pressure(self):
        """
        Audits system memory pressure and offloads Least Recently Used (LRU) 
        dormant segments if pressure exceeds the critical threshold.
        """
        mem = psutil.virtual_memory()
        current_usage = mem.percent

        if current_usage > self.critical_threshold:
            logger.warning(f"High memory pressure detected: {current_usage}%. Initiating offload...")
            
            # Identify segments currently in memory, sorted by last_used (LRU)
            active_segments = sorted(
                [k for k, v in self.registry.items() if not v["on_disk"]],
                key=lambda k: self.registry[k]["last_used"]
            )

            for name in active_segments:
                self._offload_to_disk(name)
                
                # Re-check pressure after each offload and cache clear
                if psutil.virtual_memory().percent < self.safe_threshold:
                    logger.info(f"Memory pressure stabilized at {psutil.virtual_memory().percent}%")
                    break

    def _offload_to_disk(self, name: str):
        """
        Moves a tensor from MPS to Disk via CPU serialization.
        """
        entry = self.registry[name]
        logger.info(f"Offloading segment '{name}' to disk storage.")
        
        # Move to CPU to free up Unified Memory/MPS allocation
        cpu_tensor = entry["tensor"].to("cpu")
        torch.save(cpu_tensor, entry["path"])
        
        # Clear references
        entry["tensor"] = None
        entry["on_disk"] = True
        
        # Explicitly empty MPS cache to return memory to the M4 Unified pool
        torch.mps.empty_cache()

    def purge_swap(self):
        """
        Cleans up all swap files and resets the registry.
        """
        if os.path.exists(self.swap_dir):
            shutil.rmtree(self.swap_dir)
        self.registry.clear()
        logger.info("Swap cache purged and registry cleared.")

# Note on DiscreteDecisionEngine Initialization Error:
# The registry indicates conflicting signatures for DiscreteDecisionEngine.
# To avoid the 'unexpected keyword argument dim' error, use the canonical 
# factory method if available: h2q.core.discrete_decision_engine.get_canonical_dde(config)
