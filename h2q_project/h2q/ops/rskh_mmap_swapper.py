import torch
import numpy as np
import os
import mmap
from typing import Dict, Any, Optional
from h2q.persistence.rskh import RSKH
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from h2q.core.sst import SpectralShiftTracker

class RSKHMmapSwapper:
    """
    High-performance memory-mapped persistence layer for H2Q Manifold Knots.
    Uses RSKH (Reversible Spectral Knot Hash) signatures to manage hot-swapping
    between MPS VRAM and SSD, enforcing a 16GB RAM ceiling.
    """
    def __init__(
        self, 
        storage_dir: str = "/tmp/h2q_vault", 
        vram_limit_gb: float = 12.0, 
        manifold_dim: int = 256
    ):
        self.storage_dir = storage_dir
        self.vram_limit_bytes = vram_limit_gb * 1024**3
        self.manifold_dim = manifold_dim
        self.current_vram_usage = 0
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # Initialize RSKH for signature-based indexing
        self.rskh = RSKH()
        
        # Initialize DDE for swap prioritization (Avoiding 'dim' kwarg per feedback)
        config = LatentConfig()
        self.dde = DiscreteDecisionEngine(config=config)
        self.sst = SpectralShiftTracker()

        # Registry: knot_id -> {path, in_vram, last_eta, shape}
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.vram_cache: Dict[str, torch.Tensor] = {}

    def _get_path(self, knot_id: str) -> str:
        return os.path.join(self.storage_dir, f"{knot_id}.h2q_knot")

    def push(self, tensor: torch.Tensor, eta: float) -> str:
        """
        Registers a manifold knot. If VRAM is full, swaps out the knot with the 
        lowest spectral importance (eta).
        """
        # Ensure tensor is on CPU for hashing/mmap prep if needed, but keep on MPS if possible
        knot_id = self.rskh.compute_hash(tensor) if hasattr(self.rskh, 'compute_hash') else f"knot_{len(self.registry)}"
        
        tensor_bytes = tensor.element_size() * tensor.nelement()
        
        # Check VRAM constraints
        while self.current_vram_usage + tensor_bytes > self.vram_limit_bytes:
            self._evict_lowest_importance()

        # Store in VRAM
        self.vram_cache[knot_id] = tensor.to("mps")
        self.current_vram_usage += tensor_bytes
        
        self.registry[knot_id] = {
            "path": self._get_path(knot_id),
            "in_vram": True,
            "eta": eta,
            "shape": tensor.shape,
            "dtype": tensor.dtype
        }
        
        return knot_id

    def fetch(self, knot_id: str) -> torch.Tensor:
        """Retrieves a knot, swapping it back into VRAM if it was persisted to SSD."""
        if knot_id not in self.registry:
            raise KeyError(f"Knot {knot_id} not found in RSKH Registry.")

        meta = self.registry[knot_id]
        
        if meta["in_vram"]:
            return self.vram_cache[knot_id]

        # Swap In from SSD
        tensor_bytes = np.prod(meta["shape"]) * torch.tensor([], dtype=meta["dtype"]).element_size()
        
        while self.current_vram_usage + tensor_bytes > self.vram_limit_bytes:
            self._evict_lowest_importance()

        # Use mmap for O(1) disk access
        with open(meta["path"], "rb") as f:
            mmapped_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Convert mmap buffer to numpy then torch
            arr = np.frombuffer(mmapped_data, dtype=np.float32).reshape(meta["shape"])
            tensor = torch.from_numpy(arr.copy()).to("mps")
            mmapped_data.close()

        self.vram_cache[knot_id] = tensor
        self.current_vram_usage += tensor_bytes
        meta["in_vram"] = True
        
        return tensor

    def _evict_lowest_importance(self):
        """Identifies the knot with the lowest Spectral Shift (eta) and moves it to SSD."""
        if not self.vram_cache:
            return

        # Find knot with minimum eta in VRAM
        vram_knots = [k for k, v in self.registry.items() if v["in_vram"]]
        if not vram_knots:
            return
            
        target_id = min(vram_knots, key=lambda k: self.registry[k]["eta"])
        tensor = self.vram_cache.pop(target_id)
        meta = self.registry[target_id]

        # Persist to SSD using numpy memmap for speed
        fp = np.memmap(meta["path"], dtype='float32', mode='w+', shape=meta["shape"])
        fp[:] = tensor.cpu().numpy()[:]
        fp.flush()
        del fp

        tensor_bytes = tensor.element_size() * tensor.nelement()
        self.current_vram_usage -= tensor_bytes
        meta["in_vram"] = False
        
        # Clear MPS memory
        del tensor
        if torch.backends.mps.is_available():
            torch.backends.mps.empty_cache()

    def audit_persistence(self) -> Dict[str, Any]:
        """Returns metrics on VRAM vs SSD distribution."""
        vram_count = sum(1 for m in self.registry.values() if m["in_vram"])
        ssd_count = len(self.registry) - vram_count
        return {
            "vram_usage_gb": self.current_vram_usage / 1024**3,
            "vram_knots": vram_count,
            "ssd_knots": ssd_count,
            "total_knots": len(self.registry)
        }
