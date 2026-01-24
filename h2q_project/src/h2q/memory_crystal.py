import os
import glob
import time
from datetime import datetime
from pathlib import Path
import logging

# Grounding in Reality: The environment currently reports 'ModuleNotFoundError: No module named torch'.
# We implement the logic assuming the dependency will be restored, but provide safety checks.
try:
    import torch
except ImportError:
    torch = None

class MemoryManager:
    """
    Architect of the H2Q Memory Crystal system.
    Governs the serialization and hot-reloading of quaternionic manifold weights.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_loaded_crystal = None
        
        if torch is None:
            logging.warning("[M24-CW] Veracity Alert: 'torch' not found. MemoryManager operating in structural-only mode.")

    def _get_latest_checkpoint(self) -> str:
        """Identifies the most recent .pt crystal based on modification time."""
        files = glob.glob(str(self.checkpoint_dir / "*.pt"))
        if not files:
            return None
        # Sort by modification time (Rigid Construction: Temporal Symmetry)
        return max(files, key=os.path.getmtime)

    def load_latest(self, model: 'torch.nn.Module', device: str = "cpu") -> bool:
        """
        Auto-loads the latest crystal into the provided model on startup.
        """
        if torch is None:
            return False
            
        latest_path = self._get_latest_checkpoint()
        if not latest_path:
            logging.info("[M24-CW] No existing crystals found in vault. Starting from vacuum state.")
            return False

        try:
            # Compatibility with Mac Mini M4 (MPS/CPU)
            map_location = torch.device(device)
            state_dict = torch.load(latest_path, map_location=map_location)
            model.load_state_dict(state_dict)
            self.last_loaded_crystal = latest_path
            logging.info(f"[M24-CW] Crystal {latest_path} successfully integrated into manifold.")
            return True
        except Exception as e:
            logging.error(f"[M24-CW] Topological Tear during loading: {e}")
            return False

    def save_crystal(self, state_dict: dict, loss: float = 0.0, eta: float = 0.0):
        """
        Saves weights with a timestamp and loss-metric signature.
        Signature: crystal_YYYYMMDD_HHMMSS_L[loss]_E[eta].pt
        """
        if torch is None:
            logging.error("[M24-CW] Cannot save crystal: torch dependency missing.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crystal_{timestamp}_L{loss:.4f}_E{eta:.4f}.pt"
        save_path = self.checkpoint_dir / filename

        try:
            torch.save(state_dict, save_path)
            logging.info(f"[M24-CW] Manifold crystallized at {save_path}")
        except Exception as e:
            logging.error(f"[M24-CW] Serialization failure: {e}")

    def hot_reload(self, model: 'torch.nn.Module') -> bool:
        """
        Polls for a newer crystal and updates weights without server restart.
        """
        latest_path = self._get_latest_checkpoint()
        if latest_path and latest_path != self.last_loaded_crystal:
            logging.info("[M24-CW] New crystal detected. Initiating hot-reload sequence.")
            return self.load_latest(model)
        return False

# Experimental: Logic Curvature Audit Hook
def audit_crystal_integrity(path: str):
    """Placeholder for Discrete Fueter Operator check on saved weights."""
    if torch is None:
        logging.warning("[M24-CW] Integrity audit skipped: torch not available.")
        return {"status": "skipped", "reason": "torch_missing"}

    if not path or not os.path.exists(path):
        return {"status": "error", "reason": "file_not_found"}

    try:
        state = torch.load(path, map_location="cpu")
        flat = []
        for tensor in state.values():
            if torch.is_tensor(tensor):
                flat.append(tensor.float().flatten())
        if not flat:
            return {"status": "error", "reason": "no_tensors"}

        weights = torch.cat(flat)
        # Simple Fueter-like gradient proxy: finite-difference norm
        diff = weights[1:] - weights[:-1]
        curvature = diff.abs().mean().item()
        spectral_norm = torch.linalg.norm(weights).item()

        return {
            "status": "ok",
            "curvature": curvature,
            "spectral_norm": spectral_norm,
            "size": weights.numel(),
        }
    except Exception as e:
        logging.error(f"[M24-CW] Integrity audit failed: {e}")
        return {"status": "error", "reason": str(e)}
