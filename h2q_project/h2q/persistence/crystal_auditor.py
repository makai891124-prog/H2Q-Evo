import torch
import hashlib
import json
import os
from typing import Dict, Any, Optional
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.calibration.berry_phase import BerryPhaseCalibrator
from h2q.quaternion_ops import quaternion_norm

class CrystalAuditor:
    """
    Performs bit-accurate veracity checks and Berry Phase validation on .h2q memory crystals.
    Ensures reloaded weights preserve the exact manifold holonomy of the training state.
    """

    def __init__(self, dde_config: Optional[Dict[str, Any]] = None):
        # Use canonical DDE to avoid 'dim' keyword errors identified in feedback
        self.dde = get_canonical_dde()
        self.calibrator = BerryPhaseCalibrator()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def calculate_sha256(self, file_path: str) -> str:
        """Computes SHA-256 hash of the crystal file for bit-level veracity."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_holonomy(self, state_dict: Dict[str, torch.Tensor], tolerance: float = 1e-6) -> bool:
        """
        Enforces SU(2) symmetry by verifying that quaternionic atoms (w, i, j, k) 
        maintain unit norm (S³ isomorphism).
        """
        for name, tensor in state_dict.items():
            if "weight" in name and tensor.dim() >= 1:
                # Assuming quaternionic weights are stored as [..., 4]
                if tensor.shape[-1] == 4:
                    norms = quaternion_norm(tensor)
                    deviation = torch.abs(norms - 1.0).max().item()
                    if deviation > tolerance:
                        print(f"[HOLONOMY_ERROR] Layer {name} deviation: {deviation}")
                        return False
        return True

    def verify_berry_phase(self, 
                           state_dict: Dict[str, torch.Tensor], 
                           expected_phase: float, 
                           tolerance: float = 1e-4) -> bool:
        """
        Calculates the Pancharatnam-Berry phase of the reloaded manifold 
        and compares it against the training-time signature.
        """
        # Extract a representative manifold slice for phase calculation
        # In H2Q, this is typically the scattering matrix trace
        current_phase = self.calibrator.calculate_geometric_phase(state_dict)
        
        phase_diff = torch.abs(torch.tensor(current_phase - expected_phase))
        if phase_diff > tolerance:
            print(f"[PHASE_DRIFT] Expected: {expected_phase}, Found: {current_phase}")
            return False
        return True

    def audit_crystal(self, crystal_path: str, metadata_path: str) -> Dict[str, Any]:
        """
        Full audit pipeline: SHA-256 -> Holonomy -> Berry Phase.
        """
        if not os.path.exists(crystal_path):
            return {"status": "FAILED", "reason": "Crystal file not found"}

        # 1. Veracity Check
        actual_hash = self.calculate_sha256(crystal_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if actual_hash != metadata.get("sha256"):
            return {"status": "FAILED", "reason": "SHA-256 Mismatch - Crystal Corrupted"}

        # 2. Load Tensors
        state_dict = torch.load(crystal_path, map_location=self.device)

        # 3. Holonomy Audit
        holonomy_valid = self.verify_holonomy(state_dict)
        if not holonomy_valid:
            return {"status": "FAILED", "reason": "SU(2) Symmetry Broken"}

        # 4. Berry Phase Audit
        expected_phase = metadata.get("berry_phase", 0.0)
        phase_valid = self.verify_berry_phase(state_dict, expected_phase)
        if not phase_valid:
            return {"status": "FAILED", "reason": "Berry Phase Drift Detected"}

        return {
            "status": "PASSED",
            "sha256": actual_hash,
            "holonomy_deviation": "< 1e-6",
            "berry_phase": expected_phase
        }

if __name__ == "__main__":
    # Experimental usage
    auditor = CrystalAuditor()
    print("[AUDITOR] Initialized with Canonical DDE.")
