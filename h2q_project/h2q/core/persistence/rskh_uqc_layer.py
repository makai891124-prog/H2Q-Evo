import os
import torch
import numpy as np
import hashlib
import json
from typing import Dict, Any, Tuple
from h2q.core.discrete_decision_engine import get_canonical_dde

class RSKH_UQC_Persistence:
    """
    RSKH-UQC Unified Persistence Layer.
    Handles bit-accurate serialization of SU(2) manifolds with SHA-256 veracity checksums
    and O(1) memory-mapped retrieval for Mac Mini M4 (16GB) constraints.
    """

    def __init__(self):
        # Initialize DDE without 'dim' to avoid the reported Runtime Error
        self.dde = get_canonical_dde()
        self.header_size = 4096  # Fixed header size for O(1) offset calculation
        self.magic_bytes = b"H2Q-RSKH-UQC-V1"

    def _calculate_checksum(self, data: bytes) -> str:
        """Generates SHA-256 hash for veracity auditing."""
        return hashlib.sha256(data).hexdigest()

    def save_manifold(self, manifold_tensor: torch.Tensor, path: str, metadata: Dict[str, Any] = None):
        """
        Serializes the manifold state with a veracity header.
        """
        if manifold_tensor.shape[-1] != 256:
            raise ValueError(f"Invalid manifold dimension: {manifold_tensor.shape[-1]}. Expected 256.")

        # Ensure CPU for serialization stability
        data_np = manifold_tensor.detach().cpu().numpy()
        data_bytes = data_np.tobytes()
        checksum = self._calculate_checksum(data_bytes)

        header = {
            "magic": self.magic_bytes.decode('ascii'),
            "shape": list(data_np.shape),
            "dtype": str(data_np.dtype),
            "checksum": checksum,
            "metadata": metadata or {},
            "η_signature": metadata.get("eta", 0.0) if metadata else 0.0
        }

        header_json = json.dumps(header).encode('utf-8')
        if len(header_json) > self.header_size - 8:
            raise OverflowError("Metadata too large for fixed header size.")

        with open(path, "wb") as f:
            f.write(self.magic_bytes)
            f.write(header_json.ljust(self.header_size - len(self.magic_bytes)))
            f.write(data_bytes)

    def load_mapped(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        O(1) Memory-mapped retrieval of the manifold.
        Verifies SHA-256 veracity before returning the map.
        """
        with open(path, "rb") as f:
            magic = f.read(len(self.magic_bytes))
            if magic != self.magic_bytes:
                raise ValueError("Invalid file format: Magic bytes mismatch.")

            header_bytes = f.read(self.header_size - len(self.magic_bytes))
            header = json.loads(header_bytes.decode('utf-8').strip())

            # Verify Veracity (Note: Full verification is O(N), but mapping is O(1))
            # We map first, then verify if requested by DDE
            f.seek(self.header_size)
            mmap_data = np.memmap(path, dtype=header['dtype'], mode='r', 
                                  offset=self.header_size, shape=tuple(header['shape']))

            # Audit Checksum
            actual_checksum = self._calculate_checksum(mmap_data.tobytes())
            if actual_checksum != header['checksum']:
                raise SecurityError(f"Veracity Violation: Checksum mismatch in {path}")

            return mmap_data, header

    def audit_manifold_integrity(self, path: str) -> bool:
        """
        Holomorphic Auditing: Checks for 'topological tears' (data corruption).
        """
        try:
            _, header = self.load_mapped(path)
            # DDE logic to decide if the η_signature is within analytic bounds
            decision = self.dde.decide(torch.tensor([header['η_signature']]))
            return True if decision > 0.5 else False
        except Exception as e:
            print(f"[Holomorphic Audit] Failure: {e}")
            return False

class SecurityError(Exception): pass
