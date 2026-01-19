import torch
import os
import time
from typing import Dict, Any, List
from h2q.core.memory.rskh_vault import RSKHVault
from h2q.core.logic_auditing import HolomorphicAuditKernel
from h2q.core.serialization.manifold_snapshot import ManifoldSnapshot
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class BerryPhaseCheckpointer:
    """
    Service responsible for the periodic crystallization of the RSKH Vault.
    Enforces the Veracity Compact via global Holomorphic Audits (Fueter Operator).
    """
    def __init__(self, vault: RSKHVault, checkpoint_dir: str = "checkpoints/vaults"):
        self.vault = vault
        self.checkpoint_dir = checkpoint_dir
        self.audit_kernel = HolomorphicAuditKernel()
        self.sst = SpectralShiftTracker()
        # Use canonical DDE to avoid 'dim' keyword argument errors
        self.dde = get_canonical_dde()
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def calculate_hdi(self, singular_values: torch.Tensor) -> float:
        """
        Calculates the Heat-Death Index (HDI) via Von Neumann entropy.
        H = -Σ p log(p)
        """
        p = torch.softmax(singular_values, dim=-1)
        hdi = -torch.sum(p * torch.log(p + 1e-9)).item()
        return hdi

    def perform_holomorphic_audit(self) -> Dict[str, Any]:
        """
        Performs a global audit of all context knots using the Discrete Fueter Operator.
        Df = ∂w + i∂x + j∂y + k∂z. Residuals > 0.05 signify topological tears.
        """
        knots = self.vault.get_all_knots()
        audit_results = {
            "total_knots": len(knots),
            "tears_detected": 0,
            "mean_residual": 0.0,
            "knot_status": []
        }

        total_residual = 0.0
        for knot_id, knot_tensor in knots.items():
            # Discrete Fueter Operator implementation
            # Assuming knot_tensor is [..., 4] representing (w, x, y, z)
            residual = self.audit_kernel.calculate_fueter_residual(knot_tensor)
            total_residual += residual
            
            is_hallucination = residual > 0.05
            if is_hallucination:
                audit_results["tears_detected"] += 1
            
            audit_results["knot_status"].append({
                "id": knot_id,
                "residual": residual,
                "stable": not is_hallucination
            })

        if len(knots) > 0:
            audit_results["mean_residual"] = total_residual / len(knots)
        
        return audit_results

    def crystallize(self, tag: str = "auto") -> str:
        """
        Crystallizes the current vault state into a bit-accurate .h2q format.
        """
        print(f"[BerryPhaseCheckpointer] Initiating crystallization: {tag}")
        
        # 1. Perform Holomorphic Audit
        audit_report = self.perform_holomorphic_audit()
        if audit_report["tears_detected"] > 0:
            print(f"[WARNING] Holomorphic Audit detected {audit_report['tears_detected']} topological tears.")

        # 2. Capture Manifold Snapshot
        # We extract the singular value spectrum from the vault's latent manifold
        sv_spectrum = self.vault.get_singular_spectrum()
        hdi = self.calculate_hdi(sv_spectrum)
        
        # 3. Calculate Spectral Shift (η)
        eta = self.sst.compute_eta(self.vault.get_scattering_matrix())

        snapshot = ManifoldSnapshot(
            vault_data=self.vault.state_dict(),
            hdi=hdi,
            spectral_shift=eta,
            audit_report=audit_report,
            timestamp=time.time(),
            version="1.1"
        )

        # 4. Bit-accurate serialization
        file_name = f"vault_{tag}_{int(time.time())}.h2q"
        file_path = os.path.join(self.checkpoint_dir, file_name)
        
        # Using torch.save with weights_only=False for custom H2Q objects if necessary,
        # but ensuring bit-accuracy for the quaternionic manifold.
        torch.save(snapshot, file_path)
        
        print(f"[SUCCESS] Vault crystallized to {file_path}. HDI: {hdi:.4f}, η: {eta:.4f}")
        return file_path

    def run_maintenance_cycle(self):
        """
        Standard maintenance loop: Audit -> Heal (if needed) -> Crystallize.
        """
        audit = self.perform_holomorphic_audit()
        if audit["mean_residual"] > 0.02:
            # Trigger Geodesic Healing if the manifold is drifting
            print("[BerryPhaseCheckpointer] High residual detected. Triggering Geodesic Healing...")
            self.vault.apply_geodesic_healing()
        
        return self.crystallize(tag="maintenance")
