import torch
import torch.nn as nn
from h2q.core.sst import SpectralShiftTracker
from h2q.utils.mps_compat import mps_safe_det
from h2q.core.discrete_decision_engine import get_canonical_dde

class ManifoldCrosstalkAuditor(nn.Module):
    """
    Manifold Crosstalk Auditor
    
    Measures the spectral overlap between disjoint modalities (e.g., Code vs. Genomic FASTA)
    by calculating the Frobenius norm of the difference between modality-specific η-signatures.
    This prevents semantic 'collision' in the shared 256-D quaternionic manifold.
    """
    def __init__(self):
        super().__init__()
        self.sst = SpectralShiftTracker()
        # Use canonical DDE to avoid 'dim' keyword argument error identified in feedback
        self.dde = get_canonical_dde()
        self.num_knots = 64  # 256-D / 4 atoms per knot

    def _to_su2_representation(self, manifold_state):
        """
        Converts a flat 256-D manifold state into 64 SU(2) 2x2 complex matrices.
        manifold_state: [Batch, 256]
        returns: [Batch, 64, 2, 2] (complex64)
        """
        batch_size = manifold_state.shape[0]
        # Reshape to [Batch, 64 knots, 4 atoms]
        knots = manifold_state.view(batch_size, self.num_knots, 4)
        
        w, x, y, z = knots[..., 0], knots[..., 1], knots[..., 2], knots[..., 3]
        
        # Construct SU(2) matrix: [[w + ix, y + iz], [-y + iz, w - ix]]
        real_part = torch.stack([
            torch.stack([w, y], dim=-1),
            torch.stack([-y, w], dim=-1)
        ], dim=-2)
        
        imag_part = torch.stack([
            torch.stack([x, z], dim=-1),
            torch.stack([z, -x], dim=-1)
        ], dim=-2)
        
        return torch.complex(real_part, imag_part)

    def calculate_eta_signature(self, manifold_state):
        """
        Calculates the η-signature vector for a given manifold state.
        η = (1/π) arg{det(S)}
        """
        su2_matrices = self._to_su2_representation(manifold_state)
        
        # Use mps_safe_det for Mac Mini M4 compatibility
        # det_s shape: [Batch, 64]
        det_s = mps_safe_det(su2_matrices)
        
        # η calculation (Spectral Shift)
        eta = (1.0 / torch.pi) * torch.angle(det_s)
        return eta

    def measure_crosstalk(self, modality_a_state, modality_b_state):
        """
        [STABLE] Measures the Frobenius overlap between two modality signatures.
        
        Args:
            modality_a_state: Manifold state for Modality A (e.g. Code)
            modality_b_state: Manifold state for Modality B (e.g. Genomic)
            
        Returns:
            overlap_score: Frobenius norm of the difference in η-signatures.
        """
        eta_a = self.calculate_eta_signature(modality_a_state)
        eta_b = self.calculate_eta_signature(modality_b_state)
        
        # Calculate Frobenius norm of the difference
        # Since eta is [Batch, 64], we treat it as a signature vector
        diff = eta_a - eta_b
        overlap_score = torch.norm(diff, p='fro', dim=-1)
        
        return overlap_score.mean()

    def audit_report(self, code_stream, genomic_stream, threshold=0.05):
        """
        Performs a full audit and returns a collision warning if overlap exceeds threshold.
        """
        overlap = self.measure_crosstalk(code_stream, genomic_stream)
        
        collision_detected = overlap < threshold
        
        report = {
            "spectral_overlap_norm": overlap.item(),
            "collision_risk": collision_detected,
            "status": "CRITICAL" if collision_detected else "STABLE",
            "recommendation": "Increase Geodesic Curvature" if collision_detected else "Maintain Flow"
        }
        
        return report
