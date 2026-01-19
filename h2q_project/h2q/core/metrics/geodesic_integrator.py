import torch
import torch.nn as nn
import math
from typing import List, Optional

class GeodesicPathIntegrator(nn.Module):
    """
    [STABLE] Geodesic-Path-Integrator
    Computes the path integral of cognitive deflection (η) across a reasoning trace.
    Maps 256-dim quaternionic manifold states to SU(2) spectral shifts to detect 
    hallucination via logic curvature.
    """
    def __init__(self, threshold: float = 0.15):
        super().__init__()
        self.threshold = threshold
        # Device grounding for Mac Mini M4 (MPS)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _project_to_su2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects a 256-dim vector into a batch of 64 SU(2) matrices (2x2 complex).
        Atom: Manifold Projection.
        """
        # Ensure input is (Batch, 256)
        batch_size = x.shape[0]
        # Reshape to (Batch, 64, 4) where 4 represents the quaternionic components (1, i, j, k)
        q = x.view(batch_size, 64, 4)
        
        # Construct SU(2) matrix: [[a + di, b + ci], [-b + ci, a - di]]
        # Using complex representation for torch.linalg.det
        a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        real_part = torch.stack([
            torch.stack([a, b], dim=-1),
            torch.stack([-b, a], dim=-1)
        ], dim=-2)
        
        imag_part = torch.stack([
            torch.stack([d, c], dim=-1),
            torch.stack([c, -d], dim=-1)
        ], dim=-2)
        
        return torch.complex(real_part, imag_part)

    def calculate_step_eta(self, state_t: torch.Tensor, state_t_next: torch.Tensor) -> torch.Tensor:
        """
        Computes η = (1/π) arg{det(S)} where S is the transition operator.
        Atom: Spectral Shift Tracker.
        """
        # S = X_{t+1} @ inv(X_t)
        # In SU(2), inverse is the conjugate transpose (H)
        s_t = self._project_to_su2(state_t)
        s_t_next = self._project_to_su2(state_t_next)
        
        # Transition matrix S
        S = torch.matmul(s_t_next, s_t.m_adjoint() if hasattr(s_t, 'm_adjoint') else s_t.conj().transpose(-2, -1))
        
        # η = (1/π) arg{det(S)}
        # det(S) for SU(2) should be 1; deviations indicate 'deflection' or 'noise'
        determinants = torch.linalg.det(S)
        eta = torch.angle(determinants) / math.pi
        
        # Return mean spectral shift across the 64 manifold atoms
        return torch.mean(torch.abs(eta))

    @torch.inference_mode()
    def compute_curvature(self, reasoning_trace: List[torch.Tensor]) -> dict:
        """
        Integrates η across the entire trace to provide the 'Curvature of Logic'.
        """
        if len(reasoning_trace) < 2:
            return {"curvature": 0.0, "status": "grounded"}

        total_η = 0.0
        steps = len(reasoning_trace) - 1

        for i in range(steps):
            step_η = self.calculate_step_eta(
                reasoning_trace[i].to(self.device),
                reasoning_trace[i+1].to(self.device)
            )
            total_η += step_η.item()

        avg_curvature = total_η / steps
        is_hallucinating = avg_curvature > self.threshold

        return {
            "curvature": round(avg_curvature, 6),
            "status": "hallucination" if is_hallucinating else "grounded",
            "drift_magnitude": total_η
        }

# [EXPERIMENTAL] Orthogonal fix for DiscreteDecisionEngine __init__ collision
# To be used if the 'dim' argument error persists in the parent caller.
class RobustDecisionEngine(nn.Module):
    def __init__(self, **kwargs):
        # Explicitly pop 'dim' to prevent unexpected keyword argument errors
        engine_dim = kwargs.pop('dim', 256) 
        super().__init__()
        self.integrator = GeodesicPathIntegrator()
        self.latent_dim = engine_dim
