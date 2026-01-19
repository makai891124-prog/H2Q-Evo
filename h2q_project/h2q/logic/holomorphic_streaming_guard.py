import torch
import torch.nn as nn
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.quaternion_ops import quaternion_norm

class HolomorphicStreamingGuard(nn.Module):
    """
    HolomorphicStreamingGuard Middleware
    
    Performs token-by-token pruning during autoregressive generation by calculating 
    the 2nd-order Fueter-Laplace curvature (Δf) on the SU(2) manifold.
    
    Threshold: 0.05 (Topological Tear Limit)
    """
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
        # Use canonical DDE factory to avoid 'dim' keyword argument errors found in runtime logs
        self.dde = get_canonical_dde()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def calculate_curvature(self, q_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discrete 2nd-order Fueter-Laplace curvature.
        q_trajectory: (Batch, Seq, 4) - Quaternionic states on S³
        """
        if q_trajectory.size(1) < 3:
            # Not enough history to compute 2nd order curvature; assume stable flow
            return torch.zeros(q_trajectory.size(0), device=q_trajectory.device)

        # Discrete Laplacian: Δq = q_t - 2q_{t-1} + q_{t-2}
        # This measures the deviation from the holomorphic geodesic flow.
        q_t = q_trajectory[:, -1, :]
        q_t_minus_1 = q_trajectory[:, -2, :]
        q_t_minus_2 = q_trajectory[:, -3, :]

        # 2nd-order difference as a proxy for Fueter-Laplace curvature
        laplacian = q_t - 2 * q_t_minus_1 + q_t_minus_2
        
        # Curvature is the norm of the Laplacian in the quaternionic space
        # Using torch.linalg.vector_norm for MPS compatibility and precision
        curvature = torch.linalg.vector_norm(laplacian, dim=-1)
        return curvature

    @torch.no_grad()
    def audit_step(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Audits the current generation step.
        hidden_states: (Batch, Seq, Hidden_Dim)
        Returns: (Batch,) Boolean mask where True = Keep, False = Prune
        """
        # Rigid Construction: Map hidden states to the SU(2) manifold (first 4 components)
        # In H2Q, the manifold state is isomorphic to the quaternionic unit 3-sphere.
        q_trajectory = hidden_states[..., :4]
        
        # Ensure manifold grounding: Normalize to unit quaternions
        q_norm = torch.linalg.vector_norm(q_trajectory, dim=-1, keepdim=True) + 1e-8
        q_trajectory = q_trajectory / q_norm

        curvature = self.calculate_curvature(q_trajectory)

        # Pruning Logic: Terminate branches exceeding the 0.05 threshold
        # This identifies 'topological tears' indicative of logical hallucinations.
        keep_mask = curvature <= self.threshold

        return keep_mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for integration into the h2q_server generation pipeline.
        """
        # Move to MPS if available for M4 performance
        if hidden_states.device.type != self.device.type:
            hidden_states = hidden_states.to(self.device)
            
        return self.audit_step(hidden_states)