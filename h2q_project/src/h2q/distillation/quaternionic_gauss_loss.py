import torch
import torch.nn as nn
from h2q.quaternion_ops import quaternion_normalize
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.discrete_decision_engine import LatentConfig

class QuaternionicGaussLinkingLoss(nn.Module):
    """
    Implements the Quaternionic-Gauss-Linking-Loss to penalize topological mismatch
    between genomic invariants (FASTA) and algorithmic knots (StarCoder).
    
    The loss treats the imaginary components of the SU(2) manifold as coordinates in R^3
    and computes the Gauss Linking Integral between the two paths.
    """
    def __init__(self, epsilon=1e-6, sampling_rate=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.sampling_rate = sampling_rate
        # Fix for Runtime Error: DiscreteDecisionEngine.__init__() got an unexpected keyword argument 'dim'
        # Using canonical factory to ensure signature compatibility across the registry.
        self.dde = get_canonical_dde()

    def _extract_r3_path(self, q_tensor):
        """Extracts the imaginary vector part (i, j, k) from quaternions (w, i, j, k)."""
        # q_tensor shape: [Batch, Length, 4]
        return q_tensor[..., 1:]

    def _compute_linking_number(self, path_a, path_b):
        """
        Computes a differentiable approximation of the Gauss Linking Integral.
        Lk(A, B) = (1/4pi) * sum_i sum_j [ (r_ai - r_bj) . (dr_ai x dr_bj) ] / |r_ai - r_bj|^3
        """
        device = path_a.device
        B, L1, _ = path_a.shape
        _, L2, _ = path_b.shape

        # Compute tangents (dr)
        dr_a = path_a[:, 1:, :] - path_a[:, :-1, :]
        dr_b = path_b[:, 1:, :] - path_b[:, :-1, :]

        # Compute segment midpoints (r)
        r_a = (path_a[:, 1:, :] + path_a[:, :-1, :]) / 2.0
        r_b = (path_b[:, 1:, :] + path_b[:, :-1, :]) / 2.0

        # Sub-sampling for M4 efficiency (O(N^2) reduction)
        if self.sampling_rate < 1.0:
            idx_a = torch.randperm(L1-1)[:max(1, int((L1-1) * self.sampling_rate))]
            idx_b = torch.randperm(L2-1)[:max(1, int((L2-1) * self.sampling_rate))]
            r_a, dr_a = r_a[:, idx_a, :], dr_a[:, idx_a, :]
            r_b, dr_b = r_b[:, idx_b, :], dr_b[:, idx_b, :]

        # Expand for pairwise interaction
        # r_diff: [B, L1, L2, 3]
        r_diff = r_a.unsqueeze(2) - r_b.unsqueeze(1)
        dist = torch.norm(r_diff, dim=-1, keepdim=True) + self.epsilon

        # Cross product of tangents: [B, L1, L2, 3]
        # (dr_a x dr_b)
        cross_prod = torch.cross(dr_a.unsqueeze(2).expand(-1, -1, r_b.size(1), -1), 
                                 dr_b.unsqueeze(1).expand(-1, r_a.size(1), -1, -1), dim=-1)

        # Scalar triple product: (r_diff . cross_prod)
        numerator = torch.sum(r_diff * cross_prod, dim=-1, keepdim=True)
        
        # Integral kernel
        kernel = numerator / (dist ** 3)
        
        # Sum over segments and normalize
        linking_num = torch.sum(kernel, dim=(1, 2)) / (4.0 * torch.pi)
        return linking_num

    def forward(self, genomic_quats, code_quats, mu_env=None):
        """
        Args:
            genomic_quats: [B, L, 4] SU(2) representations of FASTA stream.
            code_quats: [B, L, 4] SU(2) representations of StarCoder stream.
            mu_env: Environmental drag for DDE modulation.
        """
        # Ensure normalization on the SU(2) manifold
        q_gen = quaternion_normalize(genomic_quats)
        q_cod = quaternion_normalize(code_quats)

        # Extract paths
        path_gen = self._extract_r3_path(q_gen)
        path_cod = self._extract_r3_path(q_cod)

        # Compute topological entanglement
        linking_val = self._compute_linking_number(path_gen, path_cod)
        
        # The loss is the magnitude of the linking number (penalizing mismatch/entanglement drift)
        # In cross-modal distillation, we often want the 'knots' to be isomorphic (Lk -> 0 or target constant)
        base_loss = torch.mean(torch.abs(linking_val))

        # DDE Modulation: Adjust penalty based on cognitive load/environmental drag
        # This prevents 'topological tears' (Df > 0.05) during high-drag scenarios
        if mu_env is not None:
            decision = self.dde(base_loss, mu_env)
            return base_loss * decision.eta
        
        return base_loss

# Experimental: Verification of Fueter-Symmetry
def verify_linking_symmetry(loss_fn, q1, q2):
    l1 = loss_fn(q1, q2)
    l2 = loss_fn(q2, q1)
    return torch.allclose(l1, l2, atol=1e-5)