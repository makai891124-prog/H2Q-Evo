import torch
import torch.nn as nn
from h2q.quaternion_ops import quaternion_normalize
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class SpectralManifoldPooling(nn.Module):
    """
    Spectral Manifold Pooling Layer.
    
    Performs 4:1 topological decimation of the sequence length by calculating 
    the Fréchet mean of quaternionic clusters on the SU(2) manifold (S³).
    
    Architecture Alignment:
    - Manifold: 256-dimensional (64 irreducible quaternionic knots).
    - Decimation: 4:1 (Sequence length T -> T/4).
    - Metric: Fréchet mean approximated via Chordal Mean (normalized arithmetic mean).
    - Tracking: Spectral Shift Tracker (η) integration.
    """
    def __init__(self, dde=None):
        super().__init__()
        # Rigid Construction: Use canonical DDE to avoid 'dim' keyword errors reported in feedback
        self.dde = dde or get_canonical_dde()
        self.sst = SpectralShiftTracker()
        self.decimation_factor = 4
        self.manifold_dim = 256
        self.knot_dim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, SeqLen, 256]
        Returns:
            torch.Tensor: Decimated tensor of shape [Batch, SeqLen // 4, 256]
        """
        b, t, c = x.shape
        assert c == self.manifold_dim, f"Input dimension must be {self.manifold_dim}, got {c}"

        # Verify Symmetry: Ensure sequence length is compatible with 4:1 decimation
        if t % self.decimation_factor != 0:
            padding = self.decimation_factor - (t % self.decimation_factor)
            # Pad with the last frame to maintain topological continuity
            x = torch.cat([x, x[:, -1:, :].expand(-1, padding, -1)], dim=1)
            t = x.shape[1]

        num_knots = c // self.knot_dim # 64 knots
        
        # Identify Atoms: Reshape into clusters for decimation
        # Shape: [Batch, NewSeqLen, ClusterSize, NumKnots, QuatComponents]
        x_clusters = x.view(b, t // self.decimation_factor, self.decimation_factor, num_knots, self.knot_dim)

        # Calculate Fréchet Mean (Chordal Approximation on S³)
        # On the unit 3-sphere, the normalized arithmetic mean is the Fréchet mean 
        # under the chordal metric, providing a robust O(1) approximation for SU(2).
        # We pool across the cluster dimension (dim 2)
        arithmetic_mean = torch.mean(x_clusters, dim=2) # [B, T//4, 64, 4]
        pooled_knots = quaternion_normalize(arithmetic_mean)

        # Elastic Extension: Spectral Shift Tracking (η)
        # η = (1/π) arg{det(S)}. We update the tracker with the cluster scattering proxy.
        self._audit_spectral_transition(x_clusters, pooled_knots)

        # Return to manifold shape: [Batch, SeqLen // 4, 256]
        return pooled_knots.reshape(b, t // self.decimation_factor, self.manifold_dim)

    def _audit_spectral_transition(self, clusters: torch.Tensor, pooled: torch.Tensor):
        """
        Updates the Spectral Shift Tracker based on the scattering matrix of the decimation.
        """
        with torch.no_grad():
            # Calculate the deviation from the Fréchet mean as a proxy for manifold curvature
            # diff shape: [B, T//4, 4, 64, 4]
            diff = clusters - pooled.unsqueeze(2)
            
            # Scattering proxy (S): Mean squared norm of deviations
            # This maps to the trace of the scattering matrix in the Krein-like formula
            s_proxy = torch.norm(diff, dim=-1).mean()
            
            # Update η via the global tracker
            if hasattr(self.sst, 'update'):
                self.sst.update(s_proxy)

    def extra_repr(self) -> str:
        return f'decimation_factor={self.decimation_factor}, manifold_dim={self.manifold_dim}'