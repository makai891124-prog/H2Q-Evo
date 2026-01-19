import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Fixed initialization to resolve 'dim' keyword error.
    Maps latent quaternionic energy to discrete stride selections.
    """
    def __init__(self, in_features: int, num_actions: int):
        super().__init__()
        # Fixed: Changed 'dim' to 'in_features' to match standard linear layer expectations
        # and resolved the unexpected keyword argument reported in feedback.
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.classifier(x)

class SpectralShiftTracker(nn.Module):
    """
    [EXPERIMENTAL] Calculates eta = (1/pi) arg{det(S)} to map environmental drag.
    Used to derive the Heat-Death Index (HDI).
    """
    def __init__(self):
        super().__init__()

    def forward(self, q_state: torch.Tensor):
        # q_state shape: [B, C, 4] (Quaternions)
        # Approximate S-matrix via covariance of quaternionic components
        # In a full implementation, S would be the scattering matrix from the manifold
        batch_size = q_state.shape[0]
        flat_q = q_state.view(batch_size, -1)
        
        # Compute pseudo-determinant of the state phase
        # eta = (1/pi) * phase(det(S))
        # For the router, we use the variance of the Hamilton norm as a proxy for drag mu(E)
        norm = torch.norm(q_state, dim=-1)
        eta = torch.var(norm, dim=-1) / math.pi
        return eta

class TopologicalEntropyRouter(nn.Module):
    """
    [STABLE] Dynamic Stride Auto-Tuner.
    Adjusts compression (2:1 to 16:1) based on Heat-Death Index (HDI).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.tracker = SpectralShiftTracker()
        # Actions: 0->2:1, 1->4:1, 2->8:1, 3->16:1
        self.decision_engine = DiscreteDecisionEngine(in_features=1, num_actions=4)
        self.strides = [2, 4, 8, 16]

    def hamilton_product(self, q1, q2):
        """SU(2) Geodesic Flow primitive."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    def forward(self, x, prev_state=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            prev_state: Previous quaternionic state [B, C, 4]
        Returns:
            downsampled_x, stride_selected, hdi
        """
        device = x.device
        B, C, H, W = x.shape

        # 1. Project to Quaternionic Substrate (Simplified for routing decision)
        # We treat the mean spatial energy as the real part of the quaternion
        q_current = torch.zeros((B, 1, 4), device=device)
        q_current[:, :, 0] = torch.mean(x, dim=(1, 2, 3))

        # 2. Calculate Heat-Death Index (HDI)
        # HDI spikes when spectral shift eta is low (system stagnation)
        eta = self.tracker(q_current)
        hdi = 1.0 - torch.tanh(eta)

        # 3. Decide Stride
        stride_logits = self.decision_engine(hdi.unsqueeze(-1))
        stride_idx = torch.argmax(stride_logits, dim=-1)
        
        # For M4 optimization, we use the mode of the batch to maintain kernel symmetry
        # or apply adaptive pooling if batch-heterogeneous strides are required.
        chosen_stride = self.strides[stride_idx[0].item()]

        # 4. Execute Compression (Fractal Expansion Protocol inverse)
        # Using adaptive pooling to simulate the dynamic stride shift
        new_h, new_w = H // chosen_stride, W // chosen_stride
        if new_h == 0 or new_w == 0:
             new_h, new_w = 1, 1
             
        compressed_x = F.adaptive_avg_pool2d(x, (new_h, new_w))

        return compressed_x, chosen_stride, hdi

# Verification for M4 (MPS) compatibility
if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    router = TopologicalEntropyRouter(channels=64).to(device)
    dummy_input = torch.randn(8, 64, 128, 128).to(device)
    out, stride, hdi = router(dummy_input)
    print(f"Selected Stride: {stride} | HDI Mean: {hdi.mean().item():.4f} | Output Shape: {out.shape}")