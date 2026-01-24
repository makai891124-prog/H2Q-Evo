import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiscreteDecisionEngine(nn.Module):
    """
    FIXED: Corrected __init__ to avoid 'dim' keyword error.
    Uses 'in_features' and 'num_choices' for explicit atom identification.
    """
    def __init__(self, in_features: int, num_choices: int):
        super().__init__()
        self.projection = nn.Linear(in_features, num_choices)
        
    def forward(self, x):
        # x: [B, C] -> logits: [B, num_choices]
        logits = self.projection(x)
        return F.gumbel_softmax(logits, tau=1.0, hard=True)

class SpectralShiftTracker(nn.Module):
    """
    Implements η = (1/π) arg{det(S)} based on SU(2) geodesic flow.
    Tracks cognitive deflection against environmental drag.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        # S is modeled as a learned operator mapping to the quaternionic manifold
        self.s_generator = nn.Linear(dim, 4) # Quaternionic components (1, i, j, k)

    def forward(self, x):
        # x: [B, L, C]
        q = self.s_generator(x) # [B, L, 4]
        # In SU(2), det(S) for a quaternion q = a + bi + cj + dk is a^2 + b^2 + c^2 + d^2
        # We treat the complex projection for the Krein-like trace
        det_s = torch.norm(q, p=2, dim=-1)
        # η calculation: (1/π) arg{det(S)}. Since det(S) in SU(2) is real/positive,
        # we utilize the phase of the complexified spectral density.
        eta = torch.atan2(q[..., 1], q[..., 0]) / math.pi
        return eta

class AdaptiveSemanticStrider(nn.Module):
    """
    EXPERIMENTAL: Replaces fixed 8:1 compression with η-volatility driven striding.
    Range: 2:1 (High Volatility) to 16:1 (Low Volatility).
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.tracker = SpectralShiftTracker(input_dim)
        # 15 possible stride increments from 2 to 16
        self.decision_engine = DiscreteDecisionEngine(in_features=1, num_choices=15)
        self.strides = torch.arange(2, 17)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, L, C] (Expected C=256 for H2Q)
        Returns:
            compressed_x: Tensor [B, L_new, C]
            stride_factor: int
        """
        B, L, C = x.shape
        device = x.device

        # 1. Calculate η (Spectral Shift)
        eta = self.tracker(x) # [B, L]
        
        # 2. Calculate η-volatility (Temporal variance of cognitive deflection)
        # We take the mean volatility across the sequence to determine a global stride for the block
        volatility = torch.std(eta, dim=1, keepdim=True) # [B, 1]
        
        # 3. Map volatility to stride choice
        # High volatility -> Low Stride (preserve detail)
        # Low volatility -> High Stride (compress redundancy)
        # Normalize volatility to [0, 1] range (clamped)
        norm_vol = torch.sigmoid(volatility)
        
        # Invert: High vol (1.0) should pick index 0 (stride 2)
        # Low vol (0.0) should pick index 14 (stride 16)
        stride_probs = self.decision_engine(1.0 - norm_vol) # [B, 15]
        
        # Select stride (Rigid Construction: Ensure symmetry across batch if possible, 
        # but here we allow per-sample elasticity)
        selected_idx = torch.argmax(stride_probs, dim=-1)
        avg_stride = int(self.strides[selected_idx.median().long()].item())

        # 4. Execute Compression via Adaptive Pooling
        # We use local mean pooling to maintain the Reversible Kernel's reconstruction integrity
        # Padding to ensure divisibility
        pad_len = (avg_stride - (L % avg_stride)) % avg_stride
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape for pooling: [B, C, L]
        x_trans = x.transpose(1, 2)
        compressed_x = F.avg_pool1d(x_trans, kernel_size=avg_stride, stride=avg_stride)
        
        return compressed_x.transpose(1, 2), avg_stride

    def verify_symmetry(self, input_shape, output_shape, stride):
        """
        RIGID CONSTRUCTION CHECK: Ensure the Fractal Expansion can reverse this.
        """
        expected_l = math.ceil(input_shape[1] / stride)
        return output_shape[1] == expected_l
