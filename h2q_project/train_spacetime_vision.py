import torch
import torch.nn as nn

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine for manifold selection.
    Fixed: Removed invalid 'dim' argument from __init__.
    """
    def __init__(self, latent_dim: int, num_choices: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_choices = num_choices
        self.projection = nn.Linear(latent_dim, num_choices)

    def forward(self, x):
        return self.projection(x)

def ycbcr_to_rgb_tensor(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Converts a YCbCr tensor to RGB, ensuring compatibility with batch dimensions
    during multi-GPU/MPS parallel evaluation.
    
    [RIGID CONSTRUCTION]
    - Atom: Tensor Shape Consistency. Handles (C, H, W) and (B, C, H, W).
    - Atom: Device Alignment. Constants are generated on the input device.
    """
    # Ensure 4D tensor (B, C, H, W)
    is_3d = False
    if ycbcr.ndim == 3:
        ycbcr = ycbcr.unsqueeze(0)
        is_3d = True

    if ycbcr.shape[1] != 3:
        raise ValueError(f"Expected 3 channels at dim 1, got {ycbcr.shape[1]}")

    # Slicing preserves the batch dimension (B, 1, H, W)
    y  = ycbcr[:, 0:1, :, :]
    cb = ycbcr[:, 1:2, :, :]
    cr = ycbcr[:, 2:3, :, :]

    # H2Q Manifold Constants (BT.601 standard coefficients)
    # Optimized for MPS via vectorized addition/multiplication
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    # Reconstruct along the channel dimension (dim=1)
    rgb = torch.cat([r, g, b], dim=1)
    rgb = torch.clamp(rgb, 0, 1)

    return rgb.squeeze(0) if is_3d else rgb

# [EXPERIMENTAL] Fractal Backpropagation Kernel (FDC)
# This section is stable for MPS but experimental for multi-node clusters.
def apply_spectral_shift(S: torch.Tensor) -> torch.Tensor:
    """
    Calculates η = (1/π) arg{det(S)} for geodesic evolution.
    """
    det_s = torch.linalg.det(S)
    eta = (1.0 / 3.1415926535) * torch.angle(det_s)
    return eta