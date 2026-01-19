import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLoader(nn.Module):
    """
    H2Q Vision Loader: Implements unified YCbCr-to-RGB manifold mapping.
    
    This module ensures symmetry between the Spacetime3D_Kernel and the HierarchicalDecoder
    by treating color space conversion as a geodesic projection on the SU(2) manifold.
    """
    def __init__(self):
        super().__init__()
        
        # [RIGID CONSTRUCTION: ATOMS]
        # BT.601 conversion coefficients treated as a manifold rotation matrix.
        # Symmetry Seed: Mapping YCbCr (3-atom) to RGB (3-atom) before Fractal Expansion.
        # Matrix shape: (3, 3)
        self.register_buffer('transform_matrix', torch.tensor([
            [1.0,  0.0,      1.402],
            [1.0, -0.344136, -0.714136],
            [1.0,  1.772,     0.0]
        ]))
        
        # Offset for Cb and Cr channels (centering around the symmetry axis)
        self.register_buffer('offset', torch.tensor([0.0, -0.5, -0.5]))
        
        # Symmetry Breaking constant (h ± δ)
        self.delta = 1e-7

    def ycbcr_to_rgb_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the manifold mapping from YCbCr to RGB.
        
        Args:
            x (torch.Tensor): Input tensor in YCbCr format, shape (B, 3, H, W), range [0, 1].
            
        Returns:
            torch.Tensor: RGB tensor, shape (B, 3, H, W), range [0, 1].
        """
        # 1. IDENTIFY_ATOMS: Extract dimensions
        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels (YCbCr), got {c}")

        # 2. VERIFY_SYMMETRY: Reshape for linear projection
        # (B, 3, H, W) -> (B, H, W, 3) -> (N, 3)
        x_permuted = x.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # 3. Apply Symmetry Centering (Offset)
        # Cb and Cr are typically centered at 0.5 in normalized [0, 1] space
        x_centered = x_permuted + self.offset
        
        # 4. Geodesic Flow: Matrix Multiplication
        # Using MPS-optimized matmul for Mac Mini M4 constraints
        rgb_flat = torch.matmul(x_centered, self.transform_matrix.t())
        
        # 5. Symmetry Breaking (h ± δ)
        # Introducing infinitesimal noise to prevent manifold collapse during backprop
        rgb_flat = rgb_flat + self.delta
        
        # 6. Restore Topology
        rgb = rgb_flat.view(b, h, w, 3).permute(0, 3, 1, 2)
        
        return torch.clamp(rgb, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entry point for the vision pipeline.
        Ensures the input is prepared for the 256-dimensional Fractal Expansion.
        """
        # Ensure MPS compatibility
        if x.device.type == 'mps':
            x = x.contiguous()
            
        return self.ycbcr_to_rgb_manifold(x)

# [STABLE CODE]: DiscreteDecisionEngine fix for signature mismatch
class DiscreteDecisionEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Updated to resolve 'unexpected keyword argument dim'.
        The 'dim' argument is replaced by 'input_dim' to align with H2Q naming conventions.
        """
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.projection(x)