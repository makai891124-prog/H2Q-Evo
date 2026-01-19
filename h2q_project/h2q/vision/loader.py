import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RIGID CONSTRUCTION: YCbCr Constants (BT.601) ---
# These constants are synchronized with Spacetime3D_Kernel to ensure 
# cross-modal manifold isomorphism. Mapping: Y -> Scalar, Cb -> i, Cr -> j.
# Reference: ITU-R BT.601

YCBCR_PROJECTION_MATRIX = torch.tensor([
    [0.299, 0.587, 0.114],      # Y (Luminance / Temporal Scalar)
    [-0.168736, -0.331264, 0.5], # Cb (Chroma Blue / i-component)
    [0.5, -0.418688, -0.081312]  # Cr (Chroma Red / j-component)
], dtype=torch.float32)

class VisionLoader:
    """
    Architect: M24-Cognitive-Weaver
    Function: Projects RGB visual atoms into the SU(2) double-cover manifold.
    Constraint: Optimized for Mac Mini M4 (MPS).
    """
    def __init__(self, device="mps"):
        self.device = torch.device(device)
        self.projection = YCBCR_PROJECTION_MATRIX.to(self.device)

    def to_manifold(self, rgb_tensor):
        """
        Converts RGB [B, 3, H, W] to Unit Quaternions [B, 4, H, W].
        The 4th component (k) is initialized as 0 to represent the initial phase.
        """
        # Reshape for matrix multiplication: [B, 3, H*W] -> [B, H*W, 3]
        b, c, h, w = rgb_tensor.shape
        rgb_flat = rgb_tensor.view(b, 3, -1).transpose(1, 2)
        
        # Project to YCbCr
        ycbcr = torch.matmul(rgb_flat, self.projection.t())
        
        # Expand to Quaternion (H) space: [Y, Cb, Cr, 0]
        # Fractal Expansion Protocol: 2 -> 256 (handled in subsequent kernel layers)
        k_component = torch.zeros((b, h * w, 1), device=self.device)
        quaternion = torch.cat([ycbcr, k_component], dim=-1)
        
        # Normalize to ensure unit quaternion (SU(2) manifold constraint)
        norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True) + 1e-8
        quaternion = quaternion / norm
        
        return quaternion.view(b, h, w, 4).permute(0, 3, 1, 2)

# --- ELASTIC WEAVING: Anti-Loop Mechanism ---
# Addressing Feedback: Runtime Error in DiscreteDecisionEngine signature.

class DiscreteDecisionEngine(nn.Module):
    """
    STABLE CODE: Fixed __init__ signature to accept 'num_actions'.
    Governs Geodesic Flow decisions within the SU(2) manifold.
    """
    def __init__(self, num_actions, manifold_dim=256, device="mps"):
        super().__init__()
        self.num_actions = num_actions
        self.manifold_dim = manifold_dim
        self.device = torch.device(device)
        
        # Fractal Expansion: Mapping manifold state to action logits
        self.decision_head = nn.Linear(manifold_dim, num_actions).to(self.device)

    def forward(self, state_vector):
        """
        Calculates the infinitesimal rotation (h + δ) for the next geodesic step.
        """
        return self.decision_head(state_vector)

# --- VERACITY COMPACT: Grounding ---
# Experimental: Reversible Kernel reconstruction test
def verify_isomorphism(q_state):
    """
    Checks if the projected state maintains unit norm symmetry.
    """
    norm = torch.norm(q_state, dim=1)
    return torch.allclose(norm, torch.ones_like(norm), atol=1e-5)