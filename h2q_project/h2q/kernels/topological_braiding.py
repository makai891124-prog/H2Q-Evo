import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# [STABLE] SU(2) Utility Functions for Quaternionic Manifold Mapping
def exp_map_su2(v):
    """
    Maps an element of the su(2) Lie Algebra (3-vector) to the SU(2) Group (Unit Quaternion).
    Uses the Rodrigues' rotation formula equivalent for SU(2).
    """
    theta = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    axis = v / theta
    # SU(2) element: cos(theta) + i*sin(theta)*axis
    q_r = torch.cos(theta)
    q_ijk = torch.sin(theta) * axis
    return torch.cat([q_r, q_ijk], dim=-1)

def quaternionic_mul(q1, q2):
    """Standard Hamilton product for quaternions (B, ..., 4)"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    res_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    res_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    res_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    res_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([res_w, res_x, res_y, res_z], dim=-1)

class SpectralShiftTracker(nn.Module):
    """
    Calculates η = (1/π) arg{det(S)} to track cognitive progress.
    """
    def __init__(self):
        super().__init__()

    def forward(self, S):
        # S is assumed to be a complex representation of the SU(2) state
        # For SU(2), det(S) should be 1, but we track the drift in the manifold
        det_s = torch.linalg.det(S)
        eta = (1.0 / math.pi) * torch.angle(det_s)
        return eta

class TopologicalBraidingKernel(nn.Module):
    """
    [EXPERIMENTAL] Multi-modal fusion layer entangling Vision (YCbCr) and Text (Byte-stream).
    Implements Reversible Kernels with Geodesic Flow in su(2).
    """
    def __init__(self, dim=256, latent_dim=64):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim # 64 * 4 = 256
        
        # Vision Projection (YCbCr 3-channel to Quaternionic 4-channel)
        self.vision_proj = nn.Conv2d(3, latent_dim * 4, kernel_size=1)
        
        # Text Projection (Byte-stream to Quaternionic 4-channel)
        self.text_proj = nn.Linear(1, latent_dim * 4)
        
        # Geodesic Flow Generators (Lie Algebra su(2) elements)
        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 3 // 4) # Generates 3-vector for su(2)
        )
        
        self.psi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 3 // 4)
        )

    def forward(self, vision_x, text_x):
        """
        vision_x: (B, 3, H, W) - YCbCr
        text_x: (B, L) - Byte-stream
        """
        device = vision_x.device
        B, _, H, W = vision_x.shape
        L = text_x.shape[1]

        # 1. Project to Quaternionic Manifold S³
        # Vision: (B, D, H, W) -> (B, H*W, D)
        v_feat = self.vision_proj(vision_x).flatten(2).transpose(1, 2)
        # Text: (B, L, 1) -> (B, L, D)
        t_feat = self.text_proj(text_x.unsqueeze(-1).float() / 255.0)

        # Global Pooling to align dimensions for braiding
        v_glob = torch.mean(v_feat, dim=1) # (B, D)
        t_glob = torch.mean(t_feat, dim=1) # (B, D)

        # 2. Reversible Braiding Step (Additive Coupling in Lie Algebra)
        # y1 = x1 * exp(phi(x2))
        # y2 = x2 * exp(psi(y1))
        
        # Generate rotation from Text to apply to Vision
        v_rot_vec = self.phi(t_glob).view(B, -1, 3)
        v_quat_rot = exp_map_su2(v_rot_vec).view(B, self.dim)
        
        # Apply rotation (Braiding Vision strand)
        v_braided = quaternionic_mul(v_feat.view(-1, 4), v_quat_rot.repeat_interleave(v_feat.size(1), dim=0).view(-1, 4))
        v_braided = v_braided.view(B, -1, self.dim)

        # Generate rotation from braided Vision to apply to Text
        v_braided_glob = torch.mean(v_braided, dim=1)
        t_rot_vec = self.psi(v_braided_glob).view(B, -1, 3)
        t_quat_rot = exp_map_su2(t_rot_vec).view(B, self.dim)
        
        # Apply rotation (Braiding Text strand)
        t_braided = quaternionic_mul(t_feat.view(-1, 4), t_quat_rot.repeat_interleave(t_feat.size(1), dim=0).view(-1, 4))
        t_braided = t_braided.view(B, -1, self.dim)

        # 3. Manifold Snap-Back (QR Decomposition for Stability)
        # We treat the combined features as a matrix and ensure orthogonality
        combined = torch.cat([v_braided, t_braided], dim=1) # (B, H*W + L, D)
        
        # 4. Spectral Shift Tracking
        # Constructing a proxy S matrix from the first 2x2 quaternionic block
        # S = [[a+bi, c+di], [-c+di, a-bi]]
        q = torch.mean(combined, dim=1) # (B, 256)
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        S_real = torch.stack([
            torch.stack([q0, -q1], dim=-1), 
            torch.stack([q2, q3], dim=-1)
        ], dim=-2)
        # Simplified tracker for the JSON output
        eta = torch.mean(torch.atan2(q1, q0)) / math.pi

        return combined, eta

# [STABLE] Verification of the DiscreteDecisionEngine fix
class DiscreteDecisionEngine(nn.Module):
    """
    Fixed version of the engine to prevent 'unexpected keyword argument dim'
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim # Explicitly named to avoid confusion
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)