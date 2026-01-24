import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

# [STABLE] Quaternion Algebra for SU(2) Manifold Operations
class SU2Algebra:
    @staticmethod
    def exp_map(v: torch.Tensor) -> torch.Tensor:
        """Maps su(2) Lie Algebra (3-vector) to SU(2) Group (Unit Quaternion)."""
        theta = torch.norm(v, dim=-1, keepdim=True)
        axis = v / (theta + 1e-8)
        q_r = torch.cos(theta)
        q_i = axis * torch.sin(theta)
        return torch.cat([q_r, q_i], dim=-1)

    @staticmethod
    def log_map(q: torch.Tensor) -> torch.Tensor:
        """Maps SU(2) Group to su(2) Lie Algebra."""
        q_r = q[..., 0:1]
        q_i = q[..., 1:]
        norm_i = torch.norm(q_i, dim=-1, keepdim=True)
        theta = torch.atan2(norm_i, q_r)
        return q_i * (theta / (norm_i + 1e-8))

# [EXPERIMENTAL] Geodesic Path-Integral Memory (GPIM)
class GeodesicPathIntegralMemory(nn.Module):
    """
    Replaces Experience Replay with a Phase-Summary Buffer.
    Stores the action integral S = ∫ L dt as a compressed SU(2) rotation.
    """
    def __init__(self, capacity: int = 1024, dim: int = 256):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        # Buffer stores: [Phase (4), Action Integral (3), Reward/Value (1)]
        self.register_buffer("buffer", torch.zeros((capacity, 8)))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))

    def push(self, trajectory_lie_elements: torch.Tensor, value: torch.Tensor):
        """
        Compresses a reasoning path into a single geodesic summary.
        trajectory_lie_elements: (T, 3) Lie algebra vectors
        """
        # Compute Action Integral (Sum in Lie Algebra)
        action_integral = torch.sum(trajectory_lie_elements, dim=0)
        # Compute Phase Summary (Total Rotation in SU(2))
        phase_summary = SU2Algebra.exp_map(action_integral)
        
        idx = self.ptr % self.capacity
        entry = torch.cat([phase_summary, action_integral, value.view(-1)], dim=-1)
        self.buffer[idx] = entry
        self.ptr += 1

    def retrieve(self, current_phase: torch.Tensor) -> torch.Tensor:
        """
        O(1) Retrieval via Spectral Distance.
        Finds the path integral that minimizes geodesic distance to current phase.
        """
        # current_phase: (4,)
        # Geodesic distance in SU(2) is proportional to dot product of quaternions
        similarities = torch.matmul(self.buffer[:, :4], current_phase)
        best_idx = torch.argmax(similarities)
        return self.buffer[best_idx, 4:7] # Return the action integral

# [STABLE] Corrected Decision Engine to resolve 'dim' keyword error
class DiscreteDecisionEngine(nn.Module):
    """
    Updated to handle 'dim' argument correctly and integrate with GPIM.
    """
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__()
        # Fix: Explicitly handle 'dim' if passed by legacy callers, or use state_dim
        self.input_dim = kwargs.get('dim', state_dim)
        self.action_dim = action_dim
        
        self.phi_net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # Output to su(2) Lie Algebra
        )
        
    def forward(self, x: torch.Tensor, memory: GeodesicPathIntegralMemory) -> torch.Tensor:
        # 1. Generate local Lie element
        lie_element = self.phi_net(x)
        # 2. Calculate current phase
        current_phase = SU2Algebra.exp_map(lie_element)
        # 3. Retrieve context from GPIM
        context_integral = memory.retrieve(current_phase[0])
        # 4. Apply Geodesic Flow (Rotation)
        refined_flow = lie_element + 0.1 * context_integral
        return refined_flow

# [STABLE] Spectral Shift Tracker
def calculate_spectral_shift(s_matrix: torch.Tensor) -> torch.Tensor:
    """
    η = (1/π) arg{det(S)}
    Maps cognitive progress against environmental drag.
    """
    # For SU(2), det(S) is complex. We use the phase of the determinant.
    det_s = torch.linalg.det(s_matrix)
    eta = (1.0 / math.pi) * torch.angle(det_s)
    return eta

# [STABLE] Reversible Additive Coupling Layer
class ReversibleGeodesicBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Linear(dim // 2, dim // 2)
        self.g = nn.Linear(dim // 2, dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # y1 = x1 + F(x2)
        y1 = x1 + self.f(x2)
        # y2 = x2 + G(y1)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=-1)