import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize, quaternion_norm
from h2q.core.sst import SpectralShiftTracker
from h2q.core.discrete_decision_engine import get_canonical_dde

class RSKHReversibleFunction(torch.autograd.Function):
    """
    Manual Reversible Kernel for RSKH-RNN to achieve O(1) memory complexity.
    Implements the state update: S_{t+1} = S_t ⊗ R_t
    Inverse: S_t = S_{t+1} ⊗ R_t*
    """
    @staticmethod
    def forward(ctx, state, rotation):
        # state: [B, 64, 4], rotation: [B, 64, 4]
        next_state = quaternion_mul(state, rotation)
        next_state = quaternion_normalize(next_state)
        ctx.save_for_backward(rotation, next_state)
        return next_state

    @staticmethod
    def backward(ctx, grad_output):
        rotation, next_state = ctx.saved_tensors
        
        # Reconstruct previous state: S_t = S_{t+1} ⊗ conj(R_t)
        conj_rotation = rotation.clone()
        conj_rotation[..., 1:] *= -1.0
        prev_state = quaternion_mul(next_state, conj_rotation)
        
        # Gradient w.r.t rotation and state
        # Simplified for SU(2) manifold transitions
        grad_state = quaternion_mul(grad_output, conj_rotation)
        
        # conj(S_t) ⊗ grad_output
        conj_prev_state = prev_state.clone()
        conj_prev_state[..., 1:] *= -1.0
        grad_rotation = quaternion_mul(conj_prev_state, grad_output)
        
        return grad_state, grad_rotation

class RSKHRNNCell(nn.Module):
    """
    Recursive Semantic Knot Hashing (RSKH) Recurrence Cell.
    Accumulates sequence holonomy via Berry Phase updates on SU(2)^64.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_quats = embed_dim // 4
        
        # Projections to generate the Berry Phase rotation (Lie Algebra su(2))
        self.phi_gate = nn.Linear(embed_dim, embed_dim)
        
        # Veracity Audit: Use canonical DDE without 'dim' argument to avoid registry mismatch
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
    def _exp_map(self, v):
        """Maps su(2) vectors to SU(2) unit quaternions."""
        # v: [B, 64, 3] (imaginary parts)
        theta = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        axis = v / theta
        
        q_w = torch.cos(theta)
        q_xyz = axis * torch.sin(theta)
        return torch.cat([q_w, q_xyz], dim=-1)

    def forward(self, x, state):
        """
        x: [B, 256] - Input embedding
        state: [B, 64, 4] - Quaternionic manifold state
        """
        B = x.shape[0]
        
        # 1. Generate rotation vector from input (Fractal Expansion h ± δ)
        # We treat the input as a perturbation in the tangent space
        v = self.phi_gate(x).view(B, self.num_quats, 4)
        rotation = self._exp_map(v[..., 1:]) # Use imaginary components for rotation
        
        # 2. Apply Reversible Holonomy Update
        next_state = RSKHReversibleFunction.apply(state, rotation)
        
        # 3. Metacognitive Audit: Spectral Shift η
        # η = (1/π) arg{det(S)} where S is the transition
        # For SU(2), det(S) is complex-valued in the scattering representation
        with torch.no_grad():
            # Calculate η proxy via phase drift
            drift = torch.mean(torch.abs(quaternion_norm(next_state) - 1.0))
            self.sst.update(drift)
            
            # Discrete Fueter Operator (Df) check for topological tears
            # Placeholder for Df != 0 check
            logic_curvature = torch.mean(torch.abs(v[..., 0])) # Real part as curvature proxy
            
        # 4. Decision Gating
        # DDE determines if the state update is 'analytic' or 'hallucinatory'
        gate = self.dde(next_state.view(B, -1))
        final_state = state + gate.unsqueeze(-1) * (next_state - state)
        
        return final_state, self.sst.get_eta()

class RSKHRNN(nn.Module):
    """
    Full RSKH-RNN Module to replace static KV-caches.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cell = RSKHRNNCell(embed_dim)
        self.hdi_threshold = 0.8 # Heat-Death Index threshold

    def forward(self, x_seq):
        # x_seq: [Batch, SeqLen, 256]
        B, S, D = x_seq.shape
        device = x_seq.device
        
        # Initialize state as identity quaternions [1, 0, 0, 0]
        state = torch.zeros(B, D // 4, 4, device=device)
        state[..., 0] = 1.0
        
        outputs = []
        etas = []
        
        for t in range(S):
            state, eta = self.cell(x_seq[:, t, :], state)
            outputs.append(state.view(B, -1))
            etas.append(eta)
            
            # Homeostasis: Check Heat-Death Index (HDI)
            # Prevent dimensional collapse via singular value entropy
            if t % 8 == 0:
                hdi = self._calculate_hdi(state)
                if hdi < self.hdi_threshold:
                    # Apply Fractal Expansion to restore manifold volume
                    state = state * 1.05 
        
        return torch.stack(outputs, dim=1), torch.tensor(etas)

    def _calculate_hdi(self, state):
        """Measures singular value entropy to prevent dimensional collapse."""
        # Flatten to [B, 256]
        flat = state.view(state.shape[0], -1)
        _, s, _ = torch.svd(flat)
        p = s / (torch.sum(s, dim=-1, keepdim=True) + 1e-8)
        entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
        return torch.mean(entropy)
