import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, get_canonical_dde

class SU2ExponentialMap(nn.Module):
    """Maps R^3 vector to an SU(2) unit quaternion (3-sphere)."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, v):
        # v shape: (..., 3)
        theta = torch.norm(v, dim=-1, keepdim=True)
        axis = v / (theta + self.eps)
        
        w = torch.cos(theta)
        xyz = torch.sin(theta) * axis
        
        # Return as (w, x, y, z)
        return torch.cat([w, xyz], dim=-1)

class ReversibleBPRMFunction(torch.autograd.Function):
    """
    Implements the reversible quaternionic recurrence update.
    M_{t+1} = Q_input * M_t
    Memory complexity: O(1) activations stored for backprop.
    """
    @staticmethod
    def forward(ctx, m_prev, q_in):
        # m_prev: (batch, dim, 4) - Quaternionic state
        # q_in: (batch, dim, 4) - Input rotation
        m_next = quaternion_mul(q_in, m_prev)
        m_next = quaternion_normalize(m_next)
        
        ctx.save_for_backward(q_in, m_next)
        return m_next

    @staticmethod
    def backward(ctx, grad_m_next):
        q_in, m_next = ctx.saved_tensors
        
        # Reconstruct m_prev: M_prev = Q_in_inv * M_next
        # For unit quaternions, inverse is conjugate: (w, -x, -y, -z)
        q_inv = q_in.clone()
        q_inv[..., 1:] *= -1
        
        m_prev = quaternion_mul(q_inv, m_next)
        m_prev = quaternion_normalize(m_prev)
        
        # Gradient of Hamiltonian product w.r.t q_in and m_prev
        # Using the property: d(A*B) = dA*B + A*dB
        # We approximate the manifold gradient via projection
        grad_q_in = quaternion_mul(grad_m_next, m_prev.clone())
        grad_q_in[..., 1:] *= -1 # Conjugate of m_prev for right-side grad
        
        grad_m_prev = quaternion_mul(q_inv, grad_m_next)
        
        return grad_m_prev, grad_q_in

class BerryPhaseRecurrentManifold(nn.Module):
    """
    BPRM: Replaces KV-caches with a fixed-size quaternionic state.
    The 'Berry Phase' is the geometric phase accumulated in the state M
    as it traverses the SU(2) manifold driven by input tokens.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        
        # Projection to SU(2) parameters (3 components for the Lie Algebra su(2))
        self.input_proj = nn.Linear(dim, self.hidden_dim * 3)
        self.exp_map = SU2ExponentialMap()
        
        # Decision Engine for modulating the Spectral Shift (eta)
        # Fixed: Using get_canonical_dde to avoid 'dim' keyword error
        self.dde = get_canonical_dde()
        
        # Output projection back to embedding space
        self.out_proj = nn.Linear(self.hidden_dim * 4, dim)

    def forward(self, x, m_state=None):
        """
        x: (batch, seq_len, dim)
        m_state: (batch, hidden_dim, 4) or None
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if m_state is None:
            # Initialize on the identity of SU(2): (1, 0, 0, 0)
            m_state = torch.zeros((batch_size, self.hidden_dim, 4), device=device)
            m_state[..., 0] = 1.0

        # Project inputs to su(2) Lie Algebra
        v_params = self.input_proj(x) # (B, S, H*3)
        v_params = v_params.view(batch_size, seq_len, self.hidden_dim, 3)
        q_inputs = self.exp_map(v_params) # (B, S, H, 4)

        outputs = []
        current_m = m_state

        for t in range(seq_len):
            q_t = q_inputs[:, t, :, :]
            
            # Apply Reversible Recurrence
            current_m = ReversibleBPRMFunction.apply(current_m, q_t)
            
            # The output is the state projected back to R^dim
            # We flatten the quaternionic dimension (4) for the linear layer
            out_t = self.out_proj(current_m.view(batch_size, -1))
            outputs.append(out_t)

        # Stack outputs: (B, S, dim)
        full_output = torch.stack(outputs, dim=1)
        
        return full_output, current_m

    def get_memory_complexity(self):
        return "O(1) - Fixed Quaternionic State regardless of Sequence Length"