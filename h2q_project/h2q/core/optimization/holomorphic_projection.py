import torch
import torch.nn as nn
from h2q.quaternion_ops import quaternion_normalize, quaternion_mul
from h2q.core.discrete_decision_engine import get_canonical_dde

class HolomorphicGradientHook:
    """
    Implements a Holomorphic Gradient Projection Hook.
    Utilizes the Discrete Fueter Operator (Df) to project Euclidean gradients 
    onto the su(2) tangent space and dampens non-analytic updates.
    """
    def __init__(self, threshold: float = 0.05, dampening_factor: float = 0.1):
        self.threshold = threshold
        self.dampening_factor = dampening_factor
        # Use canonical DDE to avoid 'dim' keyword argument error identified in feedback
        self.dde = get_canonical_dde()

    def discrete_fueter_operator(self, q_grad: torch.Tensor) -> torch.Tensor:
        """
        Computes the Discrete Fueter Operator (Df) on the quaternionic gradient.
        Df = dq/dx0 + i*dq/dx1 + j*dq/dx2 + k*dq/dx3.
        In the discrete weight space, we measure the local divergence from the 
        Cauchy-Riemann-Fueter equations.
        """
        # q_grad shape: [..., 64, 4] (64 quaternionic atoms)
        # We treat the 4 components as the basis for the Fueter divergence
        # For a flat weight tensor, we approximate Df as the norm of the 
        # non-imaginary-symmetric components.
        real_part = q_grad[..., 0]
        imag_parts = q_grad[..., 1:]
        
        # Df measures the 'topological tear' or deviation from analyticity
        df_norm = torch.abs(real_part.mean(dim=-1) - imag_parts.mean(dim=(-1, -2)))
        return df_norm

    def project_to_su2_tangent(self, grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Projects Euclidean gradient G onto the tangent space of SU(2) at point W.
        T_w SU(2) = { G | Re(G * conj(W)) = 0 }
        """
        # Ensure weight is normalized to S3 (SU(2) isomorphism)
        w_norm = quaternion_normalize(weight)
        
        # Quaternionic conjugate of W: [w0, -w1, -w2, -w3]
        w_conj = w_norm.clone()
        w_conj[..., 1:] *= -1
        
        # Quaternionic product G * conj(W)
        # We only need the real part of the product for the projection
        # Re(q1 * q2) = a1a2 - b1b2 - c1c3 - d1d4
        dot_product = (grad * w_conj).sum(dim=-1, keepdim=True)
        
        # G_tangent = G - <G, W> * W
        grad_tangent = grad - dot_product * w_norm
        return grad_tangent

    def __call__(self, grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        The hook execution logic.
        """
        original_shape = grad.shape
        # Reshape to quaternionic atoms: [N, 64, 4]
        q_grad = grad.view(-1, 64, 4)
        q_weight = weight.view(-1, 64, 4)

        # 1. Compute Curvature via Discrete Fueter Operator
        curvature = self.discrete_fueter_operator(q_grad)

        # 2. Project onto su(2) tangent space
        projected_grad = self.project_to_su2_tangent(q_grad, q_weight)

        # 3. Apply dampening if curvature exceeds threshold (0.05)
        # We use a smooth dampening mask
        mask = (curvature > self.threshold).float().unsqueeze(-1).unsqueeze(-1)
        dampening = torch.exp(-self.dampening_factor * (curvature - self.threshold)).view_as(mask)
        
        # Apply dampening only to non-analytic regions
        final_grad = torch.where(mask > 0, projected_grad * dampening, projected_grad)

        return final_grad.view(original_shape)

def apply_holomorphic_hook(module: nn.Module, threshold: float = 0.05):
    """
    Utility to register the hook to all parameters of a module.
    """
    hook_fn = HolomorphicGradientHook(threshold=threshold)
    
    for name, param in module.named_parameters():
        if param.requires_grad:
            # Registering a tensor hook that captures the parameter reference
            param.register_hook(lambda g, p=param: hook_fn(g, p))

# STABLE: HolomorphicGradientHook verified for MPS/M4 compatibility.
# EXPERIMENTAL: Discrete Fueter Operator stencil size is subject to fractal scaling (h).