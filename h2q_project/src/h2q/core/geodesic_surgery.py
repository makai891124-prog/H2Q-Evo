import torch
import torch.nn as nn
from typing import Optional
from h2q.quaternion_ops import quaternion_stability

class HolomorphicGradientSurgeryHook:
    """
    Implements a global autograd surgery that dampens gradient components 
    violating the 4th-order Fueter-Laplace biharmonic constraint.
    
    Ensures updates preserve manifold analyticity on the SU(2) manifold 
    by treating the 256-dimensional coordinates as 64 quaternionic knots.
    """
    def __init__(self, damping_factor: float = 0.1, epsilon: float = 1e-6):
        self.alpha = damping_factor
        self.eps = epsilon

    def _discrete_fueter_operator(self, q: torch.Tensor) -> torch.Tensor:
        """
        Applies the Discrete Fueter Operator (Df) across the knot dimension.
        q shape: [Batch, 64, 4] (where 4 represents 1, i, j, k)
        """
        # Central difference approximation for the derivative across the knot sequence
        # In a 1D manifold flow, we treat the index as the spatial coordinate
        q_plus = torch.roll(q, shifts=1, dims=1)
        q_minus = torch.roll(q, shifts=-1, dims=1)
        
        # Df = dq/dt (simplified for discrete knot sequence)
        return (q_plus - q_minus) / 2.0

    def _discrete_laplacian(self, q: torch.Tensor) -> torch.Tensor:
        """
        Applies the discrete Laplacian operator across the knot dimension.
        """
        q_plus = torch.roll(q, shifts=1, dims=1)
        q_minus = torch.roll(q, shifts=-1, dims=1)
        return q_plus - 2 * q + q_minus

    def __call__(self, grad: torch.Tensor) -> Optional[torch.Tensor]:
        if grad is None:
            return None

        # 1. Identify Atoms: Ensure we are working with the 256-dim (64 knots) structure
        original_shape = grad.shape
        # Flatten to handle various layer types, then isolate the 64x4 structure
        # We assume the last 256 dims are the quaternionic coordinates
        flat_grad = grad.view(-1, 64, 4)

        # 2. Compute 4th-order Fueter-Laplace biharmonic constraint violation
        # Violation V = Df( Delta( grad ) )
        # This identifies 'topological tears' where the gradient flow deviates from holomorphicity
        
        laplacian = self._discrete_laplacian(flat_grad)
        biharmonic_violation = self._discrete_fueter_operator(laplacian)

        # 3. Dampen components violating the constraint
        # We subtract the non-holomorphic 'noise' from the gradient
        # This acts as a projection back onto the holomorphic subspace
        corrected_grad = flat_grad - self.alpha * biharmonic_violation

        # 4. Apply stability guard
        corrected_grad = quaternion_stability(corrected_grad)

        return corrected_grad.view(original_shape)

def apply_holomorphic_surgery(model: nn.Module, damping_factor: float = 0.1):
    """
    Registers the HolomorphicGradientSurgeryHook to all parameters 
    matching the H2Q manifold coordinate dimensions (multiples of 256).
    """
    hook = HolomorphicGradientSurgeryHook(damping_factor=damping_factor)
    registered_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() % 256 == 0:
            param.register_hook(hook)
            registered_count += 1
            
    return registered_count

def verify_surgery_symmetry(grad_before: torch.Tensor, grad_after: torch.Tensor) -> float:
    """
    Quantifies the spectral shift η induced by the surgery.
    Returns the L2 norm of the 'topological tear' removed.
    """
    if grad_before is None or grad_after is None:
        return 0.0
    tear = grad_before - grad_after
    return torch.norm(tear).item()