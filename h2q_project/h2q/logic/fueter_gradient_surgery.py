import torch
import torch.nn as nn

class BiharmonicGradientSurgeryHook:
    """
    Applies a 4th-order Fueter-Laplace filter to backpropagated gradients.
    Zeroes out components that contribute to non-analytic manifold curvature
    (topological tears) by evaluating the biharmonic residual across the 
    quaternionic knots.
    """
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return None

        # Atom 1: Quaternionic Atomization
        # The H2Q manifold is 256-dim, isomorphic to 64 quaternions (knots).
        original_shape = grad.shape
        # Reshape to [Batch, Knots, Components] -> [*, 64, 4]
        # We treat the last dimension as the quaternionic basis {1, i, j, k}
        try:
            flat_grad = grad.view(-1, 64, 4)
        except RuntimeError:
            # Fallback for non-standard shapes (e.g. bias terms)
            return grad

        # Atom 2: Discrete Laplacian Operator (2nd Order)
        # L(q) = q_{i+1} + q_{i-1} - 2q_i
        def discrete_laplacian(x):
            # Circular roll maintains the symmetry of the SU(2)^64 manifold
            x_prev = torch.roll(x, shifts=1, dims=1)
            x_next = torch.roll(x, shifts=-1, dims=1)
            return x_next + x_prev - 2 * x

        # Atom 3: 4th-Order Fueter-Laplace (Biharmonic) Filter
        # In quaternionic analysis, the biharmonic operator Delta^2 identifies 
        # non-analytic singularities (topological tears).
        # B(q) = Delta(Delta(q))
        laplacian_1 = discrete_laplacian(flat_grad)
        biharmonic_residual = discrete_laplacian(laplacian_1)

        # Atom 4: Curvature-Based Surgery
        # Identify 'topological tears' where non-analytic curvature exceeds threshold.
        # Analytic components satisfy the Fueter equation (Df = 0), implying low residual.
        curvature_norm = torch.norm(biharmonic_residual, dim=-1, keepdim=True)
        
        # Surgery: Zero out components contributing to high-frequency tears.
        # Stable Code: Hard thresholding as per Foundational Directive.
        mask = (curvature_norm < self.threshold).to(grad.dtype)
        
        # Atom 5: Manifold Reconstruction
        cleaned_grad = flat_grad * mask
        
        return cleaned_grad.view(original_shape)

class FueterGradientSurgery:
    """
    Registry-compatible interface for Biharmonic Surgery.
    Ensures symmetry with the H2Q Global Interface Registry.
    """
    def __init__(self, threshold: float = 0.05):
        self.hook_fn = BiharmonicGradientSurgeryHook(threshold=threshold)

    def attach(self, model: nn.Module):
        """Attaches the surgery hook to all parameters requiring gradients."""
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(self.hook_fn)

def apply_fueter_surgery(model: nn.Module, threshold: float = 0.05):
    """
    Functional entry point for applying the surgery hook.
    Usage: model = apply_fueter_surgery(model, threshold=0.05)
    """
    surgery = FueterGradientSurgery(threshold=threshold)
    surgery.attach(model)
    return model

# Experimental: Soft-thresholding variant for Elastic Extension
# Label: EXPERIMENTAL
def soft_fueter_filter(grad, threshold=0.05, alpha=0.1):
    """
    Orthogonal approach: Instead of zeroing, apply a damping factor alpha
    to high-curvature components to prevent gradient vanishing.
    """
    hook = BiharmonicGradientSurgeryHook(threshold=threshold)
    # Implementation would involve blending grad and cleaned_grad
    pass