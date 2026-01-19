import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.quaternion_ops import quaternion_norm

class HolomorphicGatingUnit(nn.Module):
    """
    HGU: Dampens logical paths where the quaternionic field exhibits high divergence (topological tears).
    Utilizes the Discrete Fueter Operator to calculate logic curvature.
    """
    def __init__(self, latent_dim=256, threshold=0.05, dampening_strength=20.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.threshold = threshold
        self.dampening_strength = dampening_strength
        # Ensure compatibility with M4 MPS device
        self.register_buffer("curvature_history", torch.zeros(100))
        self.history_ptr = 0

    def compute_fueter_residual(self, q_tensor):
        """
        Calculates the discrete Fueter residual (divergence) across the manifold atoms.
        q_tensor: [batch, 256] interpreted as [batch, 64, 4] quaternions.
        """
        batch_size = q_tensor.shape[0]
        # Reshape to quaternionic atoms: (Batch, Atoms, 4)
        q = q_tensor.view(batch_size, -1, 4)
        
        # Discrete Fueter Operator approximation: 
        # Df = (dq/dx + i*dq/dy + j*dq/dz + k*dq/dw)
        # In the manifold sequence, we measure the 'tear' between adjacent atoms
        dq = q[:, 1:, :] - q[:, :-1, :]
        
        # Logic Curvature (kappa) is the norm of the non-analytic divergence
        # We use the Frobenius norm across the quaternionic components
        kappa = torch.norm(dq, p=2, dim=-1).mean(dim=-1)
        return kappa

    def fueter_gating_hook(self, module, input, output):
        """
        PyTorch forward-hook implementation to dampen high-curvature paths.
        """
        # Handle potential tuple outputs from complex layers
        if isinstance(output, tuple):
            main_output = output[0]
        else:
            main_output = output

        # Calculate logic curvature
        kappa = self.compute_fueter_residual(main_output)
        
        # Calculate dampening factor: 
        # If kappa > threshold, gate < 1.0
        # gate = exp(-strength * max(0, kappa - threshold))
        gate = torch.exp(-self.dampening_strength * torch.clamp(kappa - self.threshold, min=0.0))
        
        # Apply dampening to the manifold state
        # Reshape gate for broadcasting: [batch, 1]
        dampened_output = main_output * gate.unsqueeze(-1)

        # Update telemetry (Internal Audit)
        with torch.no_grad():
            self.curvature_history[self.history_ptr] = kappa.mean()
            self.history_ptr = (self.history_ptr + 1) % 100

        if isinstance(output, tuple):
            return (dampened_output,) + output[1:]
        return dampened_output

    def apply_to_layer(self, layer):
        """
        Registers the Holomorphic Gating as a forward hook on a target layer.
        """
        return layer.register_forward_hook(self.fueter_gating_hook)

    def forward(self, x):
        """
        Standard forward pass if used as a standalone layer rather than a hook.
        """
        kappa = self.compute_fueter_residual(x)
        gate = torch.exp(-self.dampening_strength * torch.clamp(kappa - self.threshold, min=0.0))
        return x * gate.unsqueeze(-1)

def attach_holomorphic_guard(model, layer_name, threshold=0.05):
    """
    Utility to find a layer by name and attach the HGU hook.
    """
    hgu = HolomorphicGatingUnit(threshold=threshold)
    for name, module in model.named_modules():
        if name == layer_name:
            hgu.apply_to_layer(module)
            return hgu
    return None