import torch
import torch.nn as nn
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.optimizers.hjb_solver import HJBGeodesicSolver
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class HolomorphicHealingBackprop(nn.Module):
    """
    HolomorphicHealingBackprop Kernel
    
    Utilizes HJB Geodesic Solver to synthesize 'healing' gradients during the Sleep Phase.
    Applies conjugate SU(2) rotations to minimize the Fueter-analyticity residual (Df),
    effectively repairing 'topological tears' (hallucinations) in reasoning traces.
    """
    def __init__(self, hjb_solver: HJBGeodesicSolver = None, threshold: float = 0.05):
        super().__init__()
        self.hjb_solver = hjb_solver if hjb_solver else HJBGeodesicSolver()
        self.dde = get_canonical_dde() # Avoids 'dim' keyword argument error
        self.sst = SpectralShiftTracker()
        self.threshold = threshold
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def compute_fueter_residual(self, q_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Fueter Operator D = ∂w + i∂x + j∂y + k∂z.
        Any non-zero result indicates a deviation from the holomorphic manifold.
        q_tensor shape: [B, N, 4] (w, x, y, z)
        """
        q_tensor.requires_grad_(True)
        # We treat the components as a quaternionic function f(q) = q
        # In a reasoning trace, we evaluate the divergence of the flow
        w, x, y, z = q_tensor.unbind(-1)
        
        # Compute partials (simplified discrete approximation for reasoning traces)
        # In a real FDC context, these are infinitesimal rotations
        dw = torch.autograd.grad(w.sum(), q_tensor, create_graph=True)[0][..., 0]
        dx = torch.autograd.grad(x.sum(), q_tensor, create_graph=True)[0][..., 1]
        dy = torch.autograd.grad(y.sum(), q_tensor, create_graph=True)[0][..., 2]
        dz = torch.autograd.grad(z.sum(), q_tensor, create_graph=True)[0][..., 3]

        # Fueter residual Df
        df = dw + dx + dy + dz
        return df.abs()

    def synthesize_healing_rotation(self, q_tensor: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """
        Uses HJB Solver to find the geodesic path back to Df = 0.
        Returns the conjugate SU(2) rotation required to 'heal' the state.
        """
        # The HJB solver provides the optimal control (rotation) to minimize the cost (Df)
        # Cost functional: J = ∫ (Df^2 + |u|^2) dt
        target_state = q_tensor.detach().clone()
        
        # If Df > threshold, we calculate the corrective geodesic
        mask = (df > self.threshold).float().unsqueeze(-1)
        
        # The 'healing' gradient is the conjugate of the error rotation
        # We approximate the error rotation as the deviation from the identity quaternion [1, 0, 0, 0]
        # and scale it by the HJB-derived optimal path
        correction_geodesic = self.hjb_solver.solve(q_tensor, df)
        
        # Conjugate rotation in SU(2)
        # q* = [w, -x, -y, -z]
        healing_q = correction_geodesic.clone()
        healing_q[..., 1:] *= -1 
        
        return quaternion_normalize(healing_q) * mask + (1 - mask) * torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_tensor.device)

    def heal_trace(self, reasoning_trace: torch.Tensor) -> torch.Tensor:
        """
        Performs the Sleep Phase healing cycle on a batch of reasoning traces.
        """
        df = self.compute_fueter_residual(reasoning_trace)
        
        # Log the topological tear if detected
        max_tear = df.max().item()
        if max_tear > self.threshold:
            self.sst.update_shift(max_tear) # Track the 'healing' progress
            
        healing_rotation = self.synthesize_healing_rotation(reasoning_trace, df)
        
        # Apply the healing rotation via Hamilton Product
        healed_trace = quaternion_mul(reasoning_trace, healing_rotation)
        
        return quaternion_normalize(healed_trace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for integration into Sleep Phase trainers."""
        return self.heal_trace(x)

# Verification of Symmetry: If we rotate to heal, the new Df must be < threshold.
def audit_healing_integrity(kernel: HolomorphicHealingBackprop, trace: torch.Tensor):
    initial_df = kernel.compute_fueter_residual(trace).mean()
    healed_trace = kernel.heal_trace(trace)
    final_df = kernel.compute_fueter_residual(healed_trace).mean()
    
    return {
        "initial_df": initial_df.item(),
        "final_df": final_df.item(),
        "is_healed": final_df < initial_df
    }