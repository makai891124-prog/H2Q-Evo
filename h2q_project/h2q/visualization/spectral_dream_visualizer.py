import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from h2q.core.sst import SpectralShiftTracker
from h2q.core.discrete_decision_engine import get_canonical_dde

class SpectralDreamRenderer:
    """
    Visualizes the H2Q Sleep-phase healing cycle by projecting the 256-dimensional 
    quaternionic manifold onto a series of Poincaré disks, mapping Berry Phase accumulation.
    """
    def __init__(self, manifold_dim: int = 256, device: str = "mps"):
        self.dim = manifold_dim
        self.device = device if torch.backends.mps.is_available() else "cpu"
        # Initialize DDE without 'dim' argument to avoid registry mismatch
        self.dde = get_canonical_dde()
        
    def project_to_poincare(self, q: torch.Tensor) -> torch.Tensor:
        """
        Projects a quaternionic state (w, x, y, z) from S^3 to the Poincaré Disk D^2.
        Formula: (u, v) = (x/(1+w), y/(1+w))
        """
        # Ensure q is normalized to S^3
        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        w, x, y, z = q_norm.unbind(-1)
        
        # Stereographic projection from the hypersphere to hyperbolic space
        denom = 1.0 + w
        u = x / denom
        v = y / denom
        return torch.stack([u, v], dim=-1)

    def calculate_berry_phase(self, path: torch.Tensor) -> torch.Tensor:
        """
        Calculates the accumulated Berry Phase (geometric phase) along a geodesic path.
        gamma = sum(arg(<psi_n | psi_{n+1}>))
        """
        # Treat quaternions as complex pairs for phase calculation
        # q = (w + ix) + j(y + iz)
        q_complex = torch.view_as_complex(path.reshape(-1, self.dim // 4, 2, 2))
        
        # Inner product between successive states
        inner_prods = (q_complex[:-1].conj() * q_complex[1:]).sum(dim=-1)
        phases = torch.angle(inner_prods)
        return torch.cumsum(phases, dim=0)

    def render_sleep_cycle(self, 
                           manifold_states: torch.Tensor, 
                           sst: SpectralShiftTracker,
                           save_path: Optional[str] = None):
        """
        Generates the Poincaré disk visualization of the healing process.
        """
        steps, batch, dims = manifold_states.shape
        # Reshape to quaternions (steps, batch, 64, 4)
        q_states = manifold_states.view(steps, batch, -1, 4)
        
        # Project to 2D
        projections = self.project_to_poincare(q_states) # (steps, batch, 64, 2)
        berry_phases = self.calculate_berry_phase(q_states[:, 0]) # Use first batch item
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'aspect': 'equal'})
        circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
        
        # Map Spectral Shift (eta) to color intensity
        eta = sst.eta.item() if hasattr(sst, 'eta') else 0.0
        cmap = plt.get_cmap('magma')
        
        for i in range(64): # Visualize the SU(2)^64 components
            path = projections[:, 0, i, :].cpu().numpy()
            color = cmap(berry_phases[-1, i % berry_phases.shape[1]].item() % 1.0)
            ax.plot(path[:, 0], path[:, 1], alpha=0.3, color=color, linewidth=0.5)
            ax.scatter(path[-1, 0], path[-1, 1], s=10, color=color)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"H2Q Spectral Dream: Berry Phase Accumulation (η={eta:.4f})")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()

def synthesize_sleep_gradients(manifold_delta: torch.Tensor, sst: SpectralShiftTracker):
    """
    Experimental: Converts manifold healing gradients into a spectral signature 
    for the visualizer to process.
    """
    # Grounding in Veracity Compact: Ensure delta is compatible with 256-dim manifold
    if manifold_delta.shape[-1] != 256:
        raise ValueError(f"Symmetry Break: Expected 256 dims, got {manifold_delta.shape[-1]}")
    
    # Calculate spectral entropy of the gradient
    grad_norm = torch.norm(manifold_delta)
    spectral_weight = torch.sigmoid(sst.eta * grad_norm)
    return spectral_weight