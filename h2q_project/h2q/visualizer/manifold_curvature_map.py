import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple

# Internal H2Q Imports verified via Registry
from h2q.quaternion_ops import quaternion_norm
from h2q.core.discrete_decision_engine import get_canonical_dde

class ManifoldCurvatureMap:
    """
    Visualizes the H2Q Quaternionic Manifold (S³) projected onto a 2D Poincare Disk.
    Identifies 'Topological Tears' where the Discrete Fueter Operator (Df) 
    detects non-holomorphic (hallucinatory) transitions.
    """
    def __init__(self, num_knots: int = 64, device: str = "mps"):
        self.num_knots = num_knots
        self.device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
        
        # Initialize DDE for logic gating visualization
        self.dde = get_canonical_dde()
        
        # Visualization state
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.circle = Circle((0, 0), 1, fill=False, color='white', linestyle='--', alpha=0.5)
        
        # Color mapping for curvature (Spectral Shift η)
        self.cmap = plt.get_cmap("magma")

    def _project_to_poincare(self, q: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects a 4D Quaternion (w, x, y, z) from the unit 3-sphere S³ 
        to the Poincare Disk using stereographic projection.
        Formula: P(q) = (x/(1-w), y/(1-w))
        """
        # Ensure unit norm for projection
        norm = quaternion_norm(q).unsqueeze(-1)
        q_unit = q / (norm + 1e-8)
        
        w = q_unit[..., 0].cpu().numpy()
        x = q_unit[..., 1].cpu().numpy()
        y = q_unit[..., 2].cpu().numpy()
        
        # Avoid singularity at w=1
        denom = 1.0 - w + 1e-6
        px = x / denom
        py = y / denom
        
        # Map to unit disk (Hyperbolic scaling)
        r = np.sqrt(px**2 + py**2)
        r_new = np.tanh(r) # Elastic extension to keep points within the disk
        theta = np.arctan2(py, px)
        
        return r_new * np.cos(theta), r_new * np.sin(theta)

    def render_frame(self, 
                     manifold_state: torch.Tensor, 
                     fueter_residue: torch.Tensor, 
                     spectral_shift: float,
                     token_idx: int):
        """
        Renders a single frame of the logic curvature map.
        
        Args:
            manifold_state: [64, 4] Quaternionic knots.
            fueter_residue: [64] Magnitude of Df (Discrete Fueter Operator).
            spectral_shift: Current η value.
            token_idx: Current position in autoregressive stream.
        """
        self.ax.clear()
        self.ax.add_patch(self.circle)
        self.ax.set_facecolor('#0a0a0a')
        self.fig.patch.set_facecolor('#0a0a0a')

        # Project knots
        px, py = self._project_to_poincare(manifold_state)
        
        # Normalize Fueter residue for 'Tear' visualization (0 to 1)
        tear_intensity = torch.clamp(fueter_residue, 0, 1).cpu().numpy()
        
        # Plot Knots
        # Size represents stability, Color represents logic curvature
        sizes = 50 + (1.0 - tear_intensity) * 100
        colors = self.cmap(tear_intensity)
        
        self.ax.scatter(px, py, s=sizes, c=colors, edgecolors='white', linewidths=0.5, alpha=0.8)

        # Highlight 'Topological Tears' (Hallucinations)
        tear_indices = np.where(tear_intensity > 0.7)[0]
        for idx in tear_indices:
            self.ax.annotate("TEAR", (px[idx], py[idx]), color='cyan', fontsize=8, alpha=0.9)

        # Metadata Overlay
        self.ax.text(-0.95, 0.9, f"Token: {token_idx}", color='white', fontsize=10)
        self.ax.text(-0.95, 0.8, f"Spectral Shift (η): {spectral_shift:.4f}", color='yellow', fontsize=10)
        self.ax.text(-0.95, 0.7, f"Manifold: SU(2) Quaternionic", color='gray', fontsize=8)

        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.axis('off')
        
        plt.pause(0.01)

    def save_snapshot(self, path: str):
        self.fig.savefig(path, facecolor=self.fig.get_facecolor(), edgecolor='none')

if __name__ == "__main__":
    # Experimental Test Loop
    visualizer = ManifoldCurvatureMap(num_knots=64)
    
    # Mock data representing a 1M token stream segment
    for i in range(10):
        mock_manifold = torch.randn(64, 4)
        mock_residue = torch.rand(64) # High values = Hallucination/Tear
        mock_eta = 0.15 * np.sin(i / 5.0) + 0.5
        
        visualizer.render_frame(mock_manifold, mock_residue, mock_eta, i)
