import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from h2q.quaternion_ops import quaternion_norm
from h2q.core.logic_auditing import HolomorphicAuditKernel

class HolomorphicPathVisualizer:
    """
    Maps O(1) RSKH vault retrieval trajectories onto a 2D Poincaré disk.
    Used to identify 'topological tears' (hallucinations) where Fueter residuals Df != 0.
    """
    def __init__(self, manifold_dim: int = 256):
        self.manifold_dim = manifold_dim
        self.auditor = HolomorphicAuditKernel()
        
    def _project_to_poincare(self, knot: torch.Tensor) -> Tuple[float, float]:
        """
        Projects a high-dimensional quaternionic state (SU(2)^64) to the Poincaré Disk.
        Uses the hyperbolic tangent of the geodesic distance as the radius.
        """
        # 1. Calculate Geodesic Distance from manifold origin (Identity rotation)
        # For SU(2), distance is related to the angle of rotation
        norm = quaternion_norm(knot).mean().item()
        
        # 2. Map distance to Poincaré radius r in [0, 1)
        # r = tanh(d/2)
        r = np.tanh(norm / 2.0)
        
        # 3. Extract phase/direction for theta
        # We use the mean phase of the complex components of the quaternions
        # This is a dimensionality reduction from 256D to 1D angle
        theta = torch.atan2(knot[..., 1].mean(), knot[..., 0].mean()).item()
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def visualize_reasoning_trajectory(
        self, 
        path_knots: List[torch.Tensor], 
        save_path: str = "logic_curvature_map.png"
    ):
        """
        Renders the trajectory of reasoning steps.
        Color intensity represents the Fueter residual (Logic Curvature).
        """
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Draw Poincaré Disk Boundary
        boundary = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', alpha=0.3)
        ax.add_artist(boundary)
        
        points = []
        residuals = []
        
        for i in range(len(path_knots)):
            knot = path_knots[i]
            
            # Project to 2D
            x, y = self._project_to_poincare(knot)
            points.append((x, y))
            
            # Calculate Holomorphic Veracity (Fueter Residual)
            # Df != 0 indicates a 'topological tear' or hallucination
            with torch.no_grad():
                # Mocking the audit call based on registry interface
                # In production, this measures the non-analyticity of the transition
                res = self.auditor.validate_reasoning_step(knot) if hasattr(self.auditor, 'validate_reasoning_step') else torch.tensor(0.0)
                residuals.append(res.item())

        # Plot Trajectory
        pts = np.array(points)
        res_arr = np.array(residuals)
        
        # Normalize residuals for coloring (Red = High Curvature/Hallucination)
        norm_res = (res_arr - res_arr.min()) / (res_arr.max() - res_arr.min() + 1e-6)
        
        for i in range(len(pts) - 1):
            color = plt.cm.autumn(norm_res[i])
            plt.plot(pts[i:i+2, 0], pts[i:i+2, 1], color=color, marker='o', markersize=4, alpha=0.7)
            
            # Annotate O(1) Retrieval Jumps
            if i % 5 == 0:
                plt.text(pts[i, 0], pts[i, 1], f"t_{i}", fontsize=8, alpha=0.5)

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.title("H2Q Holomorphic Path: Poincaré Projection of Logic Curvature")
        plt.xlabel("Re(η)")
        plt.ylabel("Im(η)")
        plt.grid(True, which='both', alpha=0.1)
        plt.savefig(save_path)
        plt.close()
        
        return save_path

# Experimental: Integration with RSKH Vault
def debug_vault_retrieval(vault, query_sequence: List[torch.Tensor]):
    """
    Helper to trace RSKH retrieval paths.
    """
    visualizer = HolomorphicPathVisualizer()
    retrieved_path = []
    
    for q in query_sequence:
        # O(1) Retrieval from RSKH Vault
        knot = vault.retrieve(q) 
        retrieved_path.append(knot)
        
    return visualizer.visualize_reasoning_trajectory(retrieved_path)