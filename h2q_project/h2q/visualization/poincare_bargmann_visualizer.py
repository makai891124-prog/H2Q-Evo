import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from h2q.core.interface_registry import get_canonical_dde

class PoincareBargmannVisualizer:
    """
    Poincare-Bargmann-Visualizer: Maps 3-point Bargmann invariants onto a hyperbolic disk
    to visualize Berry Phase accumulation and manifold stabilization during HJB Healing.
    """
    def __init__(self, latent_dim: int = 256, device: str = "mps"):
        self.device = device
        self.latent_dim = latent_dim
        
        # Initialize DDE using canonical registry to avoid 'dim' keyword errors
        self.dde = get_canonical_dde(latent_dim=latent_dim)
        self.dde.to(device)
        
        # Visualization state
        self.fig, self.ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        self.ax.set_ylim(0, 1)
        self.ax.set_title("H2Q Poincare-Bargmann Hyperbolic Disk (Sleep Phase)")
        
    def compute_bargmann_invariant(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """
        Calculates the 3-point Bargmann invariant: B = <z1|z2><z2|z3><z3|z1>
        z are expected to be complex spinors (batch, 2).
        """
        inner12 = torch.sum(z1.conj() * z2, dim=-1)
        inner23 = torch.sum(z2.conj() * z3, dim=-1)
        inner31 = torch.sum(z3.conj() * z1, dim=-1)
        return inner12 * inner23 * inner31

    def project_to_poincare_disk(self, invariant: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps the complex Bargmann invariant to (theta, r) on the Poincare Disk.
        Phase (Berry Phase) -> Theta
        Magnitude (Coherence) -> Radius (1 - exp(-|B|))
        """
        phase = torch.angle(invariant).cpu().numpy()
        magnitude = torch.abs(invariant).cpu().numpy()
        
        # Normalize radius to [0, 1). High magnitude (stable) -> Center; Low magnitude (noise) -> Edge
        # In H2Q, we invert this for the Poincare model: Geodesic flow is the center.
        r = 1.0 - np.exp(-magnitude)
        return phase, r

    def update_dashboard(self, states: torch.Tensor, healing_factor: float):
        """
        Updates the real-time diagnostic plot.
        states: (Batch, 3, 2) complex spinors representing a topological triangle.
        """
        self.ax.clear()
        self.ax.set_ylim(0, 1)
        self.ax.set_facecolor('#0a0a0a')
        
        z1, z2, z3 = states[:, 0], states[:, 1], states[:, 2]
        B = self.compute_bargmann_invariant(z1, z2, z3)
        theta, r = self.project_to_poincare_disk(B)
        
        # Scatter plot with alpha modulated by healing factor (HJB progress)
        colors = plt.cm.viridis(np.linspace(0, 1, len(theta)))
        self.ax.scatter(theta, r, c=colors, alpha=min(1.0, healing_factor), s=20, edgecolors='none')
        
        # Draw the unit boundary
        boundary_theta = np.linspace(0, 2*np.pi, 100)
        self.ax.plot(boundary_theta, np.ones_like(boundary_theta), color='cyan', linestyle='--', alpha=0.3)
        
        plt.pause(0.01)

    def visualize_hjb_healing(self, manifold_trajectory: List[torch.Tensor], curvature_scores: List[float]):
        """
        Simulates the visualization of a sleep cycle where curvature is reduced.
        """
        print("[H2Q] Initiating Poincare-Bargmann Diagnostic for HJB Healing...")
        for i, (states, curvature) in enumerate(zip(manifold_trajectory, curvature_scores)):
            # Healing factor increases as curvature decreases
            healing_factor = 1.0 / (1.0 + curvature)
            self.update_dashboard(states, healing_factor)
            if i % 10 == 0:
                print(f"Step {i}: Logic Curvature = {curvature:.4f} | Berry Phase Stability = {healing_factor:.4f}")

def demo_visualizer():
    """Experimental entry point for Mac Mini M4 validation."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    viz = PoincareBargmannVisualizer(device=device)
    
    # Generate synthetic SU(2) spinors for 3-point invariants
    batch_size = 100
    steps = 50
    
    trajectory = []
    curvatures = []
    
    for t in range(steps):
        # Simulate spinors converging towards a geodesic (healing)
        noise = torch.randn(batch_size, 3, 2, dtype=torch.complex64, device=device) * (1.0 / (t + 1))
        base = torch.tensor([1.0, 0.0], dtype=torch.complex64, device=device).view(1, 1, 2).expand(batch_size, 3, 2)
        states = base + noise
        states = states / torch.norm(states, dim=-1, keepdim=True)
        
        trajectory.append(states)
        curvatures.append(0.5 / (t + 1)) # Curvature decaying
        
    viz.visualize_hjb_healing(trajectory, curvatures)

if __name__ == "__main__":
    demo_visualizer()