import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from h2q.core.optimizers.hjb_solver import HJBGeodesicSolver
from h2q.core.sst import SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde

class HJBValueSurfaceRenderer:
    """
    Visualizes the Hamilton-Jacobi-Bellman (HJB) cost functional surface 
    over a Poincare projection of the SU(2) manifold.
    
    This tool tracks the 'Healing' process during the Manifold Sleep Phase,
    where the system minimizes the spectral drag mu(E).
    """
    def __init__(self, device=None):
        self.device = device if device else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.solver = HJBGeodesicSolver() # Verified in h2q.core.optimizers.hjb_solver
        self.sst = SpectralShiftTracker() # Verified in h2q.core.sst
        # Use canonical DDE to avoid 'dim' keyword argument error identified in feedback
        self.dde = get_canonical_dde()

    def _stereographic_projection(self, quaternions):
        """
        Maps 4D quaternions (S^3) to 3D Poincare coordinates.
        q = [w, x, y, z] -> P = [x/(1-w), y/(1-w), z/(1-w)]
        """
        w = quaternions[..., 0]
        xyz = quaternions[..., 1:]
        # Avoid singularity at w=1
        denom = 1.0 - w + 1e-6
        return xyz / denom.unsqueeze(-1)

    @torch.no_grad()
    def generate_hjb_telemetry(self, manifold_state, resolution=20):
        """
        Computes the HJB value surface around the current manifold state.
        """
        # Create a local grid in the tangent space
        u = torch.linspace(-1, 1, resolution, device=self.device)
        v = torch.linspace(-1, 1, resolution, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        
        # Perturb the manifold state to sample the surface
        # In H2Q, HJB value V(q) is the cost-to-go for geodesic alignment
        hjb_values = torch.zeros((resolution, resolution), device=self.device)
        
        for i in range(resolution):
            for j in range(resolution):
                # Simulate a local geodesic displacement
                perturbed_state = manifold_state + 0.1 * (uu[i,j] + vv[i,j])
                # HJB Value V = Spectral Shift (eta) + Integrated Environment Drag (mu)
                hjb_values[i, j] = self.solver.compute_value(perturbed_state)

        return uu.cpu().numpy(), vv.cpu().numpy(), hjb_values.cpu().numpy()

    def render_healing_phase(self, history_states, save_path=None):
        """
        Renders the HJB surface and the 'Healing' trajectory (geodesic flow).
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Sample the surface at the final state
        current_state = history_states[-1]
        X, Y, Z = self.generate_hjb_telemetry(current_state)

        # Plot HJB Value Surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
        
        # Project and plot the healing trajectory
        trajectory_3d = []
        for state in history_states:
            # Map quaternionic state to Poincare coordinates for visualization
            proj = self._stereographic_projection(state.view(-1, 4)[0])
            val = self.solver.compute_value(state)
            trajectory_3d.append([proj[0].item(), proj[1].item(), val.item()])
        
        traj = np.array(trajectory_3d)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='red', marker='o', markersize=4, label='Healing Path')

        ax.set_title("HJB Value Surface: Manifold Sleep Phase Telemetry")
        ax.set_xlabel("Poincare X")
        ax.set_ylabel("Poincare Y")
        ax.set_zlabel("HJB Cost (V)")
        ax.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Experimental: Real-time telemetry hook
def monitor_hjb_healing(trainer_state):
    """
    Hook for H2QSleepHealer to provide real-time telemetry.
    """
    renderer = HJBValueSurfaceRenderer()
    renderer.render_healing_phase(trainer_state['manifold_history'])
