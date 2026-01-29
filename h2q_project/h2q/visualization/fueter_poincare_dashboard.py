import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from h2q.core.interface_registry import get_canonical_dde, normalize_dde_kwargs
from h2q.quaternion_ops import quaternion_normalize

class PoincareFueterDashboard:
    """
    Real-time dashboard for mapping H2Q geodesic flow onto a Poincaré Disk.
    Maps SU(2) quaternionic states to hyperbolic space to visualize Berry Phase 
    accumulation and logic curvature (Df).
    """
    def __init__(self, dim=256, device="cpu"):
        # Rigid Construction: Fix 'dim' unexpected keyword argument by using canonical registry
        dde_params = normalize_dde_kwargs(dim=latent_dim)
        self.dde = get_canonical_dde(**dde_params)
        
        self.device = device
        self.history_z = []
        self.history_df = []
        self.history_eta = []

    def _project_to_poincare(self, q):
        """
        Stereographic projection from SU(2) (3-sphere) to the Poincaré Disk.
        q = [a, b, c, d] where a^2 + b^2 + c^2 + d^2 = 1
        z = (b + ic) / (1 + a)
        """
        q = quaternion_normalize(q)
        a, b, c, d = q[0], q[1], q[2], q[3]
        
        # Avoid singularity at a = -1
        denom = 1.0 + a.item() + 1e-8
        re = b.item() / denom
        im = c.item() / denom
        
        # Clamp to unit disk boundary
        mag = np.sqrt(re**2 + im**2)
        if mag > 0.99:
            re /= (mag / 0.99)
            im /= (mag / 0.99)
            
        return re, im

    def update(self, state_quaternion, spectral_shift, logic_curvature):
        """
        Update the dashboard with new inference atoms.
        state_quaternion: [4] or [B, 4] tensor
        spectral_shift: float (eta)
        logic_curvature: float (Df)
        """
        re, im = self._project_to_poincare(state_quaternion)
        self.history_z.append((re, im))
        self.history_df.append(logic_curvature)
        self.history_eta.append(spectral_shift)
        
        # Elastic Extension: Keep history windowed to prevent OOM on Mac Mini M4
        if len(self.history_z) > 100:
            self.history_z.pop(0)
            self.history_df.pop(0)
            self.history_eta.pop(0)

    def render_frame_base64(self):
        """
        Generates a real-time visualization frame for the h2q_server interface.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a0a')
        
        # Subplot 1: Poincaré Disk (Berry Phase Path)
        circle = plt.Circle((0, 0), 1, color='cyan', fill=False, linestyle='--', alpha=0.3)
        ax[0].add_artist(circle)
        
        if len(self.history_z) > 1:
            pts = np.array(self.history_z)
            # Color path by logic curvature (Df)
            colors = plt.cm.magma(np.linspace(0, 1, len(pts)))
            ax[0].scatter(pts[:, 0], pts[:, 1], c=colors, s=20, edgecolors='none')
            ax[0].plot(pts[:, 0], pts[:, 1], color='white', alpha=0.2)
            
        ax[0].set_xlim(-1.1, 1.1)
        ax[0].set_ylim(-1.1, 1.1)
        ax[0].set_title(f"Poincaré Geodesic Flow (η={self.history_eta[-1]:.4f})", color='white')
        ax[0].set_aspect('equal')
        ax[0].axis('off')

        # Subplot 2: Logic Curvature (Df) vs Heat-Death Index (HDI)
        ax[1].plot(self.history_df, color='#ff0055', label='Logic Curvature (Df)')
        ax[1].axhline(y=0.05, color='yellow', linestyle='--', label='Hallucination Threshold')
        ax[1].set_facecolor('#111111')
        ax[1].tick_params(colors='white')
        ax[1].set_title("Reasoning Veracity (Fueter Operator)", color='white')
        ax[1].legend()

        # Convert to Base64 for Server Integration
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_telemetry_json(self):
        """
        Returns raw metrics for the h2q_server telemetry stream.
        """
        return {
            "spectral_shift": self.history_eta[-1] if self.history_eta else 0,
            "logic_curvature": self.history_df[-1] if self.history_df else 0,
            "is_analytic": self.history_df[-1] < 0.05 if self.history_df else True,
            "poincare_coord": self.history_z[-1] if self.history_z else (0, 0)
        }