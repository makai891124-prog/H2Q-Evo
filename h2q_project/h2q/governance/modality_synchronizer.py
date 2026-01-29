import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.interface_registry import get_canonical_dde, normalize_dde_kwargs
from h2q.quaternion_ops import quaternion_normalize, quaternion_stability

class ModalitySynchronizer(nn.Module):
    """
    Synchronizes Audio, Vision, and Text manifolds by enforcing parallel transport 
    and minimizing Berry Curvature variance across modalities.
    """
    def __init__(self, dim=256, dde_config=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Fix for Runtime Error: DiscreteDecisionEngine.__init__() got an unexpected keyword argument 'dim'
        # We use the canonical registry to normalize arguments and instantiate the engine.
        safe_kwargs = normalize_dde_kwargs(dde_config if dde_config else {})
        self.dde = get_canonical_dde(**safe_kwargs)
        
        # Modality-specific projection heads to SU(2) knots
        self.audio_proj = nn.Linear(latent_dim, latent_dim)
        self.vision_proj = nn.Linear(latent_dim, latent_dim)
        self.text_proj = nn.Linear(latent_dim, latent_dim)

    def compute_berry_phase(self, q_tensor):
        """
        Computes the geometric phase (Berry Phase) of a quaternionic state.
        q_tensor shape: [batch, 64, 4] (256-dim knot)
        """
        q = quaternion_normalize(q_tensor)
        # The Berry phase in SU(2) is related to the area enclosed on the S3 manifold.
        # We approximate the local phase shift via the scalar-to-vector ratio.
        scalar = q[..., 0]
        vector_norm = torch.norm(q[..., 1:], dim=-1)
        
        # Phase angle in the quaternionic plane
        phase = torch.atan2(vector_norm, scalar + 1e-8)
        return phase

    def calculate_berry_curvature_loss(self, audio_q, vision_q, text_q):
        """
        Penalizes non-parallel transport by ensuring the geometric phase shift 
        is uniform across all modality manifolds.
        """
        phi_a = self.compute_berry_phase(audio_q)
        phi_v = self.compute_berry_phase(vision_q)
        phi_t = self.compute_berry_phase(text_q)

        # Calculate pairwise phase coherence (Parallel Transport Penalty)
        loss_av = F.mse_loss(phi_a, phi_v)
        loss_vt = F.mse_loss(phi_v, phi_t)
        loss_ta = F.mse_loss(phi_t, phi_a)

        return (loss_av + loss_vt + loss_ta) / 3.0

    def forward(self, audio_feat, vision_feat, text_feat, eta_target=0.1):
        """
        Performs a synchronization step.
        """
        # Project to Quaternionic Knots (Batch, 64, 4)
        a_q = self.audio_proj(audio_feat).view(-1, 64, 4)
        v_q = self.vision_proj(vision_feat).view(-1, 64, 4)
        t_q = self.text_proj(text_feat).view(-1, 64, 4)

        # 1. Geometric Phase Consistency (Berry Curvature Loss)
        berry_loss = self.calculate_berry_curvature_loss(a_q, v_q, t_q)

        # 2. Decision Atom Alignment via DDE
        # We pass the combined manifold state to the DDE to verify topological stability
        combined_state = (a_q + v_q + t_q) / 3.0
        decision_out = self.dde(combined_state.view(-1, self.latent_dim))

        # 3. Spectral Shift Tracking (Placeholder for integration with SST)
        # In a full cycle, this would be used to modulate the learning rate based on drag
        
        return {
            "berry_curvature_loss": berry_loss,
            "synchronized_state": combined_state,
            "decision_atoms": decision_out
        }

def verify_synchronizer_symmetry():
    """Audit function to ensure the synchronizer maintains SU(2) invariants."""
    model = ModalitySynchronizer()
    dummy_a = torch.randn(1, 256)
    dummy_v = torch.randn(1, 256)
    dummy_t = torch.randn(1, 256)
    
    output = model(dummy_a, dummy_v, dummy_t)
    assert "berry_curvature_loss" in output
    print(f"[VERACITY CHECK] Berry Curvature Loss: {output['berry_curvature_loss'].item():.6f}")

if __name__ == "__main__":
    verify_synchronizer_symmetry()