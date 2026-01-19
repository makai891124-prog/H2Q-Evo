import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.interferometer import BerryPhaseInterferometer
from h2q.core.sst import SpectralShiftTracker
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.data.universal_stream import UniversalStreamLoader

class MultiModalBerryInterferometer(nn.Module):
    """
    Measures geometric phase interference (Pancharatnam-Berry phase) between 
    Audio, Vision, and Text manifolds to prove semantic isomorphism.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        # Fix: Using get_canonical_dde to avoid 'dim' keyword argument error
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        # Interferometers for each modality pair
        self.interferometer_av = BerryPhaseInterferometer()
        self.interferometer_vt = BerryPhaseInterferometer()
        self.interferometer_ta = BerryPhaseInterferometer()

        # Projections into the Quaternionic Manifold (S³)
        self.proj_audio = nn.Linear(latent_dim, latent_dim)
        self.proj_vision = nn.Linear(latent_dim, latent_dim)
        self.proj_text = nn.Linear(latent_dim, latent_dim)

    def to_quaternion(self, x):
        """Projects Euclidean embeddings into SU(2) / S³."""
        x = x.view(-1, self.latent_dim // 4, 4)
        return quaternion_normalize(x)

    def calculate_geometric_phase(self, q1, q2, q3):
        """
        Calculates the Berry Phase (gamma) for a closed loop q1 -> q2 -> q3 -> q1.
        In SU(2), this is the argument of the trace of the product of rotations.
        """
        # Loop: (q1* . q2) * (q2* . q3) * (q3* . q1)
        # This measures the holonomy of the connection.
        q1_inv = q1 * torch.tensor([1, -1, -1, -1], device=q1.device)
        q2_inv = q2 * torch.tensor([1, -1, -1, -1], device=q2.device)
        q3_inv = q3 * torch.tensor([1, -1, -1, -1], device=q3.device)

        step1 = quaternion_mul(q1_inv, q2)
        step2 = quaternion_mul(q2_inv, q3)
        step3 = quaternion_mul(q3_inv, q1)

        holonomy = quaternion_mul(quaternion_mul(step1, step2), step3)
        # The scalar part (w) of the resulting quaternion represents cos(theta/2)
        # The phase is the deviation from the identity quaternion (1,0,0,0)
        phase_shift = 1.0 - holonomy[..., 0].mean()
        return phase_shift

    def forward(self, audio_feat, vision_feat, text_feat):
        # 1. Project to Manifold
        q_a = self.to_quaternion(self.proj_audio(audio_feat))
        q_v = self.to_quaternion(self.proj_vision(vision_feat))
        q_t = self.to_quaternion(self.proj_text(text_feat))

        # 2. Measure Interference (Isomorphism Check)
        # If the manifolds are perfectly aligned, the phase shift should be 0.
        gamma_avt = self.calculate_geometric_phase(q_a, q_v, q_t)
        
        # 3. Track Spectral Shift (η)
        # η measures the cognitive deflection caused by modality mismatch
        eta = self.sst.update(gamma_avt)

        return {
            "berry_phase": gamma_avt,
            "spectral_shift": eta,
            "isomorphism_residual": torch.abs(gamma_avt)
        }

class BerryPhaseTrainer:
    def __init__(self, device="mps"):
        self.device = torch.device(device)
        self.model = MultiModalBerryInterferometer().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loader = UniversalStreamLoader() # Registry: h2q.data.universal_stream

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # Extract multimodal atoms
        audio = batch['audio'].to(self.device)
        vision = batch['vision'].to(self.device)
        text = batch['text'].to(self.device)

        # Forward pass through interferometer
        results = self.model(audio, vision, text)
        
        # Loss: Minimize the Berry Phase residual to enforce semantic isomorphism
        # A zero Berry Phase implies the parallel transport of a concept is path-independent.
        loss = results["isomorphism_residual"] + 0.1 * results["spectral_shift"]
        
        loss.backward()
        self.optimizer.step()
        
        return results

def run_interferometer_suite():
    """Entry point for the Multi-modal Berry-Phase experiment."""
    trainer = BerryPhaseTrainer()
    print("[M24-CW] Initializing Berry-Phase Interferometer Suite...")
    
    # Mock data for demonstration of the geometric logic
    mock_batch = {
        'audio': torch.randn(8, 256),
        'vision': torch.randn(8, 256),
        'text': torch.randn(8, 256)
    }
    
    metrics = trainer.train_step(mock_batch)
    print(f"[RESULT] Berry Phase Residual: {metrics['berry_phase'].item():.6f}")
    print(f"[RESULT] Spectral Shift (η): {metrics['spectral_shift']:.6f}")

if __name__ == "__main__":
    run_interferometer_suite()