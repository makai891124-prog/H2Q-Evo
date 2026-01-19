import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# [STABLE] H2Q Foundational Math: SU(2) Quaternion Operations
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Fixed: Added explicit action_dim to __init__ to resolve 
    'unexpected keyword argument num_actions' error.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 256-dimensional topological manifold (H2Q Core)
        self.manifold_projection = nn.Linear(state_dim, 256 * 4) 
        self.policy_head = nn.Linear(256 * 4, action_dim)

    def forward(self, x):
        # Project to Quaternion Space
        q_latent = self.manifold_projection(x).view(-1, 256, 4)
        # Normalize to SU(2) (Unit Quaternions)
        q_latent = F.normalize(q_latent, p=2, dim=-1)
        flat_latent = q_latent.view(x.size(0), -1)
        return self.policy_head(flat_latent), q_latent

class SpectralShiftTracker:
    """
    [EXPERIMENTAL] Implements η = (1/π) arg{det(S)}
    Quantifies cognitive deflection from the geodesic path.
    """
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def compute_eta(self, S: torch.Tensor) -> torch.Tensor:
        # S is the scattering matrix / transition Jacobian
        # For simplicity in this atom, we use the determinant of the latent covariance
        # representing the 'spread' or 'deflection' of the manifold
        if S.dim() < 2: return torch.tensor(0.0)
        # Use a small epsilon for stability on MPS
        det_s = torch.linalg.det(S + torch.eye(S.size(-1), device=S.device) * 1e-6)
        eta = torch.angle(det_s.to(torch.complex64)) / math.pi
        return torch.abs(eta)

class H2QSleepMechanism:
    """
    [EXPERIMENTAL] The Dreaming Phase.
    Reinforces high-η traces via Geodesic Flow (Infinitesimal Rotations).
    """
    def __init__(self, model: DiscreteDecisionEngine, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory_buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

    def record_trace(self, state: torch.Tensor, latent: torch.Tensor, eta: float):
        # Only store high-salience (high deflection) traces
        if eta > 0.1:
            self.memory_buffer.append((state.detach(), latent.detach(), eta))
            if len(self.memory_buffer) > 1000:
                self.memory_buffer.pop(0)

    def dream(self, cycles: int = 5):
        """
        Performs Geodesic Flow updates: Re-aligning the manifold to high-η deflections.
        """
        if not self.memory_buffer:
            return 0.0

        total_loss = 0.0
        for _ in range(cycles):
            # Sample high-η traces
            states, targets, etas = zip(*self.memory_buffer)
            states = torch.cat(states)
            target_latents = torch.cat(targets)
            
            self.optimizer.zero_grad()
            _, current_latents = self.model(states)
            
            # Geodesic Loss: Distance on the SU(2) manifold
            # 1 - <q1, q2>^2 (Squared inner product of quaternions)
            inner_prod = (current_latents * target_latents).sum(dim=-1)
            geodesic_dist = 1.0 - inner_prod**2
            
            # Weight loss by η (Spectral Shift)
            loss = (geodesic_dist * torch.tensor(etas, device=states.device)).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        # Clear buffer after consolidation (forgetting mechanism)
        self.memory_buffer = []
        return total_loss / cycles

def train_iteration(model, sleep_sys, tracker, state_batch):
    # 1. WAKE PHASE
    logits, latents = model(state_batch)
    
    # Calculate η (Spectral Shift)
    # We treat the latent covariance as the S-matrix for the trace formula
    S = torch.matmul(latents.view(latents.size(0), -1).T, latents.view(latents.size(0), -1))
    eta = tracker.compute_eta(S)
    
    # Record for Dreaming
    sleep_sys.record_trace(state_batch, latents, eta.item())
    
    return logits, eta

if __name__ == "__main__":
    # Mac Mini M4 (MPS) Check
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize Atoms
    model = DiscreteDecisionEngine(state_dim=64, action_dim=10).to(device)
    tracker = SpectralShiftTracker()
    sleep_sys = H2QSleepMechanism(model)
    
    # Mock Training Loop
    for epoch in range(10):
        # Wake Phase
        mock_input = torch.randn(32, 64).to(device)
        logits, eta = train_iteration(model, sleep_sys, tracker, mock_input)
        print(f"Epoch {epoch} | Spectral Shift (η): {eta:.4f}")
        
        # Sleep Phase (Consolidation)
        if epoch % 2 == 0:
            dream_loss = sleep_sys.dream()
            print(f"--- Sleep Phase Complete | Geodesic Loss: {dream_loss:.6f} ---")
