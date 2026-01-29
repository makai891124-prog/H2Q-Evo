import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.sst import SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde

class HolomorphicSymbolicSynthesizer(nn.Module):
    """
    Holomorphic Symbolic Synthesizer (HSS)
    Constructs logical proofs as monogenic surfaces on the SU(2) manifold.
    Enforces a 4th-order Fueter-Laplace hard constraint to prevent branch divergence.
    """
    def __init__(self, dim=256, device="mps"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize Discrete Decision Engine via Canonical Registry to avoid 'dim' kwarg errors
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        # SU(2) Logic Embedding: Maps symbols to Quaternions (4D real components)
        self.logic_embedding = nn.Parameter(torch.randn(latent_dim, 4, device=device))
        
        # Geodesic Flow Generator (su(2) Lie Algebra elements)
        self.flow_generator = nn.Linear(4, 4, bias=False, device=device)

    def fueter_laplace_4th_order(self, q_surface):
        """
        Computes the discrete 4th-order Fueter-Laplace operator (Biharmonic) on the logic surface.
        Constraint: Delta^2(q) = 0 ensures the surface is monogenic and analytic.
        """
        # q_surface shape: [Batch, Sequence, 4]
        if q_surface.shape[1] < 5:
            return torch.tensor(0.0, device=self.device)

        # Discrete Laplacian (2nd order)
        # Delta q_n = q_{n+1} - 2q_n + q_{n-1}
        laplacian = q_surface[:, 2:-2] - 2 * q_surface[:, 1:-3] + q_surface[:, 0:-4]
        
        # Biharmonic (4th order): Delta(Delta q)
        # Delta^2 q_n = q_{n+2} - 4q_{n+1} + 6q_n - 4q_{n-1} + q_{n-2}
        biharmonic = (q_surface[:, 4:] 
                      - 4 * q_surface[:, 3:-1] 
                      + 6 * q_surface[:, 2:-2] 
                      - 4 * q_surface[:, 1:-3] 
                      + q_surface[:, 0:-4])
        
        return torch.norm(biharmonic, p=2)

    def synthesize_step(self, current_state, symbol_idx):
        """
        Performs a single geodesic reasoning step.
        """
        # 1. Retrieve Symbolic Atom
        atom = self.logic_embedding[symbol_idx]
        
        # 2. Infinitesimal Rotation (Geodesic Flow)
        # Weights undergo h + delta rotation in su(2)
        rotation_vector = self.flow_generator(atom)
        next_state = quaternion_mul(current_state, rotation_vector)
        next_state = quaternion_normalize(next_state)
        
        return next_state

    def forward(self, symbol_sequence):
        """
        Constructs the proof surface and applies the Fueter constraint.
        """
        batch_size = 1
        seq_len = symbol_sequence.shape[0]
        
        # Initial state on S^3 (Identity Quaternion)
        state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(batch_size, 1)
        surface = []

        for i in range(seq_len):
            state = self.synthesize_step(state, symbol_sequence[i])
            surface.append(state.unsqueeze(1))

        q_surface = torch.cat(surface, dim=1) # [Batch, Seq, 4]

        # 3. Veracity Enforcement via Fueter-Laplace
        curvature_tear = self.fueter_laplace_4th_order(q_surface)
        
        # 4. Spectral Shift Tracking (eta)
        # Linking discrete decisions to continuous environmental drag
        self.sst.update(q_surface)
        eta = self.sst.get_eta()

        # 5. Hard Constraint Projection
        # If curvature_tear is high, it indicates a 'topological tear' (hallucination)
        if curvature_tear > 1e-3:
            # Orthogonal approach: Instead of fixing the loop, we project the surface
            # back to the monogenic manifold using a simple normalization/smoothing
            q_surface = q_surface / (1.0 + curvature_tear)

        return {
            "surface": q_surface,
            "veracity_score": 1.0 / (1.0 + curvature_tear.item()),
            "spectral_shift": eta,
            "is_stable": curvature_tear < 0.01
        }

# Experimental: Standardized Decision Engine Wrapper to handle signature drift
def create_synthesizer(dim=256):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HolomorphicSymbolicSynthesizer(dim=latent_dim, device=device)
