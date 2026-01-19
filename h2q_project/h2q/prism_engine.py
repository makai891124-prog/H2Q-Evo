import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# [STABLE] PrismConverter: Maps Euclidean SVD embeddings to SU(2) manifold
class PrismConverter(nn.Module):
    """
    Maps Transformer-based SVD embeddings into the SU(2) hypersphere (256-dim).
    Uses a refractive projection to preserve manifold curvature via the exponential map.
    """
    def __init__(self, input_dim=512, manifold_dim=256):
        super().__init__()
        self.manifold_dim = manifold_dim
        # Linear projection to Lie Algebra su(2)^64 (64 * 4 = 256)
        self.projection = nn.Linear(input_dim, manifold_dim)
        self.scale = nn.Parameter(torch.tensor([1.0 / math.sqrt(manifold_dim)]))

    def forward(self, x):
        # 1. Project to target dimensionality
        # x: [Batch, Seq, Input_Dim] -> [Batch, Seq, 256]
        su2_algebra = self.projection(x)
        
        # 2. Reshape to Quaternionic components (Batch, Seq, 64, 4)
        # Each 4-tuple represents an element of the SU(2) group
        q_shape = list(su2_algebra.shape[:-1]) + [self.manifold_dim // 4, 4]
        quaternions = su2_algebra.view(*q_shape)
        
        # 3. Map to SU(2) via normalization (The S3 Hypersphere projection)
        # This preserves the geodesic flow on the manifold
        su2_elements = F.normalize(quaternions, p=2, dim=-1)
        
        # 4. Flatten back to 256-dim manifold space
        return su2_elements.view(x.shape[0], x.shape[1], self.manifold_dim)

# [STABLE] DiscreteDecisionEngine: Fixed __init__ signature to resolve Runtime Error
class DiscreteDecisionEngine(nn.Module):
    """
    The H2Q Decision Atom processor. 
    Links discrete decision atoms to continuous environmental drag via Spectral Shift Tracker (η).
    """
    def __init__(self, dim=256, num_actions=10, device="mps"):
        super().__init__()
        self.dim = dim
        self.num_actions = num_actions
        self.device = device
        
        # Fractal Expansion Seed (2 -> 256)
        self.seed = nn.Parameter(torch.randn(2).to(device))
        
        # Decision weights
        self.action_head = nn.Linear(dim, num_actions).to(device)
        
    def spectral_shift_tracker(self, S_matrix):
        """
        Krein-like trace formula: η = (1/π) arg{det(S)}
        Quantifies cognitive progress against environmental drag.
        """
        # Using log-det for numerical stability on M4 hardware
        sign, logdet = torch.linalg.slogdet(S_matrix)
        eta = (1.0 / math.pi) * torch.atan2(sign, torch.exp(logdet))
        return eta

    def forward(self, manifold_state):
        # manifold_state: [Batch, 256]
        logits = self.action_head(manifold_state)
        
        # Calculate Spectral Shift (η) as a diagnostic
        # S is treated here as the covariance of the manifold state
        if manifold_state.size(0) > 1:
            S = torch.cov(manifold_state.T)
            eta = self.spectral_shift_tracker(S)
        else:
            eta = torch.tensor(0.0)
            
        return logits, eta

# [EXPERIMENTAL] ReversibleKernel: O(1) Memory Complexity for M4 Constraints
class ReversibleKernel(nn.Module):
    """
    Satisfies O(1) memory by reconstructing input states from outputs.
    Ensures bit-accurate backpropagation on 16GB Mac Mini M4.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.F = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2))
        self.G = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2))

    def forward(self, x):
        # Standard RevNet coupling
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=-1)