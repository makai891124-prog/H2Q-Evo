import torch
import torch.nn as nn
import torch.nn.functional as F
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from h2q.core.interface_registry import get_canonical_dde

class ThermodynamicModalityGate(nn.Module):
    """
    Thermodynamic Modality Gate (TMG)
    Dynamically scales weights between Audio, Vision, and Text streams based on 
    their instantaneous modality-specific Heat-Death Index (HDI).
    
    HDI (H) = -Σ p log(p), where p is the normalized singular value spectrum.
    """
    def __init__(self, latent_dim: int, temperature: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_temperature = temperature
        
        # Initialize DDE via canonical registry to avoid 'dim' keyword errors
        # The registry handles the mapping of configuration to the engine instance.
        self.dde = get_canonical_dde()
        
        # Modality-specific projection to a common manifold space if needed
        self.projections = nn.ModuleDict({
            'audio': nn.Linear(latent_dim, latent_dim),
            'vision': nn.Linear(latent_dim, latent_dim),
            'text': nn.Linear(latent_dim, latent_dim)
        })

    def calculate_hdi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Von Neumann entropy (Heat-Death Index) of the singular value spectrum.
        Grounding: MPS-safe SVD implementation.
        """
        # Ensure 2D for SVD: [Batch, Features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Singular Value Decomposition
        # Note: torch.linalg.svdvals is more efficient for entropy calculation
        s = torch.linalg.svdvals(x)
        
        # Normalize singular values to create a probability distribution (p)
        s_sq = s ** 2
        p = s_sq / (torch.sum(s_sq, dim=-1, keepdim=True) + 1e-9)
        
        # Calculate Von Neumann Entropy: H = -Σ p log(p)
        hdi = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        
        # Normalize HDI by max possible entropy (log of rank)
        max_h = torch.log(torch.tensor(float(s.size(-1)), device=x.device))
        return hdi / (max_h + 1e-9)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor, text: torch.Tensor):
        """
        Performs thermodynamic gating across modalities.
        """
        # 1. Calculate instantaneous HDI for each modality
        h_audio = self.calculate_hdi(audio)   # [Batch]
        h_vision = self.calculate_hdi(vision) # [Batch]
        h_text = self.calculate_hdi(text)     # [Batch]
        
        # 2. Stack HDIs for comparison
        hdis = torch.stack([h_audio, h_vision, h_text], dim=1) # [Batch, 3]
        
        # 3. Use DDE to determine if any modality has reached a 'Topological Tear' (HDI > threshold)
        # We pass the mean HDI to the DDE for meta-modulation
        dde_input = hdis.mean(dim=0)
        # DDE logic: If HDI is too high, the modality is 'dying' (noise-dominated), reduce its influence.
        
        # 4. Calculate Gating Weights
        # Inverse relationship: Lower HDI (higher order) -> Higher Weight
        # We use negative HDI in softmax to prioritize stable manifolds
        weights = F.softmax(-hdis / self.initial_temperature, dim=1) # [Batch, 3]
        
        # 5. Apply Gating
        w_a, w_v, w_t = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        fused_latent = (w_a * self.projections['audio'](audio) +
                        w_v * self.projections['vision'](vision) +
                        w_t * self.projections['text'](text))
        
        return fused_latent, weights, hdis

    def verify_thermodynamic_symmetry(self, weights: torch.Tensor):
        """
        Rigid Construction Check: Ensure weights sum to unity (Unitary Integrity).
        """
        residual = torch.abs(weights.sum(dim=1) - 1.0).max()
        if residual > 1e-5:
            raise ValueError(f"Thermodynamic Symmetry Broken: Weight sum residual {residual}")
        return True