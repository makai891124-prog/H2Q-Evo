import torch
import torch.nn as nn
from dataclasses import dataclass, fields
from typing import Optional, Any, Dict

@dataclass
class LatentConfig:
    """Canonical configuration for H2Q Latent Space and Decision Engines."""
    latent_dim: int = 256
    n_choices: int = 2
    temperature: float = 1.0
    alpha: float = 0.1
    device: str = "mps"

class DiscreteDecisionEngine(nn.Module):
    """
    Standardized Discrete Decision Engine (DDE).
    
    This implementation follows the LatentConfig pattern to ensure 
    compatibility across the H2Q manifold. It explicitly handles 
    legacy 'dim' arguments to prevent RuntimeErrors.
    """
    def __init__(self, config: Optional[LatentConfig] = None, **kwargs):
        super().__init__()
        
        # Rigid Construction: Ensure config is valid
        if config is None:
            # Elastic Extension: Handle legacy 'dim' and other kwargs
            # This prevents: TypeError: __init__() got an unexpected keyword argument 'dim'
            if 'dim' in kwargs and 'latent_dim' not in kwargs:
                kwargs['latent_dim'] = kwargs.pop('dim')
            
            # Filter kwargs to match LatentConfig fields
            valid_keys = {f.name for f in fields(LatentConfig)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = LatentConfig(**config_kwargs)
        else:
            self.config = config

        # Symmetry Verification: Map config to internal state
        self.latent_dim = self.config.latent_dim
        self.n_choices = self.config.n_choices
        
        # Decision Manifold: SU(2) projection
        self.gate = nn.Linear(self.latent_dim, self.n_choices)
        
        # Device Grounding
        self.to(self.config.device)

    def forward(self, x: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Spectral Shift (eta) modulation.
        """
        # Ensure input is on the correct device
        x = x.to(self.config.device)
        
        logits = self.gate(x)
        
        if eta is not None:
            # η (Spectral Shift) modulates the decision boundary
            # η = (1/π) arg{det(S)}
            eta = eta.to(self.config.device).view(-1, 1)
            logits = logits * (1.0 + eta)
            
        return torch.softmax(logits / self.config.temperature, dim=-1)

def get_canonical_dde(**kwargs) -> DiscreteDecisionEngine:
    """Factory function for standardized DDE instantiation."""
    return DiscreteDecisionEngine(**kwargs)

def verify_dde_integrity(engine: DiscreteDecisionEngine) -> bool:
    """Audit function to ensure DDE adheres to the Veracity Compact."""
    return hasattr(engine, 'config') and isinstance(engine.config, LatentConfig)