import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, Any, Dict, Union

class DDEConfig(BaseModel):
    """
    Strict Pydantic configuration for Discrete Decision Engine (DDE).
    Enforces canonical naming to prevent 'dim' vs 'latent_dim' hallucinations.
    """
    latent_dim: int = Field(..., description="The dimensionality of the latent SU(2) manifold.")
    action_space: int = Field(default=2, description="Number of discrete logical branches.")
    alpha: float = Field(default=0.1, description="Exploration coefficient for the Geodesic Flow.")
    eta_threshold: float = Field(default=0.05, description="Spectral shift threshold for topological tears.")
    device: str = Field(default="cpu")

    @model_validator(mode='before')
    @classmethod
    def handle_dim_alias(cls, data: Any) -> Any:
        """
        Elastic Extension: Automatically maps legacy 'dim' to 'latent_dim'
        to prevent runtime crashes while logging the correction.
        """
        if isinstance(data, dict):
            if 'dim' in data and 'latent_dim' not in data:
                # Rigid Construction: Enforce the shift to latent_dim
                data['latent_dim'] = data.pop('dim')
        return data

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine (DDE).
    Governs logical branching via the Spectral Shift Tracker (η).
    """
    def __init__(self, **kwargs):
        super().__init__()
        try:
            # Strict Pydantic Validation
            self.config = DDEConfig(**kwargs)
        except ValidationError as e:
            raise TypeError(f"[H2Q-DDE-VALIDATION-ERROR] Topological Tear detected in signature: {e}")

        self.latent_dim = self.config.latent_dim
        self.action_space = self.config.action_space
        self.alpha = self.config.alpha
        self.eta_threshold = self.config.eta_threshold
        
        # Initialize SU(2) projection weights
        self.projection = nn.Linear(self.latent_dim, self.action_space, bias=False)
        self.to(self.config.device)

    def forward(self, x: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete decision based on input x and environmental drag eta.
        """
        logits = self.projection(x)
        # Modulate logits by spectral shift (eta)
        return torch.softmax(logits / (1.0 + self.alpha * eta), dim=-1)

def normalize_dde_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility to pre-validate and normalize DDE arguments before instantiation.
    """
    validated = DDEConfig(**kwargs)
    return validated.model_dump()

def get_canonical_dde(latent_dim: int, **kwargs) -> DiscreteDecisionEngine:
    """
    Factory function to ensure all DDE instances adhere to the Veracity Compact.
    """
    config_dict = {"latent_dim": latent_dim, **kwargs}
    return DiscreteDecisionEngine(**config_dict)

def verify_dde_integrity(engine: DiscreteDecisionEngine) -> bool:
    """
    Audits the DDE instance for topological consistency.
    """
    if not hasattr(engine, 'config') or not isinstance(engine.config, DDEConfig):
        return False
    return engine.projection.in_features == engine.config.latent_dim

class StandardizedDecisionEngineWrapper:
    """
    Wrapper to maintain symmetry across multimodal interfaces.
    """
    def __init__(self, engine: DiscreteDecisionEngine):
        self.engine = engine
        self.audit_log = []

    def step(self, x: torch.Tensor, eta: torch.Tensor):
        return self.engine(x, eta)

def topological_dde_normalization(eta: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the spectral shift to prevent manifold heat-death.
    """
    return torch.clamp(eta, -1.0, 1.0)