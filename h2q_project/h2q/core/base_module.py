import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pydantic import ValidationError

# Accessing verified components from the Global Interface Registry
from h2q.core.discrete_decision_engine import LatentConfig, DiscreteDecisionEngine
from h2q.core.interface_registry import normalize_dde_kwargs, get_canonical_dde

class H2Q_Base_Module(nn.Module):
    """
    [STABLE] H2Q_Base_Module
    
    The foundational architectural unit for the H2Q framework. This module unifies 
    DiscreteDecisionEngine (DDE) instantiations to resolve the recurrent 'unexpected 
    keyword argument dim' error by enforcing Pydantic-validated LatentConfig injection.
    
    Attributes:
        config (LatentConfig): Validated configuration for the latent space and DDE.
        dde (DiscreteDecisionEngine): The standardized decision engine for the module.
    """
    def __init__(self, config: Optional[LatentConfig] = None, **kwargs):
        super().__init__()
        
        # 1. IDENTIFY_ATOMS: Extract and normalize parameters
        # normalize_dde_kwargs handles the mapping of legacy 'dim' to 'latent_dim'
        # as identified in the Global Interface Registry (h2q.core.interface_registry).
        normalized_params = normalize_dde_kwargs(**kwargs)
        
        # 2. VERIFY_SYMMETRY: Ensure config object and kwargs are reconciled
        try:
            if config is not None:
                self.config = config
            else:
                # Construct LatentConfig from normalized parameters
                # This enforces the schema defined in h2q.core.discrete_decision_engine
                self.config = LatentConfig(**normalized_params)
        except ValidationError as e:
            # Metacognitive Loop: Provide clear feedback on configuration failure
            raise ValueError(f"[H2Q_Base_Module] Configuration validation failed: {str(e)}")
            
        # 3. GROUNDING: Instantiate DDE via the canonical factory
        # This ensures that the DDE receives a validated config object rather than raw kwargs,
        # preventing the 'unexpected keyword argument dim' error at the source.
        self.dde = get_canonical_dde(self.config)

    def forward(self, *args, **kwargs):
        """
        Abstract forward method. Subclasses must implement the unitary geodesic flow logic.
        """
        raise NotImplementedError("Subclasses of H2Q_Base_Module must implement the forward pass.")

    @property
    def eta(self) -> torch.Tensor:
        """
        Spectral Tracking (η): Quantifies learning progress against environmental drag.
        Extracted from the internal DiscreteDecisionEngine.
        """
        if hasattr(self.dde, 'eta'):
            return self.dde.eta
        return torch.tensor(0.0, device=self.device)

    @property
    def device(self) -> torch.device:
        """Utility to track device placement, optimized for MPS/CPU fallback on Mac Mini M4."""
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(latent_dim={self.config.latent_dim})"