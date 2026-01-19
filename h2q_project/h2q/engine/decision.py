import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiscreteDecisionEngine(nn.Module):
    """
    [STABLE] Standardized Discrete Decision Engine (DDE).
    
    Resolves signature mismatches across the H2Q project by unifying 'dim', 
    'latent_dim', and 'input_dim' into a single canonical 'input_dim'.
    
    Architecture: 
    Maps the 256-dimensional SU(2) manifold (geodesic state) to a discrete 
    probability distribution representing cognitive branching points.
    """
    def __init__(
        self, 
        input_dim: int = 256, 
        num_actions: int = 2, 
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # RIGID CONSTRUCTION: Resolve signature aliases
        # This handles legacy calls using 'dim' or 'latent_dim'
        self.input_dim = kwargs.get('dim', kwargs.get('latent_dim', input_dim))
        self.num_actions = num_actions
        self.temperature = temperature

        # SYMMETRY VERIFICATION: Ensure manifold alignment
        # The H2Q manifold is typically 256-D (64 Quaternions)
        self.projection = nn.Linear(self.input_dim, self.num_actions)
        
        # ELASTIC EXTENSION: Spectral Shift Tracker (η) placeholder
        # Measures cognitive deflection during decision making
        self.register_buffer("eta", torch.tensor(0.0))

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass projecting manifold coordinates to action space.
        
        Args:
            x: Input tensor from the topological manifold [Batch, Input_Dim]
            return_logits: If True, returns raw scores before softmax
        """
        # Ensure input matches the standardized dimension
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"DDE Dimension Mismatch: Expected {self.input_dim}, got {x.shape[-1]}")

        logits = self.projection(x) / self.temperature
        
        if return_logits:
            return logits
            
        return F.softmax(logits, dim=-1)

    def update_temperature(self, new_temp: float):
        """Adjusts the stochasticity of the geodesic flow."""
        self.temperature = max(0.01, new_temp)

    def __repr__(self):
        return f"DiscreteDecisionEngine(input_dim={self.input_dim}, num_actions={self.num_actions}, temp={self.temperature})"