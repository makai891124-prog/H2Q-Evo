import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine
    
    Standardized signature to resolve 'num_actions' mismatch and unify dimensional nomenclature.
    Governed by Rigid Construction: latent_dim (input) -> action_dim (output).
    """
    def __init__(self, latent_dim: int, action_dim: int):
        """
        Args:
            latent_dim (int): The dimensionality of the cognitive state (SU(2) manifold projection).
            action_dim (int): The number of discrete actions available in the decision space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Rigid Construction: Mapping the latent geodesic flow to discrete logits
        # We use a high-fidelity linear projection to maintain O(1) complexity 
        # consistent with Reversible Kernel requirements.
        self.decision_head = nn.Linear(self.latent_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the phase deflection from continuous latent space to discrete action logits.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, latent_dim)
        Returns:
            torch.Tensor: Logits of shape (batch, action_dim)
        """
        # Ensure input symmetry with the latent_dim atom
        if x.shape[-1] != self.latent_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.latent_dim}, got {x.shape[-1]}")
            
        return self.decision_head(x)

    @property
    def dim(self):
        """Alias for latent_dim to satisfy legacy 'dim' references."""
        return self.latent_dim

    @property
    def num_actions(self):
        """Alias for action_dim to satisfy legacy 'num_actions' references."""
        return self.action_dim