import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteDecisionEngine(nn.Module):
    """
    H2Q Discrete Decision Engine (DDE).
    
    Standardized to (latent_dim, action_dim) to resolve signature mismatches.
    This engine maps the 256-dimensional topological manifold (SU(2) geodesic flow)
    to a discrete action space while maintaining compatibility with the 
    Spectral Shift Tracker (η).
    """
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        # RIGID CONSTRUCTION: Atomize dimensions
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # ELASTIC WEAVING: Projective mapping from manifold to action logits.
        # We use a streamlined linear projection to minimize memory overhead
        # on Mac Mini M4 (16GB) constraints.
        self.projector = nn.Linear(latent_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass projecting latent atoms to action probabilities.
        Args:
            x (torch.Tensor): Input tensor from the H2Q manifold [Batch, latent_dim]
        Returns:
            torch.Tensor: Action logits [Batch, action_dim]
        """
        # VERIFY_SYMMETRY: Ensure input matches the latent_dim atom
        if x.shape[-1] != self.latent_dim:
            raise ValueError(f"Input dimension {x.shape[-1]} does not match latent_dim {self.latent_dim}")
            
        return self.projector(x)

    def get_action_distribution(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Returns a softmax distribution over the action space.
        """
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)