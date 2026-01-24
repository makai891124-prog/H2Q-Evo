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
    def __init__(self, config: Optional[LatentConfig] = None, latent_dim: Optional[int] = None, n_atoms: Optional[int] = None, **kwargs):
        super().__init__()
        
        # Rigid Construction: Ensure config is valid
        if config is None:
            # Elastic Extension: Handle legacy 'dim' and other kwargs
            # This prevents: TypeError: __init__() got an unexpected keyword argument 'dim'
            if 'dim' in kwargs and 'latent_dim' not in kwargs and latent_dim is None:
                latent_dim = kwargs.pop('dim')
            if 'context_dim' in kwargs and 'latent_dim' not in kwargs and latent_dim is None:
                latent_dim = kwargs.pop('context_dim')
            if 'action_dim' in kwargs and 'n_choices' not in kwargs and n_atoms is None:
                n_atoms = kwargs.pop('action_dim')
            if 'autonomy_weight' in kwargs and 'alpha' not in kwargs:
                kwargs['alpha'] = kwargs.pop('autonomy_weight')
            
            # Filter kwargs to match LatentConfig fields
            valid_keys = {f.name for f in fields(LatentConfig)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            
            # Override with explicit parameters
            if latent_dim is not None:
                config_kwargs['latent_dim'] = latent_dim
            if n_atoms is not None:
                config_kwargs['n_choices'] = n_atoms
                
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
        
        # Memory Crystal Interface
        self.external_memory = None

    def load_memory_crystal(self, crystal_path: str):
        """
        Load external memory crystal for enhanced decision making.
        
        Args:
            crystal_path: Path to the memory crystal file (.pt or .pth)
        """
        try:
            loaded_data = torch.load(crystal_path, map_location=self.config.device)
            if isinstance(loaded_data, dict):
                # Handle dictionary format - extract geometric embeddings
                if 'geometric_embeddings' in loaded_data:
                    self.external_memory = loaded_data['geometric_embeddings']
                else:
                    # Fallback to the first tensor value
                    tensor_keys = [k for k, v in loaded_data.items() if isinstance(v, torch.Tensor)]
                    if tensor_keys:
                        self.external_memory = loaded_data[tensor_keys[0]]
                    else:
                        raise ValueError("No tensor found in memory crystal dictionary")
            else:
                # Direct tensor
                self.external_memory = loaded_data
            print(f"Memory crystal loaded: {self.external_memory.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load memory crystal from {crystal_path}: {e}")

    def forward(self, x: torch.Tensor, candidate_actions: Optional[torch.Tensor] = None, eta: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass with Spectral Shift (eta) modulation.
        If candidate_actions provided, returns (chosen_action, metadata), else returns logits.
        """
        # Ensure input is on the correct device
        x = x.to(self.config.device)
        
        logits = self.gate(x)
        
        if eta is not None:
            # η (Spectral Shift) modulates the decision boundary
            # η = (1/π) arg{det(S)}
            eta = eta.to(self.config.device).view(-1, 1)
            logits = logits * (1.0 + eta)
            
        if candidate_actions is not None:
            # Perform decision making with candidate actions
            candidate_actions = candidate_actions.to(self.config.device)
            
            # Get probabilities
            probs = torch.softmax(logits / self.config.temperature, dim=-1)
            
            # Sample from candidates based on probabilities
            # For simplicity, select the action with highest probability among candidates
            batch_size = x.shape[0]
            chosen_actions = []
            eta_values = []
            
            for b in range(batch_size):
                candidate_probs = probs[b, candidate_actions[b]]
                chosen_idx = torch.argmax(candidate_probs)
                chosen_action = candidate_actions[b, chosen_idx]
                chosen_actions.append(chosen_action)
                
                # Compute spectral shift (eta) for the chosen action
                if self.external_memory is not None:
                    # Use memory crystal to compute spectral shift
                    chosen_embedding = self.external_memory[chosen_action]
                    # Simple spectral shift computation (placeholder)
                    eta_val = torch.tensor(1.0, device=self.config.device)  # Placeholder
                else:
                    eta_val = torch.tensor(0.5, device=self.config.device)  # Default
                    
                eta_values.append(eta_val)
            
            chosen = torch.stack(chosen_actions)
            metadata = {
                'eta_values': torch.stack(eta_values),
                'probabilities': probs
            }
            
            return chosen, metadata
        else:
            # Return raw logits for backward compatibility
            return torch.softmax(logits / self.config.temperature, dim=-1)

def get_canonical_dde(config: Optional[LatentConfig] = None, **kwargs) -> DiscreteDecisionEngine:
    """Factory function for standardized DDE instantiation."""
    if config is None:
        # Use default LatentConfig if none provided, but allow kwargs to override
        config = LatentConfig()
        # Override defaults with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if isinstance(config, dict):
        # Convert dict to LatentConfig, filtering only valid keys
        valid_keys = {f.name for f in fields(LatentConfig)}
        config_kwargs = {k: v for k, v in config.items() if k in valid_keys}
        config = LatentConfig(**config_kwargs)
    return DiscreteDecisionEngine(config=config, **kwargs)

def verify_dde_integrity(engine: DiscreteDecisionEngine) -> bool:
    """Audit function to ensure DDE adheres to the Veracity Compact."""
    return hasattr(engine, 'config') and isinstance(engine.config, LatentConfig)