import torch
import math
from torch.optim import Optimizer
from typing import List, Optional, Callable

class SpectralEntropyLR(Optimizer):
    """
    [STABLE] SED-LR: Spectral Entropy-Driven Learning Rate Wrapper.
    
    Governed by the H2Q Veracity Compact. This optimizer modulates the learning rate 
    based on the 'Heat-Death Index' (Singular Value Entropy) of the 256-dimensional 
    quaternionic manifold to prevent manifold collapse.
    
    Symmetry: Maps the Spectral Shift Tracker (eta) to the local learning rate scale.
    """
    def __init__(
        self, 
        params,
        base_optimizer_cls: Callable,
        target_entropy: float = 0.8,
        sensitivity: float = 0.5,
        min_lr_scale: float = 0.1,
        max_lr_scale: float = 2.0,
        **kwargs
    ):
        # Rigid Construction: Ensure base optimizer is initialized without 'dim' conflicts
        self.optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        
        self.target_entropy = target_entropy
        self.sensitivity = sensitivity
        self.min_lr_scale = min_lr_scale
        self.max_lr_scale = max_lr_scale
        
        # Track the Heat-Death Index (HDI)
        self.state['hdi'] = 1.0
        self.state['step_count'] = 0

    @torch.no_grad()
    def _calculate_manifold_entropy(self) -> float:
        """
        Calculates the Von Neumann Entropy of the singular values across 
        representative weight matrices in the 256-dim manifold.
        """
        entropies = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.dim() < 2:
                    continue
                
                # Flatten to 2D for SVD analysis
                flat_p = p.view(p.size(0), -1)
                
                # Elastic Extension: Use MPS-optimized SVD if available
                # Mac Mini M4 (MPS) handles svdvals efficiently for matrices < 4096
                try:
                    s = torch.linalg.svdvals(flat_p)
                except RuntimeError:
                    # Fallback for non-square or singular edge cases
                    continue
                
                # Normalize singular values to a probability distribution
                s_norm = s / (torch.sum(s) + 1e-12)
                
                # Calculate Entropy: H = -sum(p * log(p))
                entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-12))
                
                # Normalize by max possible entropy (log of rank)
                max_h = math.log(s.size(0))
                normalized_h = entropy / (max_h + 1e-12)
                entropies.append(normalized_h)
        
        if not entropies:
            return 1.0
            
        return torch.stack(entropies).mean().item()

    def step(self, closure=None):
        """
        Performs a single optimization step with spectral modulation.
        """
        # 1. Calculate current Heat-Death Index
        hdi = self._calculate_manifold_entropy()
        self.state['hdi'] = hdi
        
        # 2. Compute Modulation Factor
        # If HDI < target (Collapse), increase LR to escape local minima
        # If HDI > target (Heat Death/Chaos), decrease LR to stabilize
        # Formula: scale = exp(sensitivity * (target - hdi))
        lr_scale = math.exp(self.sensitivity * (self.target_entropy - hdi))
        lr_scale = max(self.min_lr_scale, min(self.max_lr_scale, lr_scale))
        
        # 3. Apply modulation to all param groups
        for group in self.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']
            
            group['lr'] = group['initial_lr'] * lr_scale
            
        # 4. Execute base optimizer step
        loss = self.optimizer.step(closure)
        
        self.state['step_count'] += 1
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'base_state': self.optimizer.state_dict(),
            'hdi_state': self.state
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['base_state'])
        self.state.update(state_dict['hdi_state'])
