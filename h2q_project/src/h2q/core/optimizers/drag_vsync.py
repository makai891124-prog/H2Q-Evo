import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from h2q.core.interface_registry import get_canonical_dde, normalize_dde_kwargs
from h2q.core.sst import SpectralShiftTracker

class DynamicDragVSync(nn.Module):
    """
    DYNAMIC-DRAG-V-SYNC Scheduler
    
    A homeostatic controller that modulates Fractal Expansion delta (δ) and 
    sampling temperature based on the ratio of Spectral Shift (η) to 
    Heat-Death Index (HDI).
    
    Formula:
        R = η / (HDI + ε)
        δ_t = δ_base * sigmoid(R - target_ratio)
        T_t = T_base * tanh(R)
    """
    def __init__(
        self,
        base_delta: float = 0.01,
        base_temp: float = 1.0,
        target_ratio: float = 1.0,
        ema_alpha: float = 0.9,
        dde_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.base_delta = base_delta
        self.base_temp = base_temp
        self.target_ratio = target_ratio
        self.ema_alpha = ema_alpha
        
        # State variables
        self.register_buffer("current_delta", torch.tensor(base_delta))
        self.register_buffer("current_temp", torch.tensor(base_temp))
        self.register_buffer("smoothed_ratio", torch.tensor(target_ratio))
        
        # Initialize DDE using canonical registry to avoid 'dim' keyword errors
        # The registry handles the mapping between LatentConfig and raw kwargs
        safe_kwargs = normalize_dde_kwargs(dde_config or {})
        self.dde = get_canonical_dde(**safe_kwargs)
        
        self.eps = 1e-6

    def calculate_hdi(self, manifold_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Heat-Death Index (HDI) via Spectral Entropy.
        HDI = -sum(p * log(p)) where p are the normalized singular values of the manifold.
        """
        # Optimized for M4 MPS: use svd for small matrices, avoid complex if possible
        _, s, _ = torch.svd(manifold_weights.to(torch.float32))
        p = s / (s.sum() + self.eps)
        hdi = -torch.sum(p * torch.log(p + self.eps))
        return hdi

    def update_homeostasis(self, eta: float, hdi: float) -> Dict[str, float]:
        """
        Modulates δ and Temperature based on the η/HDI ratio.
        """
        # 1. Calculate instantaneous ratio
        instant_ratio = eta / (hdi + self.eps)
        
        # 2. Update EMA of the ratio to prevent oscillatory drag
        self.smoothed_ratio = self.ema_alpha * self.smoothed_ratio + (1 - self.ema_alpha) * instant_ratio
        
        # 3. Modulate Delta (Fractal Expansion Step)
        # If ratio is high (high intelligence/low entropy), we expand faster
        # If ratio is low (approaching heat-death), we shrink delta to stabilize
        scale_factor = torch.sigmoid(self.smoothed_ratio - self.target_ratio)
        self.current_delta = self.base_delta * (0.5 + scale_factor) 
        
        # 4. Modulate Temperature (Sampling Stochasticity)
        # High η/HDI allows for higher 'cognitive heat' (exploration)
        self.current_temp = self.base_temp * torch.tanh(self.smoothed_ratio)
        
        return {
            "delta": self.current_delta.item(),
            "temperature": self.current_temp.item(),
            "ratio": self.smoothed_ratio.item()
        }

    def forward(self, eta: torch.Tensor, manifold_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass for integration into the FDC training loop.
        """
        hdi = self.calculate_hdi(manifold_weights)
        metrics = self.update_homeostasis(eta.item(), hdi.item())
        
        return {
            "delta": self.current_delta,
            "temperature": self.current_temp,
            "hdi": hdi,
            "eta_hdi_ratio": self.smoothed_ratio
        }

# Experimental: Logic Veracity Audit Hook
def audit_v_sync_integrity(scheduler: DynamicDragVSync, d_fueter: torch.Tensor):
    """
    Labels the current state as 'Topological Tear' if Fueter residual exceeds threshold.
    """
    is_stable = torch.norm(d_fueter) < 0.05
    return {"status": "STABLE" if is_stable else "TOPOLOGICAL_TEAR", "residual": torch.norm(d_fueter).item()}