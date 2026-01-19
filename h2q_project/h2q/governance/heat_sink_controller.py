import torch
import torch.nn as nn
import torch.linalg as linalg
from typing import Optional, Tuple
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from h2q.core.sst import SpectralShiftTracker

class TopologicalHeatSinkController(nn.Module):
    """
    Topological Heat-Sink Controller (THSC)
    
    Governs the Spectral Drag coefficient μ(E) to prevent manifold rank collapse.
    Monitors the Heat-Death Index (HDI) derived from Von Neumann entropy.
    """
    def __init__(
        self,
        threshold: float = 0.85,
        base_drag: float = 0.01,
        recovery_rate: float = 0.1,
        latent_dim: int = 256
    ):
        super().__init__()
        self.threshold = threshold
        self.base_drag = base_drag
        self.recovery_rate = recovery_rate
        
        # Correcting DDE initialization based on feedback: avoiding 'dim' keyword
        # Using LatentConfig as defined in h2q.core.discrete_decision_engine
        config = LatentConfig(latent_dim=latent_dim)
        self.dde = DiscreteDecisionEngine(config=config)
        self.sst = SpectralShiftTracker()
        
        self.register_buffer("current_hdi", torch.tensor(0.0))
        self.register_buffer("active_mu", torch.tensor(base_drag))

    def calculate_hdi(self, singular_values: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Heat-Death Index (HDI) using normalized Von Neumann entropy.
        HDI = -sum(p * log(p)) / log(N)
        """
        # Ensure stability for small singular values
        s_sq = torch.pow(singular_values + 1e-9, 2)
        probs = s_sq / torch.sum(s_sq)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        max_entropy = torch.log(torch.tensor(float(singular_values.size(-1))))
        
        hdi = entropy / max_entropy
        return hdi

    def modulate_drag(self, hdi: torch.Tensor) -> torch.Tensor:
        """
        Modulates μ(E) based on HDI proximity to threshold.
        If HDI > 0.85, drag increases exponentially to slow geodesic flow.
        """
        if hdi > self.threshold:
            # Exponential braking: μ increases as HDI approaches 1.0
            excess = hdi - self.threshold
            scaling = torch.exp(excess * 10.0) 
            new_mu = self.base_drag * scaling
        else:
            # Linear recovery towards base drag
            new_mu = self.active_mu * (1.0 - self.recovery_rate) + self.base_drag * self.recovery_rate
            
        return new_mu

    def forward(
        self, 
        manifold_weights: torch.Tensor, 
        external_drag: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a governance step.
        
        Args:
            manifold_weights: The SU(2) manifold representation [..., 256]
            external_drag: Optional override for base drag
            
        Returns:
            Tuple of (modulated_mu, hdi_metrics)
        """
        device = manifold_weights.device
        
        # 1. Compute Singular Value Spectrum (MPS compatible)
        # We treat the 256-dim manifold as a flattened representation of the knots
        # Reshaping to [64, 4] to analyze quaternionic knot stability
        knot_matrix = manifold_weights.view(-1, 64, 4)
        _, s, _ = linalg.svd(knot_matrix)
        
        # 2. Calculate HDI
        avg_s = torch.mean(s, dim=0)
        hdi = self.calculate_hdi(avg_s)
        self.current_hdi.copy_(hdi)

        # 3. Update Drag Coefficient
        if external_drag is not None:
            self.base_drag = external_drag
            
        modulated_mu = self.modulate_drag(hdi)
        self.active_mu.copy_(modulated_mu)

        # 4. Veracity Check via DDE
        # DDE ensures the decision to modulate drag aligns with cognitive transition stability
        dde_input = torch.cat([hdi.unsqueeze(0), modulated_mu.unsqueeze(0)], dim=0)
        _ = self.dde(dde_input)

        return modulated_mu, hdi

    def get_governance_report(self):
        return {
            "hdi": self.current_hdi.item(),
            "spectral_drag": self.active_mu.item(),
            "status": "CRITICAL_BRAKING" if self.current_hdi > self.threshold else "STABLE_FLOW"
        }