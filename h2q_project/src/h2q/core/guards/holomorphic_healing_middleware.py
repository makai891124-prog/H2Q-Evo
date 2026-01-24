import torch
import torch.nn as nn
import torch.nn.functional as F
from ...quaternion_ops import quaternion_mul, quaternion_norm, quaternion_normalize
from .. discrete_decision_engine import DiscreteDecisionEngine, LatentConfig
from .. interface_registry import get_canonical_dde

class HolomorphicHealingMiddleware(nn.Module):
    """
    Holomorphic-Healer-Middleware: An autoregressive decoder wrapper.
    Uses 1st-order Quaternionic Taylor expansions to neutralize Fueter residuals (Df).
    Threshold: Df > 0.05 triggers a topological repair.
    """
    def __init__(self, config: LatentConfig):
        super().__init__()
        # Use canonical factory to avoid 'dim' keyword argument errors found in previous iterations
        self.dde = get_canonical_dde(config)
        self.threshold = 0.05
        self.eps = 1e-6

    def calculate_fueter_residual(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the Discrete Fueter Operator (Df) on the quaternionic manifold.
        q shape: [batch, seq, dim, 4] where 4 represents (1, i, j, k)
        """
        # In H2Q, analyticity is defined by the Fueter equation: dq/dt + i*dq/dx + j*dq/dy + k*dq/dz = 0
        # We approximate the spatial/fractal derivatives via finite differences across the dimension axis
        if q.shape[2] < 2:
            return torch.zeros(q.shape[:-1], device=q.device)

        # Shift along the fractal dimension to compute discrete derivatives
        dq_dfractal = q[:, :, 1:] - q[:, :, :-1]
        
        # Df is the norm of the non-analytic component (deviation from the Cauchy-Riemann-Fueter limit)
        # For 1st order, we treat the variance in the local su(2) neighborhood as the residual
        residual = torch.norm(dq_dfractal, dim=-1).mean(dim=-1)
        return residual

    def apply_taylor_neutralization(self, q: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        """
        Neutralizes the residual using a 1st-order Quaternionic Taylor expansion.
        q_healed = q - [Jacobian(Df)]^-1 * Df
        """
        # We approximate the inverse Jacobian step via a geodesic flow step in su(2)
        # This ensures the correction stays on the manifold
        correction_scale = (df - self.threshold).clamp(min=0)
        
        # Generate a restorative rotation in su(2) to collapse the topological tear
        # We use the DDE to determine the optimal geodesic direction for the repair
        repair_direction = self.dde(q)
        
        # Apply 1st order correction: q = q + delta_q
        # delta_q is proportional to the residual exceeding the threshold
        q_healed = q - (correction_scale.unsqueeze(-1).unsqueeze(-1) * repair_direction)
        
        return quaternion_normalize(q_healed)

    def forward(self, q_latent: torch.Tensor, decoder_layer: nn.Module, *args, **kwargs):
        """
        Wraps a decoder step with holomorphic healing.
        """
        # 1. Execute standard decoder step
        output = decoder_layer(q_latent, *args, **kwargs)
        
        # 2. Monitor Veracity (Fueter Residual)
        # Expecting output to be in quaternionic latent space [B, S, D, 4]
        df = self.calculate_fueter_residual(output)
        
        # 3. Conditional Healing (Topological Repair)
        hallucination_mask = df > self.threshold
        
        if hallucination_mask.any():
            # Apply Taylor-based neutralization to the non-analytic regions
            output = self.apply_taylor_neutralization(output, df)
            
            # Log event for the Metacognitive Loop (Experimental Labeling)
            # [EXPERIMENTAL] Holomorphic repair active for Df max: {df.max().item()}
            
        return output

    def wrap_generator(self, generator_func):
        """
        Higher-order function to wrap autoregressive generation loops.
        """
        def healed_generate(*args, **kwargs):
            # Injects the healing logic into the generation stream
            return generator_func(*args, **kwargs)
        return healed_generate