import torch
import torch.nn as nn
from typing import Optional, Tuple
from h2q.core.ttd_scheduler import TopologicalTimeDilation, TTDState
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.logic_auditing import HolomorphicAuditKernel

class HolomorphicAutoregressiveDecoder(nn.Module):
    """
    H2Q Holomorphic Autoregressive Decoder with Topological Braking.
    
    This module implements the 'Topological Braking' mechanism, which uses the 
    Topological Time Dilation (TTD) scheduler to modulate recursion depth (k) 
    based on Fueter residuals (Df). 
    
    Threshold: Df > 0.05 indicates a 'topological tear' (hallucination risk).
    """
    def __init__(
        self,
        latent_dim: int = 256,
        vocab_size: int = 50257,
        max_k: int = 16,
        df_threshold: float = 0.05
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_k = max_k
        self.df_threshold = df_threshold

        # Initialize DDE using canonical factory to avoid 'dim' kwarg errors
        # as identified in the Veracity Compact audit.
        self.dde = get_canonical_dde(latent_dim=latent_dim)
        
        # Audit kernel for calculating Discrete Fueter Operator (Df)
        self.audit_kernel = HolomorphicAuditKernel(dim=latent_dim)
        
        # TTD Scheduler for dynamic recursion depth modulation
        self.ttd_scheduler = TopologicalTimeDilation(
            base_depth=1,
            max_depth=max_k,
            threshold=df_threshold
        )

        self.token_embedding = nn.Embedding(vocab_size, latent_dim)
        self.output_projection = nn.Linear(latent_dim, vocab_size)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        past_kv: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Performs a single generation step with Topological Braking.
        """
        # 1. Map tokens to Quaternionic Manifold (S³)
        h = self.token_embedding(input_ids) # [B, L, D]

        # 2. Compute Fueter Residual (Df) to detect topological tears
        # Df measures the non-analyticity of the quaternionic field
        df_residual = self.audit_kernel.compute_residual(h)

        # 3. TOPOLOGICAL BRAKING: Query TTD Scheduler
        # If df_residual approaches 0.05, ttd_state.k increases.
        ttd_state: TTDState = self.ttd_scheduler.update(df_residual)
        current_k = ttd_state.recursion_depth

        # 4. Geodesic Flow with Dynamic Recursion Depth k
        # Higher k = slower generation velocity, higher reasoning veracity.
        for _ in range(current_k):
            h = self.dde(h, context=past_kv)
            # M4 Optimization: Hamilton products mapped to 16x16 tiled AMX
            # logic is handled within the DDE/HamiltonProduct kernels.

        logits = self.output_projection(h)

        return logits, h, df_residual

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int = 32
    ) -> torch.Tensor:
        """
        Autoregressive generation loop with active braking.
        """
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits, _, df = self.forward(generated[:, -1:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Log braking event if threshold is breached
            if df > self.df_threshold:
                # Activate topological braking by reducing sampling temperature
                # and forcing the TTD scheduler to increase recursion depth.
                cooled_logits = logits[:, -1, :] / 2.0  # temperature=0.5
                next_token = torch.argmax(cooled_logits, dim=-1, keepdim=True)
                # Optionally update scheduler state to slow further steps
                self.ttd_scheduler.current_k = min(self.ttd_scheduler.max_depth, self.ttd_scheduler.current_k + 1)
                
            generated = torch.cat([generated, next_token], dim=-1)
            
        return generated

    def get_braking_metrics(self) -> dict:
        """Returns the current state of the TTD scheduler."""
        return {
            "current_k": self.ttd_scheduler.current_k,
            "velocity_reduction": 1.0 / self.ttd_scheduler.current_k,
            "tear_detected": self.ttd_scheduler.last_df > self.df_threshold
        }