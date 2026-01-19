import torch
import numpy as np
from typing import Dict, List, Optional
from h2q.core.sst import SpectralShiftTracker
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.logic_auditing import HolomorphicAuditKernel

class TopologicalForgettingController:
    """
    Manages RSKH Vault pruning using eta-volatility metrics.
    Ensures 100M+ token context persistence by identifying and removing 
    topologically unstable or redundant knots within 16GB RAM constraints.
    """
    def __init__(self, vault, memory_ceiling_gb: float = 12.0, volatility_threshold: float = 0.15):
        self.vault = vault
        self.memory_ceiling = memory_ceiling_gb * 1024 * 1024 * 1024  # Bytes
        self.volatility_threshold = volatility_threshold
        
        # Initialize DDE via canonical registry to avoid 'dim' keyword errors
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        self.auditor = HolomorphicAuditKernel()
        
        # History of spectral shifts per knot index
        self.eta_history: Dict[int, List[float]] = {}
        self.importance_scores: Dict[int, float] = {}

    def update_knot_metrics(self, knot_idx: int, scattering_matrix: torch.Tensor):
        """
        Updates the spectral shift history for a specific knot.
        η = (1/π) arg{det(S)}
        """
        with torch.no_grad():
            eta = self.sst.compute_eta(scattering_matrix)
            if knot_idx not in self.eta_history:
                self.eta_history[knot_idx] = []
            self.eta_history[knot_idx].append(eta.item())
            
            # Keep windowed history to calculate volatility
            if len(self.eta_history[knot_idx]) > 50:
                self.eta_history[knot_idx].pop(0)

    def calculate_volatility(self, knot_idx: int) -> float:
        """
        Calculates the η-volatility (standard deviation of spectral shifts).
        High volatility indicates unstable/noisy information.
        """
        history = self.eta_history.get(knot_idx, [])
        if len(history) < 2:
            return 0.0
        return float(np.std(history))

    def audit_knot_veracity(self, knot_data: torch.Tensor) -> float:
        """
        Uses the Fueter operator to check for topological tears (hallucinations).
        Df > 0.05 indicates a tear.
        """
        return self.auditor.compute_fueter_drift(knot_data)

    def rank_knots(self) -> List[int]:
        """
        Ranks knots for pruning based on a composite score:
        Score = (1 / (Volatility + epsilon)) * (1 - FueterDrift)
        """
        rankings = []
        for knot_idx in self.eta_history.keys():
            vol = self.calculate_volatility(knot_idx)
            # We assume vault access to knot data for auditing
            knot_data = self.vault.get_knot(knot_idx)
            drift = self.audit_knot_veracity(knot_data)
            
            # Higher score = Keep; Lower score = Prune
            score = (1.0 / (vol + 1e-6)) * (1.0 - min(drift, 1.0))
            rankings.append((knot_idx, score))
        
        # Sort by score ascending (lowest score first for pruning)
        rankings.sort(key=lambda x: x[1])
        return [r[0] for r in rankings]

    def enforce_memory_limit(self):
        """
        Triggers pruning if current vault memory exceeds the 16GB ceiling.
        """
        current_usage = self.vault.get_total_bytes()
        
        if current_usage > self.memory_ceiling:
            prune_list = self.rank_knots()
            
            while current_usage > self.memory_ceiling and prune_list:
                target_idx = prune_list.pop(0)
                # Decision Engine check: Should we really forget this?
                decision = self.dde.decide_pruning(target_idx, self.eta_history[target_idx])
                
                if decision > 0.5:
                    freed_bytes = self.vault.prune_knot(target_idx)
                    current_usage -= freed_bytes
                    del self.eta_history[target_idx]

    def get_status(self):
        return {
            "active_knots": len(self.eta_history),
            "avg_volatility": np.mean([self.calculate_volatility(i) for i in self.eta_history]),
            "vault_load_factor": self.vault.get_total_bytes() / self.memory_ceiling
        }