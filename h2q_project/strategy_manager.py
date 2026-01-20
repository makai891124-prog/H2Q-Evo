"""Strategy selection with holomorphic Fueter regularity constraints."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


class HolomorphicStrategyConstraint:
    """
    Verifies strategy feasibility via Fueter regularity conditions.
    
    A function f: ℍ → ℍ is Fueter-regular if:
    ∇·(∇f) = 0  (discrete Laplacian)
    
    Applied to strategy effectiveness manifold:
    ∂²success/∂t² + ∂²success/∂q₁² + ∂²success/∂q₂² + ∂²success/∂q₃² ≈ 0
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.history: Dict[str, List[float]] = {}

    def compute_laplacian(self, strategy: str, values: List[float]) -> float:
        """
        Compute discrete Laplacian: ∑ᵢ (f(x+δ) - 2f(x) + f(x-δ)) / δ²
        
        Measures curvature in strategy effectiveness trajectory.
        Small Laplacian → geodesic (feasible), Large → instability
        """
        if len(values) < 3:
            return 0.0
        
        laplacian_sum = 0.0
        for i in range(1, len(values) - 1):
            second_deriv = values[i + 1] - 2 * values[i] + values[i - 1]
            laplacian_sum += second_deriv ** 2
        
        return np.sqrt(laplacian_sum / max(1, len(values) - 2))

    def is_feasible(self, strategy: str, effectiveness_history: List[float]) -> bool:
        """
        Check if strategy follows holomorphic manifold.
        
        Feasible if Laplacian curvature < threshold
        """
        if strategy not in self.history:
            self.history[strategy] = []
        
        self.history[strategy].extend(effectiveness_history)
        
        # Compute Laplacian curvature
        laplacian = self.compute_laplacian(strategy, self.history[strategy][-10:])
        
        return laplacian < self.threshold

    def get_regularity_score(self, strategy: str) -> float:
        """Score strategy regularity: 1.0 = perfect geodesic, 0.0 = chaotic."""
        if strategy not in self.history or len(self.history[strategy]) < 3:
            return 0.5
        
        laplacian = self.compute_laplacian(strategy, self.history[strategy])
        # Score: 1 - (laplacian / max_laplacian)
        return max(0.0, 1.0 - laplacian / (self.threshold * 2.0))


class StrategyManager:
    """
    Strategy selection with holomorphic Fueter regularity constraints.
    
    Algorithm:
    1. Track all strategies' effectiveness trajectories
    2. Compute Fueter regularity (∇²f = 0 for geodesic paths)
    3. Prefer strategies with high success AND high regularity
    4. Reject unstable strategies (large Laplacian curvature)
    """

    def __init__(self) -> None:
        self.success_counts: Dict[str, int] = {"default": 0}
        self.strategy_histories: Dict[str, List[float]] = {"default": []}
        self.holomorphic_constraint = HolomorphicStrategyConstraint(threshold=0.1)

    def select_best(self, task_info: Dict[str, Any]) -> str:
        """
        Select best strategy using Fueter regularity filtering.
        
        Score = success_rate × regularity_score
        Only consider strategies that are holomorphic-feasible
        """
        if not self.success_counts:
            return "default"
        
        # Compute scores for all strategies
        best_strategy = "default"
        best_score = -1.0
        
        for strategy, count in self.success_counts.items():
            # Get effectiveness history for this strategy
            history = self.strategy_histories.get(strategy, [])
            
            # Check holomorphic feasibility
            if not self.holomorphic_constraint.is_feasible(strategy, history):
                continue  # Skip unstable strategies
            
            # Compute combined score
            success_rate = count / max(1, sum(self.success_counts.values()))
            regularity = self.holomorphic_constraint.get_regularity_score(strategy)
            score = success_rate * regularity
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy

    def update_effectiveness(self, strategy: str, success: bool) -> None:
        """Update strategy effectiveness and holomorphic trajectory."""
        if not strategy:
            strategy = "default"
        
        # Update success count
        current = self.success_counts.get(strategy, 0)
        self.success_counts[strategy] = current + (1 if success else 0)
        
        # Update effectiveness history (for Laplacian computation)
        if strategy not in self.strategy_histories:
            self.strategy_histories[strategy] = []
        
        effectiveness = 1.0 if success else 0.0
        self.strategy_histories[strategy].append(effectiveness)
        
        # Keep history size bounded (last 100 measurements)
        if len(self.strategy_histories[strategy]) > 100:
            self.strategy_histories[strategy] = self.strategy_histories[strategy][-100:]
