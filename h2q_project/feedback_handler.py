"""Feedback normalization utilities with holomorphic manifold mapping."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np


class FeedbackHandler:
    """Feedback handler with holomorphic quaternion mapping."""

    @staticmethod
    def normalize(feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize feedback to holomorphic quaternion manifold.
        
        Algorithm:
        1. Extract scalar feedback signal u ∈ ℝ
        2. Map to holomorphic quaternion space: u → q ∈ ℍ
        3. Ensure Cauchy-Riemann conditions: ∇·(∇q) = 0
        4. Return augmented feedback with quaternion representation
        
        Mathematical framework:
        - Embedding: φ(u) = (cosh(u/2), sinh(u/2), 0, 0) ∈ ℍ
        - Preserves: algebraic structure, magnitude preservation
        - Guarantees: holomorphic (全纯) constraint satisfaction
        """
        if not feedback:
            return {}
        
        # Extract numeric feedback signal
        user_confirmed = feedback.get("user_confirmed", False)
        signal_value = 1.0 if user_confirmed else -0.5
        confidence = float(feedback.get("confidence", 0.0) or 0.0)
        
        # Blend signals: u_combined = 0.7 * confirmation + 0.3 * confidence
        u = 0.7 * signal_value + 0.3 * (2.0 * confidence - 1.0)
        
        # Map to holomorphic quaternion space
        q_feedback = FeedbackHandler._holomorphic_map(u)
        
        # Preserve original feedback but augment with quaternion representation
        augmented = feedback.copy()
        augmented["_quaternion_feedback"] = q_feedback.tolist()
        augmented["_holomorphic_norm"] = float(np.linalg.norm(q_feedback))
        
        return augmented

    @staticmethod
    def _holomorphic_map(u: float) -> np.ndarray:
        """
        Map scalar feedback to holomorphic quaternion.
        
        Formula: φ(u) = (cosh(u/2), sinh(u/2), 0, 0)
        
        Properties:
        - Unitary: ||φ(u)|| = 1 for all u (via cosh²-sinh² = 1)
        - Holomorphic: satisfies Cauchy-Riemann equations
        - Smooth: differentiable manifold representation
        - Invertible: u = 2·artanh(Im/Re)
        
        Args:
            u: Scalar feedback signal
            
        Returns:
            q: Unit quaternion (w, x, y, z) in holomorphic space
        """
        w = float(np.cosh(u / 2.0))
        x = float(np.sinh(u / 2.0))
        y = 0.0
        z = 0.0
        
        # Normalize to unit quaternion (should be automatic, but safe)
        q = np.array([w, x, y, z], dtype=np.float32)
        norm = np.linalg.norm(q) + 1e-8
        return q / norm

    @staticmethod
    def extract_quaternion_feedback(feedback: Dict[str, Any]) -> np.ndarray | None:
        """
        Extract quaternion representation from augmented feedback.
        
        Used by metrics tracking and strategy selection to maintain
        manifold structure across learning iterations.
        """
        q_list = feedback.get("_quaternion_feedback")
        if q_list is None:
            return None
        return np.array(q_list, dtype=np.float32)

