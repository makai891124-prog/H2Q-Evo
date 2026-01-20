"""Simple metrics tracker for local agent runs with quaternion SLERP interpolation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


class MetricsTracker:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home() / ".h2q-evo"
        self.metrics_file = self.home / "metrics.json"
        self.metrics = self._load()
        
        # Quaternion success trajectory (for SLERP interpolation)
        # Represents learning manifold: successful operations as points on unit sphere
        self._q_success_trajectory = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _load(self) -> Dict[str, Any]:
        if self.metrics_file.exists():
            try:
                with self.metrics_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return self._defaults()
        return self._defaults()

    def _defaults(self) -> Dict[str, Any]:
        return {
            "total_tasks": 0,
            "success_rate": 0.0,
            "history": [],
            "_quaternion_success_rate": [1.0, 0.0, 0.0, 0.0],  # Unit quaternion representation
        }

    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Spherical Linear Interpolation (SLERP) for quaternions.
        
        Formula: Slerp(q₁, q₂, t) = [sin((1-t)θ)/sin(θ)]q₁ + [sin(tθ)/sin(θ)]q₂
        where θ = arccos(⟨q₁, q₂⟩)
        
        Benefits:
        - Preserves unit quaternion constraint (flow on S³)
        - Maintains geometric meaning of learning trajectory
        - Constant angular velocity interpolation
        
        Args:
            q1: Start quaternion (unit)
            q2: End quaternion (unit)
            alpha: Interpolation parameter ∈ [0,1]
            
        Returns:
            Interpolated quaternion on manifold
        """
        # Compute quaternion dot product
        dot_product = float(np.dot(q1, q2))
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # If quaternions are very close, use linear interpolation
        if abs(dot_product) > 0.9995:
            result = (1.0 - alpha) * q1 + alpha * q2
            norm = np.linalg.norm(result) + 1e-8
            return result / norm
        
        # Compute angle between quaternions
        theta = np.arccos(dot_product)
        sin_theta = np.sin(theta)
        
        # SLERP formula
        w1 = np.sin((1.0 - alpha) * theta) / sin_theta
        w2 = np.sin(alpha * theta) / sin_theta
        
        result = w1 * q1 + w2 * q2
        norm = np.linalg.norm(result) + 1e-8
        return result / norm

    @staticmethod
    def _success_to_quaternion(success: float) -> np.ndarray:
        """
        Map scalar success [0,1] to quaternion on S³.
        
        Mapping: s ∈ [0,1] → q = (cos(πs/2), sin(πs/2), 0, 0) ∈ S³
        
        Properties:
        - Smooth manifold embedding
        - s=0 → q=(1,0,0,0) (failure axis)
        - s=1 → q=(0,1,0,0) (success axis)
        - Unit norm: ||q|| = 1
        """
        angle = np.pi * success / 2.0
        w = np.cos(angle)
        x = np.sin(angle)
        return np.array([w, x, 0.0, 0.0], dtype=np.float32)

    def record_execution(self, task: str, result: Dict[str, Any]) -> None:
        """
        Record execution with quaternion SLERP trajectory tracking.
        
        Algorithm:
        1. Extract success signal: s = 1.0 if confidence ≥ 0.7 else 0.0
        2. Map to quaternion: q_success = φ(s)
        3. SLERP blend with historical trajectory: q_t = Slerp(q_{t-1}, q_success, α)
        4. Store both scalar and quaternion metrics
        """
        confidence = float(result.get("confidence", 0.0) or 0.0)
        success = 1.0 if confidence >= 0.7 else 0.0

        # Update scalar metrics (backward compatible)
        self.metrics["total_tasks"] += 1
        self.metrics["success_rate"] = 0.95 * self.metrics["success_rate"] + 0.05 * success
        
        # Update quaternion trajectory via SLERP
        q_success = self._success_to_quaternion(success)
        q_old = np.array(self.metrics.get("_quaternion_success_rate", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        # SLERP with α = 0.05 (matching scalar EMA parameter)
        q_new = self._slerp(q_old, q_success, alpha=0.05)
        self.metrics["_quaternion_success_rate"] = q_new.tolist()
        
        # Store history
        self.metrics["history"].append({"task": task[:80], "confidence": confidence})
        self._save()

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get metrics including both scalar and quaternion representations."""
        result = self.metrics.copy()
        
        # Add computed quaternion norm (manifold stability indicator)
        q = np.array(result.get("_quaternion_success_rate", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        result["_quaternion_manifold_norm"] = float(np.linalg.norm(q))
        
        return result

    def _save(self) -> None:
        self.metrics_file.parent.mkdir(exist_ok=True)
        with self.metrics_file.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

