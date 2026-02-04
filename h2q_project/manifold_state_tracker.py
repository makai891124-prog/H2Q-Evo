"""
Manifold State Tracker

Tracks "Thought State" as a quaternion on the SU(2) hypersphere.
Maps task complexity to rotation magnitude and provides a path curve
for visualization or analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

try:
    from .quaternion_ops import (
        quaternion_slerp,
        quaternion_normalize,
        quaternion_multiply,
    )
except ImportError:
    from quaternion_ops import (
        quaternion_slerp,
        quaternion_normalize,
        quaternion_multiply,
    )


@dataclass
class ManifoldState:
    """Represents a single state on the quaternion manifold."""
    task: str
    complexity: float
    quaternion: np.ndarray


class ManifoldStateTracker:
    """Map task complexity to quaternion magnitude and track state evolution."""

    def __init__(self, max_complexity: float = 40.0):
        self.max_complexity = max(1.0, max_complexity)
        self.identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._history: List[ManifoldState] = []

    def analyze_complexity(self, task: str) -> float:
        """Estimate task complexity using token length and heuristic weights."""
        tokens = [t for t in task.split() if t.strip()]
        length_score = len(tokens)
        symbol_weight = sum(1 for ch in task if ch in "+-*/=^") * 0.5
        logical_weight = sum(1 for kw in ["prove", "therefore", "if", "then", "推理", "证明"] if kw in task.lower()) * 2.0
        return float(length_score + symbol_weight + logical_weight)

    def complexity_to_quaternion(self, complexity: float) -> np.ndarray:
        """
        Map complexity to a rotation angle.
        Simple task -> identity quaternion (no rotation).
        Complex task -> larger rotation around a stable axis.
        """
        normalized = min(max(complexity / self.max_complexity, 0.0), 1.0)
        angle = normalized * np.pi  # up to 180 degrees
        axis = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)
        return quaternion_normalize(np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64))

    def update_state(self, task: str) -> ManifoldState:
        """Compute quaternion state for a task and store it in history."""
        complexity = self.analyze_complexity(task)
        q = self.complexity_to_quaternion(complexity)
        state = ManifoldState(task=task, complexity=complexity, quaternion=q)
        self._history.append(state)
        return state

    def get_path(self, samples: int = 20) -> List[np.ndarray]:
        """Return a smooth path between historical states using SLERP."""
        if len(self._history) < 2:
            return [self.identity.copy()]

        path: List[np.ndarray] = []
        for i in range(len(self._history) - 1):
            q1 = self._history[i].quaternion
            q2 = self._history[i + 1].quaternion
            for t in np.linspace(0.0, 1.0, max(2, samples)):
                path.append(quaternion_slerp(q1, q2, float(t)))
        return path

    def evolve_state(self, task: str, steps: int = 10) -> List[np.ndarray]:
        """Evolve from identity toward the task quaternion state."""
        target = self.update_state(task).quaternion
        return [quaternion_slerp(self.identity, target, i / max(1, steps)) for i in range(steps + 1)]

    def combine_states(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Combine two states via Hamilton product to represent evolution."""
        return quaternion_normalize(quaternion_multiply(q1, q2))

    def get_history(self) -> List[ManifoldState]:
        return list(self._history)

    def get_history_summary(self) -> Dict[str, Any]:
        return {
            "count": len(self._history),
            "max_complexity": self.max_complexity,
            "states": [
                {
                    "task": s.task,
                    "complexity": s.complexity,
                    "quaternion": s.quaternion.tolist(),
                }
                for s in self._history
            ],
        }
