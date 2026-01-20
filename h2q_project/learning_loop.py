"""Local learning loop (lightweight placeholder with metrics)."""
from __future__ import annotations

from typing import Any, Dict


class LearningLoop:
    """Lightweight learning loop that tracks feedback signals.

    This does not mutate model weights by default to avoid unintended side-effects
    in environments where the core model is not loaded. It tracks feedback signals
    and update counts so future weight updates can be added safely.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.update_count = 0
        self.positive_feedback = 0
        self.negative_feedback = 0

    def update_weights(self, model: Any, feedback: Dict[str, Any]) -> None:
        """Record feedback; hook for future weight updates."""
        signal = self._feedback_signal(feedback)

        if signal > 0:
            self.positive_feedback += 1
        elif signal < 0:
            self.negative_feedback += 1

        # Placeholder: no-op weight change to avoid altering core model
        _ = model
        self.update_count += 1

    @staticmethod
    def _feedback_signal(feedback: Dict[str, Any]) -> float:
        if feedback.get("user_confirmed"):
            return 1.0
        if feedback.get("success"):
            return 0.8
        if feedback.get("partial_success"):
            return 0.2
        if feedback.get("error"):
            return -0.5
        return 0.0
