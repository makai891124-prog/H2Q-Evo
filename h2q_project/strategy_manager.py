"""Strategy selection and effectiveness tracking."""
from __future__ import annotations

from typing import Any, Dict


class StrategyManager:
    """Simple strategy registry and scoring."""

    def __init__(self) -> None:
        self.success_counts: Dict[str, int] = {"default": 0}

    def select_best(self, task_info: Dict[str, Any]) -> str:
        _ = task_info
        # Choose the strategy with the highest success count; fall back to default.
        if not self.success_counts:
            return "default"
        return max(self.success_counts.items(), key=lambda kv: kv[1])[0]

    def update_effectiveness(self, strategy: str, success: bool) -> None:
        if not strategy:
            strategy = "default"
        current = self.success_counts.get(strategy, 0)
        self.success_counts[strategy] = current + (1 if success else 0)
