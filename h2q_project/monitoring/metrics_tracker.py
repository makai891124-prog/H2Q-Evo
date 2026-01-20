"""Simple metrics tracker for local agent runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class MetricsTracker:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home() / ".h2q-evo"
        self.metrics_file = self.home / "metrics.json"
        self.metrics = self._load()

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
        }

    def record_execution(self, task: str, result: Dict[str, Any]) -> None:
        confidence = float(result.get("confidence", 0.0) or 0.0)
        success = 1.0 if confidence >= 0.7 else 0.0

        self.metrics["total_tasks"] += 1
        self.metrics["success_rate"] = 0.95 * self.metrics["success_rate"] + 0.05 * success
        self.metrics["history"].append({"task": task[:80], "confidence": confidence})
        self._save()

    def get_current_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def _save(self) -> None:
        self.metrics_file.parent.mkdir(exist_ok=True)
        with self.metrics_file.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
