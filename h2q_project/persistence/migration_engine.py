"""Checkpoint migration helpers (placeholder)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class MigrationEngine:
    def export(self, state: Dict[str, Any], output: Path) -> None:
        _ = state, output

    def import_state(self, checkpoint: Path) -> Dict[str, Any]:
        _ = checkpoint
        return {}
