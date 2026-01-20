"""Integrity checks for persisted artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


class IntegrityChecker:
    @staticmethod
    def checksum(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha.update(chunk)
        return sha.hexdigest()
