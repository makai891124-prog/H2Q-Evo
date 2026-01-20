"""H2Q-Evo CLI Configuration"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CLIConfig:
    """CLI Configuration"""
    
    version: str = "2.3.0"
    home: Optional[Path] = None
    verbose: bool = False
    timeout: int = 30
    
    def __post_init__(self):
        if self.home is None:
            self.home = Path.home() / ".h2q-evo"


def get_config() -> CLIConfig:
    """Get CLI configuration"""
    return CLIConfig()
