"""CLI configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


def agent_home() -> Path:
    """Return the agent home directory path."""
    env_home = os.getenv("H2Q_AGENT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".h2q-evo"
