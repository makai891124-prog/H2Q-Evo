"""Minimal smoke test for the h2q CLI flow.

Usage:
    PYTHONPATH=. python -m h2q_project.tools.smoke_cli

This will run: init -> execute -> status -> export-checkpoint.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return proc.stdout


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cli = [sys.executable, "-m", "h2q_project.h2q_cli.main"]

    print("[1/4] h2q init")
    print(run(cli + ["init"]))

    print("[2/4] h2q execute")
    print(run(cli + ["execute", "1+1", "--save-knowledge"]))

    print("[3/4] h2q status")
    print(run(cli + ["status"]))

    checkpoint = root / "temp_sandbox" / "smoke_checkpoint.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print("[4/4] h2q export-checkpoint")
    print(run(cli + ["export-checkpoint", str(checkpoint)]))

    print(f"Checkpoint saved at: {checkpoint}")


if __name__ == "__main__":
    main()
