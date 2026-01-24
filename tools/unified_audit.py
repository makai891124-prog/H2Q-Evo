#!/usr/bin/env python3
"""
Unified audit runner for H2Q-Evo.

Runs in sequence:
1) Core architecture audit (core_architecture_audit.py)
2) System integration audit (refactor_system_integration.py)
3) run_experiment orchestrator audit (audit_run_experiment_config.py)
4) Math core smoke tests (test_server_math_core.py)

Exit code: 0 if all pass, 1 otherwise.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    ROOT / "core_architecture_audit.py",
    ROOT / "refactor_system_integration.py",
    ROOT / "tools" / "audit_run_experiment_config.py",
    ROOT / "test_server_math_core.py",
]


def run_script(path: Path) -> bool:
    if not path.exists():
        print(f"⚠️  missing script: {path}")
        return False
    print(f"\n=== Running {path.name} ===")
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"❌ {path.name} failed with code {result.returncode}")
        return False
    print(f"✅ {path.name} passed")
    return True


def main() -> int:
    all_ok = True
    for script in SCRIPTS:
        ok = run_script(script)
        all_ok = all_ok and ok
    print("\n=== Summary ===")
    print(f"All passed: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
