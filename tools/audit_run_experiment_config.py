#!/usr/bin/env python3
"""Audit that run_experiment.py uses the unified orchestrator with the expected config."""
import ast
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "h2q_project" / "run_experiment.py"
EXPECTED_CONFIG: Dict[str, Any] = {
    "memory_threshold_gb": 14.0,
    "ssd_cache_path": "./vault/ssd_paging",
}


def load_tree(path: Path) -> ast.AST:
    try:
        return ast.parse(path.read_text())
    except FileNotFoundError:
        print(f"❌ Missing run_experiment.py at {path}")
        sys.exit(1)
    except SyntaxError as exc:
        print(f"❌ Syntax error while parsing {path}: {exc}")
        sys.exit(1)


def has_orchestrator_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "h2q.core.unified_orchestrator":
            imported = {alias.name for alias in node.names}
            if "get_orchestrator" in imported and "Unified_Homeostatic_Orchestrator" in imported:
                return True
    return False


def extract_config(tree: ast.AST):
    """Return ORCHESTRATOR_CONFIG dict value from Assign or AnnAssign nodes."""

    def parse_dict(value: ast.AST):
        if not isinstance(value, ast.Dict):
            return None
        cfg = {}
        for key, val in zip(value.keys, value.values):
            if isinstance(key, ast.Constant):
                cfg[key.value] = val.value if isinstance(val, ast.Constant) else None
        return cfg

    for node in ast.walk(tree):
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                targets = [node.target.id]
            value = node.value

        if "ORCHESTRATOR_CONFIG" in targets:
            return parse_dict(value)

    return None


def has_orchestrator_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "get_orchestrator":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get_orchestrator":
                return True
    return False


def main() -> int:
    tree = load_tree(TARGET)
    errors = []

    if not has_orchestrator_import(tree):
        errors.append("Missing 'get_orchestrator' and 'Unified_Homeostatic_Orchestrator' import")

    cfg = extract_config(tree)
    if cfg is None:
        errors.append("ORCHESTRATOR_CONFIG not defined")
    else:
        for key, expected in EXPECTED_CONFIG.items():
            if cfg.get(key) != expected:
                errors.append(f"ORCHESTRATOR_CONFIG[{key!r}] expected {expected}, found {cfg.get(key)}")

    if not has_orchestrator_call(tree):
        errors.append("get_orchestrator is never called")

    if errors:
        print("❌ run_experiment orchestrator audit failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("✅ run_experiment orchestrator audit passed")
    print(f"Validated config: {EXPECTED_CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
