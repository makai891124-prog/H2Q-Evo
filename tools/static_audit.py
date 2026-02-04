"""
Static audit for H2Q-Evo integrity checks.

Checks:
1) PrecisionGatedExecutor: external execution, dualistic verification, branching logic
2) ToolSynthesizer: file I/O, error feedback loop, dynamic generation
3) FractalMemory: quaternion ops usage, quaternion similarity
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AuditResult:
    name: str
    passed: bool
    reasons: List[str]


def read_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def has_import(tree: ast.AST, module_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                return True
    return False


def has_call_name(tree: ast.AST, func_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == func_name:
                return True
    return False


def has_call_attr(tree: ast.AST, attr_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == attr_name:
                return True
    return False


def has_try_block(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.Try) for node in ast.walk(tree))


def has_if_in_function(tree: ast.AST, func_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return any(isinstance(child, ast.If) for child in ast.walk(node))
    return False


def has_function(tree: ast.AST, func_name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == func_name for node in ast.walk(tree))


def has_write_file_io(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
                if "w" in mode or "a" in mode:
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "write":
                return True
    return False


def has_dualistic_comparison(tree: ast.AST) -> bool:
    # Heuristic: presence of verify_duality or _detect_contradictions
    return has_function(tree, "verify_duality") or has_call_attr(tree, "_detect_contradictions")


def has_dynamic_generation(tree: ast.AST) -> bool:
    # Heuristic: _generate_tool_code or _generate_test_code function present
    return has_function(tree, "_generate_tool_code") and has_function(tree, "_generate_test_code")


def audit_precision_gated_executor(path: Path) -> AuditResult:
    tree = read_ast(path)
    reasons: List[str] = []

    external_exec = has_import(tree, "subprocess") or has_import(tree, "docker")
    if not external_exec:
        reasons.append("missing external execution import (subprocess/docker)")

    if not has_dualistic_comparison(tree):
        reasons.append("dualistic verification logic not found")

    if not has_if_in_function(tree, "execute_with_precision_gating"):
        reasons.append("no branching in execute_with_precision_gating")

    passed = len(reasons) == 0
    return AuditResult("PrecisionGatedExecutor", passed, reasons)


def audit_tool_synthesizer(path: Path) -> AuditResult:
    tree = read_ast(path)
    reasons: List[str] = []

    if not has_write_file_io(tree):
        reasons.append("no file I/O for saving tools")

    if not has_try_block(tree):
        reasons.append("no try/except feedback loop detected")

    if not has_dynamic_generation(tree):
        reasons.append("no dynamic generation functions found")

    passed = len(reasons) == 0
    return AuditResult("ToolSynthesizer", passed, reasons)


def audit_fractal_memory(path: Path) -> AuditResult:
    tree = read_ast(path)
    reasons: List[str] = []

    if not has_import(tree, "h2q.quaternion_ops"):
        reasons.append("does not import h2q.quaternion_ops")

    if not has_call_name(tree, "quaternion_dot_product") and not has_call_attr(tree, "quaternion_dot_product"):
        reasons.append("no quaternion_dot_product usage in retrieval")

    if has_call_name(tree, "cosine_similarity") or has_call_attr(tree, "cosine_similarity"):
        reasons.append("uses cosine similarity (non-quaternion)")

    passed = len(reasons) == 0
    return AuditResult("FractalMemory", passed, reasons)


def print_report(results: List[AuditResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{result.name}] {status}")
        if result.reasons:
            for reason in result.reasons:
                print(f"  - {reason}")


def main() -> None:
    precision_path = ROOT / "h2q_project" / "precision_gated_executor.py"
    tool_path = ROOT / "h2q_project" / "tool_synthesizer.py"
    memory_path = ROOT / "h2q_project" / "fractal_memory.py"

    results = [
        audit_precision_gated_executor(precision_path),
        audit_tool_synthesizer(tool_path),
        audit_fractal_memory(memory_path),
    ]
    print_report(results)


if __name__ == "__main__":
    main()
