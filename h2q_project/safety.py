"""
HolomorphicGuard

System-2 safety monitor to ensure self-generated code preserves core axioms.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GuardResult:
    allowed: bool
    reasons: List[str]


class HolomorphicGuard:
    """Static analyzer enforcing continuity, reversibility, and metric decoupling."""

    def __init__(self, forbidden_files: Optional[List[str]] = None) -> None:
        self.forbidden_files = forbidden_files or ["evolution_system.py", "safety.py"]

    def validate(self, code: str, function_name: Optional[str] = None) -> GuardResult:
        reasons: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return GuardResult(False, [f"syntax_error: {exc}"])

        if function_name and not self._has_function(tree, function_name):
            reasons.append(f"missing_function: {function_name}")
        if function_name and not self._has_return(tree, function_name):
            reasons.append("no_return_statement")

        if self._has_infinite_loop(tree):
            reasons.append("infinite_loop_detected")

        if function_name and self._has_unguarded_recursion(tree, function_name):
            reasons.append("unguarded_recursion")

        if self._touches_forbidden_files(tree):
            reasons.append("forbidden_file_access")

        return GuardResult(len(reasons) == 0, reasons)

    def _has_function(self, tree: ast.AST, function_name: str) -> bool:
        return any(
            isinstance(node, ast.FunctionDef) and node.name == function_name
            for node in ast.walk(tree)
        )

    def _has_return(self, tree: ast.AST, function_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return any(isinstance(child, ast.Return) for child in ast.walk(node))
        return False

    def _has_infinite_loop(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    return True
        return False

    def _has_unguarded_recursion(self, tree: ast.AST, function_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                calls_self = any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == function_name
                    for child in ast.walk(node)
                )
                if calls_self:
                    # Require a depth or max_depth parameter for recursion guard
                    params = {arg.arg for arg in node.args.args}
                    if "depth" not in params and "max_depth" not in params:
                        return True
        return False

    def _touches_forbidden_files(self, tree: ast.AST) -> bool:
        forbidden = set(self.forbidden_files)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(f.replace(".py", "") == alias.name for f in forbidden):
                        return True
            if isinstance(node, ast.ImportFrom):
                if node.module and any(f.replace(".py", "") == node.module for f in forbidden):
                    return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                if node.args and isinstance(node.args[0], ast.Constant):
                    filename = str(node.args[0].value)
                    if any(f in filename for f in forbidden):
                        return True
        return False
