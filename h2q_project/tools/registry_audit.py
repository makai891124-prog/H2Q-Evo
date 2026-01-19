import ast
import os
from pathlib import Path
from typing import List, Dict, Any

class UnifiedSymmetryValidator:
    """
    H2Q Static Analysis Tool: Enforces signature symmetry for DiscreteDecisionEngine (DDE).
    Identifies 'Topological Tears' where instantiations mismatch the canonical LatentConfig signature.
    """

    CANONICAL_SIGNATURE = ["config"]
    FORBIDDEN_KEYWORDS = ["dim", "latent_dim", "input_dim", "state_dim"]

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.violations = []

    def audit_project(self):
        """Walks through the project and inspects all Python files."""
        py_files = list(self.project_root.rglob("*.py"))
        print(f"[M24-CW] Initiating Symmetry Audit across {len(py_files)} nodes...")

        for file_path in py_files:
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue
            self._inspect_file(file_path)

        self._report_findings()

    def _inspect_file(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"[!] Failed to parse {file_path}: {e}")
            return

        for node in ast.walk(tree):
            # 1. Check Class Definitions (Local Symmetry)
            if isinstance(node, ast.ClassDef) and node.name == "DiscreteDecisionEngine":
                self._validate_class_definition(node, file_path)

            # 2. Check Instantiations (Global Symmetry)
            if isinstance(node, ast.Call):
                if self._is_dde_call(node):
                    self._validate_instantiation(node, file_path)

    def _is_dde_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            return node.func.id == "DiscreteDecisionEngine"
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "DiscreteDecisionEngine"
        return False

    def _validate_class_definition(self, node: ast.ClassDef, file_path: Path):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                args = [arg.arg for arg in item.args.args if arg.arg != "self"]
                if args != self.CANONICAL_SIGNATURE:
                    self.violations.append({
                        "type": "LEGACY_DEFINITION",
                        "file": str(file_path),
                        "line": item.lineno,
                        "details": f"Found non-canonical signature: {args}. Expected: {self.CANONICAL_SIGNATURE}"
                    })

    def _validate_instantiation(self, node: ast.Call, file_path: Path):
        # Check for forbidden keyword arguments (e.g., dim=...)
        for kw in node.keywords:
            if kw.arg in self.FORBIDDEN_KEYWORDS:
                self.violations.append({
                    "type": "SIGNATURE_MISMATCH",
                    "file": str(file_path),
                    "line": node.lineno,
                    "details": f"Unexpected keyword argument '{kw.arg}'. DDE now requires a 'config' object."
                })

        # Check positional argument count
        if len(node.args) > 1:
            self.violations.append({
                "type": "POSITIONAL_OVERFLOW",
                "file": str(file_path),
                "line": node.lineno,
                "details": f"Too many positional arguments ({len(node.args)}). Expected 1 (config)."
            })

    def _report_findings(self):
        if not self.violations:
            print("\n[✓] Symmetry Validated: All DiscreteDecisionEngine instances are holomorphic.")
            return

        print(f"\n[!] Found {len(self.violations)} Symmetry Violations (Topological Tears):\n")
        for v in self.violations:
            print(f"[{v['type']}] {v['file']}:{v['line']}")
            print(f"    -> {v['details']}\n")

if __name__ == "__main__":
    # Grounding in Reality: Execute from project root
    validator = UnifiedSymmetryValidator(os.getcwd())
    validator.audit_project()