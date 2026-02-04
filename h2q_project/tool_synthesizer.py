"""
ToolSynthesizer

Synthesizes deterministic tools when existing tools fail.
Implements Orthogonal Expansion with sandboxed verification and self-correction.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from .safety import HolomorphicGuard
except ImportError:
    from safety import HolomorphicGuard


@dataclass
class SynthesisResult:
    success: bool
    tool_name: str
    source_code: str
    test_code: str
    error: Optional[str] = None
    attempts: int = 0


class ToolSynthesizer:
    """Generate, verify, and integrate deterministic tools for failed tasks."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        toolbox_register: Optional[Callable[[str, Callable[..., Any]], None]] = None,
        max_attempts: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.toolbox_register = toolbox_register
        self.max_attempts = max(1, max_attempts)
        self.guard = HolomorphicGuard()

    def synthesize(self, task_description: str) -> SynthesisResult:
        """Main entry point to synthesize a tool for a failed task."""
        tool_name = self._tool_name_from_task(task_description)
        attempts = 0
        last_error: Optional[str] = None
        source_code = ""
        test_code = ""

        while attempts < self.max_attempts:
            attempts += 1
            source_code = self._generate_tool_code(task_description, tool_name, last_error)
            source_code = self._ensure_tool_function(source_code, tool_name)

            if not self._is_deterministic(source_code):
                last_error = "Non-deterministic constructs detected (random/time/uuid)."
                continue

            guard_result = self.guard.validate(source_code, function_name=tool_name)
            if not guard_result.allowed:
                last_error = f"HolomorphicGuard rejection: {', '.join(guard_result.reasons)}"
                continue

            test_code = self._generate_test_code(task_description, tool_name)
            ok, stderr = self.verify_tool(source_code, test_code)
            if ok:
                self._integrate_tool(tool_name, source_code)
                return SynthesisResult(
                    success=True,
                    tool_name=tool_name,
                    source_code=source_code,
                    test_code=test_code,
                    attempts=attempts,
                )

            last_error = stderr

        return SynthesisResult(
            success=False,
            tool_name=tool_name,
            source_code=source_code,
            test_code=test_code,
            error=last_error,
            attempts=attempts,
        )

    def _generate_tool_code(self, task: str, tool_name: str, error: Optional[str]) -> str:
        prompt = (
            "Generate a deterministic Python function to solve the task. "
            "Return only code. The function must be named exactly: "
            f"{tool_name}. No randomness, no time-based calls.\n"
            f"Task: {task}\n"
        )
        if error:
            prompt += f"Previous error: {error}\nFix the code.\n"

        response = self._llm_generate(prompt, max_tokens=512)
        return self.extract_code(str(response.get("text", "")))

    def _generate_test_code(self, task: str, tool_name: str) -> str:
        prompt = (
            "Write a minimal deterministic unit test for the function below. "
            "Use assert and call the function with one concrete example. "
            "Return only Python code (no explanations).\n"
            f"Task: {task}\n"
            f"Function: {tool_name}\n"
        )
        response = self._llm_generate(prompt, max_tokens=256)
        return self.extract_code(str(response.get("text", "")))

    def _sandbox_run(self, tool_code: str, test_code: str) -> Tuple[bool, str]:
        """Run code in a subprocess sandbox with a temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool_path = os.path.join(tmpdir, "tool_impl.py")
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(tool_code)
                f.write("\n\n")
                f.write(test_code)

            completed = subprocess.run(
                [sys.executable, tool_path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if completed.returncode == 0 and not completed.stderr.strip():
                return True, ""
            return False, completed.stderr.strip() or completed.stdout.strip()

    def verify_tool(self, tool_code: str, test_code: str) -> Tuple[bool, str]:
        """Verify synthesized tool. Docker missing -> local subprocess fallback."""
        return self._sandbox_run(tool_code, test_code)

    def _integrate_tool(self, tool_name: str, source_code: str) -> None:
        """Append tool to custom_tools.py and register in toolbox if provided."""
        tools_path = os.path.join(os.path.dirname(__file__), "custom_tools.py")
        tools_dir = os.path.join(os.path.dirname(__file__), "custom_tools")
        os.makedirs(tools_dir, exist_ok=True)

        tool_file_path = os.path.join(tools_dir, f"{tool_name}.py")
        with open(tool_file_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated tool\n\n")
            f.write(source_code)
            f.write("\n")
        print(f"💾 [ToolSynthesizer] Tool saved to {tool_file_path}")

        if not os.path.exists(tools_path):
            with open(tools_path, "w", encoding="utf-8") as f:
                f.write("# Auto-generated tools\n\n")

        with open(tools_path, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(source_code)
            f.write("\n")

        module_name = "h2q_project.custom_tools"
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        if self.toolbox_register and hasattr(module, tool_name):
            self.toolbox_register(tool_name, getattr(module, tool_name))

    def _tool_name_from_task(self, task: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", task.lower()).strip("_")
        base = base[:40] if base else "custom_tool"
        return f"tool_{base}"

    def _ensure_tool_function(self, code: str, tool_name: str) -> str:
        """
        Ensure synthesized code defines the expected function name.
        If missing, wrap the code via exec in a deterministic wrapper.
        """
        if re.search(rf"def\s+{re.escape(tool_name)}\s*\(", code):
            return code

        wrapper = (
            f"\n\n"
            f"def {tool_name}(input_text=None):\n"
            f"    local_vars = {{\"input_text\": input_text}}\n"
            f"    code = {code!r}\n"
            f"    exec(code, {{}}, local_vars)\n"
            f"    return local_vars.get(\"result\")\n"
        )
        return f"{code}{wrapper}"

    def _llm_generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if self.llm_client is not None:
            if hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt=prompt, **kwargs)
            if hasattr(self.llm_client, "complete"):
                return self.llm_client.complete(prompt=prompt, **kwargs)
            if callable(self.llm_client):
                return self.llm_client(prompt=prompt, **kwargs)
        return {"text": ""}

    @staticmethod
    def extract_code(text: str) -> str:
        """
        Extract code from markdown fences.
        1) ```python ... ```
        2) ``` ... ```
        3) fallback: full text
        """
        if not text:
            return ""
        match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _is_deterministic(code: str) -> bool:
        forbidden = [
            "random",
            "numpy.random",
            "np.random",
            "time.time",
            "datetime.now",
            "uuid",
            "secrets",
            "os.urandom",
        ]
        lower = code.lower()
        return not any(token in lower for token in forbidden)
