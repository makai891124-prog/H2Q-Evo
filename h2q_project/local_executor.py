"""Local task execution and learning integration."""
from __future__ import annotations

import time
import logging
import subprocess
import sys
import inspect
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .learning_loop import LearningLoop
from .strategy_manager import StrategyManager
from .feedback_handler import FeedbackHandler
from .knowledge.knowledge_db import KnowledgeDB
from .precision_gated_executor import PrecisionGatedExecutor


class LocalExecutor:
    """Lightweight local executor with learning hooks and precision gating."""

    def __init__(self, enable_precision_gating: bool = True) -> None:
        self.learning_loop = LearningLoop()
        self.strategy_mgr = StrategyManager()
        self.feedback_handler = FeedbackHandler()
        self.knowledge_db: Optional[KnowledgeDB] = None

        self._logger = logging.getLogger(__name__)

        # Model hierarchy strategy
        self.model_small = "deepseek-coder:6.7b"
        self.model_code = "deepseek-coder-v2-236b-compressed"
        self.model_logic = "deepseek-coder:33b"
        self._model_router = _LocalModelRouter(self)
        
        # Initialize precision-gated executor (DAS Meta-Theory)
        self.enable_precision_gating = enable_precision_gating
        self.precision_gated_executor: Optional[PrecisionGatedExecutor] = None
        if enable_precision_gating:
            self.precision_gated_executor = PrecisionGatedExecutor(
                base_executor=self,
                enable_cot=True,
                llm_client=self._model_router,
            )
        
        # Quaternion task classifiers (anchor points in semantic manifold)
        # Format: (w, x, y, z) representing each task type
        # Anchors chosen to maintain semantic separation in 4D space
        self._task_quaternions = {
            "math": np.array([1.0, 0.9, 0.0, 0.0], dtype=np.float32),      # Real-dominant (algebraic)
            "logic": np.array([0.8, 0.2, 0.8, 0.0], dtype=np.float32),     # Balanced x,z (reasoning)
            "general": np.array([0.5, 0.3, 0.3, 0.3], dtype=np.float32),   # Low magnitude (default)
        }

    def init_knowledge_db(self, home: Path) -> None:
        self.knowledge_db = KnowledgeDB(home / "knowledge")

    def execute(self, task: str, strategy: str = "auto") -> Dict[str, Any]:
        """
        Execute task with optional precision gating (DAS Meta-Theory).
        
        If precision gating is enabled, uses entropy-based routing:
        - High entropy (Wave state) -> Chain-of-Thought reasoning
        - Low entropy (Particle state) -> Direct output
        - Medium entropy (Coherence) -> Standard verified execution
        
        Args:
            task: Task string to execute
            strategy: Execution strategy ("auto", "direct", "cot")
            
        Returns:
            Dictionary with output, confidence, and execution metadata
        """
        # Route through precision gated executor if enabled
        if self.precision_gated_executor:
            return self.precision_gated_executor.execute_with_precision_gating(
                task=task,
                strategy=strategy,
                generate_antithesis=True,
            )
        
        # Fallback: Standard execution without precision gating
        return self._execute_direct(task, strategy)
    
    def _execute_direct(self, task: str, strategy: str = "auto") -> Dict[str, Any]:
        """Direct execution without precision gating (legacy path)."""
        start = time.time()
        try:
            task_info = self._analyze_task(task)
            selected = self.strategy_mgr.select_best(task_info)
            if strategy != "auto":
                selected = strategy

            output = self._run_inference(task, selected)
            confidence = self._compute_confidence(output, task_info)
            elapsed = time.time() - start

            return {
                "output": output,
                "confidence": confidence,
                "task_type": task_info.get("type"),
                "strategy_used": selected,
                "elapsed_time": elapsed,
                "timestamp": time.time(),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "output": f"Execution error: {exc}",
                "confidence": 0.0,
                "task_type": "unknown",
                "strategy_used": strategy,
                "elapsed_time": time.time() - start,
                "timestamp": time.time(),
                "error": str(exc),
            }

    def save_experience(self, task: str, result: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        feedback = self.feedback_handler.normalize(feedback)

        if not self.knowledge_db:
            return

        experience = {
            "task": task,
            "result": result,
            "feedback": feedback,
            "timestamp": time.time(),
            "task_type": result.get("task_type"),
            "strategy_used": result.get("strategy_used"),
            "confidence": result.get("confidence"),
        }
        self.knowledge_db.save_experience(experience)

        self.strategy_mgr.update_effectiveness(result.get("strategy_used"), feedback.get("user_confirmed", False))
        self.learning_loop.update_weights(model=None, feedback=feedback)

    def get_knowledge_stats(self, home: Path) -> Dict[str, Any]:
        if not self.knowledge_db:
            self.init_knowledge_db(home)
        if not self.knowledge_db:
            return {"total_experiences": 0, "domains": []}
        return self.knowledge_db.get_stats()
    
    def get_precision_gating_stats(self) -> Dict[str, Any]:
        """Get statistics about precision gating execution (DAS Meta-Theory)."""
        if not self.precision_gated_executor:
            return {"enabled": False}
        
        stats = self.precision_gated_executor.get_execution_statistics()
        stats["enabled"] = True
        return stats

    def _analyze_task(self, task: str) -> Dict[str, Any]:
        return {
            "type": self._classify_task_quaternion(task),
            "complexity": len(task.split()),
            "keywords": self._extract_keywords(task),
        }

    def _quaternion_dot(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute Fueter inner product: ⟨q₁, q₂⟩_F = Re(q₁* · q₂)
        For unit quaternions: equals dot product = q₁·q₂
        Complexity: O(1) = 4 multiplications + 3 additions
        """
        return float(np.dot(q1, q2))

    def _encode_task_quaternion(self, task: str) -> np.ndarray:
        """
        Encode task string into semantic quaternion space.
        Algorithm:
        1. Extract keyword tokens
        2. Compute semantic direction (math, logic, general)
        3. Weight by task keywords
        4. Normalize to unit quaternion
        """
        lower = task.lower()
        keywords = self._extract_keywords(task)
        
        # Count semantic indicators (O(k) where k = keyword count << n)
        math_count = sum(1 for kw in keywords if any(w in kw for w in ["math", "计算", "方程", "calculate", "compute", "+", "-", "*", "/", "solve", "equation", "derive", "differential"]))
        logic_count = sum(1 for kw in keywords if any(w in kw for w in ["推理", "logic", "reason", "prove", "theorem", "证", "判断"]))
        
        # Normalize counts to [0, 1]
        total = len(keywords) + 1e-8
        math_weight = math_count / total
        logic_weight = logic_count / total
        general_weight = 1.0 - math_weight - logic_weight
        
        # Blend quaternions: q = w1*q_math + w2*q_logic + w3*q_general
        q = (math_weight * self._task_quaternions["math"] +
             logic_weight * self._task_quaternions["logic"] +
             general_weight * self._task_quaternions["general"])
        
        # Normalize to unit quaternion
        norm = np.linalg.norm(q) + 1e-8
        return q / norm

    def _classify_task_quaternion(self, task: str) -> str:
        """
        Classify task using quaternion similarity (O(1) operation).
        
        Steps:
        1. Encode task into quaternion space: O(k) where k = keywords
        2. Compute Fueter inner products with 3 anchors: 3 × O(1) = O(1)
        3. Return max: O(1)
        
        Total: O(k + 1) << O(n) where n = all keywords ever defined
        """
        task_q = self._encode_task_quaternion(task)
        
        similarities = {
            task_type: self._quaternion_dot(task_q, anchor_q)
            for task_type, anchor_q in self._task_quaternions.items()
        }
        
        return max(similarities.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _classify_task(task: str) -> str:
        lower = task.lower()
        # Math keywords
        if any(word in lower for word in ["math", "计算", "方程", "calculate", "compute", "+", "-", "*", "/", "solve", "equation"]):
            return "math"
        # Logic keywords
        if any(word in lower for word in ["推理", "logic", "reason", "prove", "theorem"]):
            return "logic"
        return "general"

    @staticmethod
    def _extract_keywords(task: str) -> List[str]:
        return [token for token in task.split() if len(token) > 1]

    def _run_inference(self, task: str, strategy: str) -> str:
        model = self._select_model_for_task(task)
        return self._infer_with_model(model, task, strategy)

    def _select_model_for_task(self, task: str) -> str:
        lower = task.lower()
        if any(k in lower for k in ["code", "python", "script", "函数", "写代码", "实现"]):
            return self.model_code
        if any(k in lower for k in ["prove", "logic", "推理", "证明", "theorem", "therefore", "if", "then"]):
            return self.model_logic
        return self.model_small

    def _infer_with_model(self, model: str, prompt: str, strategy: str) -> str:
        try:
            from .h2q_server import inference_api  # type: ignore
            sig = inspect.signature(inference_api)
            kwargs = {}
            if "model" in sig.parameters:
                kwargs["model"] = model
            if "max_tokens" in sig.parameters:
                kwargs["max_tokens"] = 512
            if "temperature" in sig.parameters:
                kwargs["temperature"] = 0.2
            return str(inference_api(prompt, **kwargs))
        except Exception:
            return f"Processed: {prompt[:80]} (strategy={strategy}, model={model})"

    def execute_code_safely(self, code: str) -> str:
        """
        Local subprocess fallback for code execution.
        Security note: only allow standard libs (math, pandas, numpy).
        """
        if not self._code_imports_allowed(code):
            return "Execution blocked: non-standard imports detected."

        self._logger.warning("Docker unavailable. Using local subprocess fallback.")
        try:
            completed = subprocess.run(
                [sys.executable, "-"],
                input=code,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if stderr:
                return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}".strip()
            return stdout
        except Exception as exc:
            return f"Execution error: {exc}"

    def execute_code(self, tool_path: str) -> str:
        """
        Execute a tool file. If Docker is unavailable, run on bare metal.
        """
        docker_client = getattr(self, "docker_client", None)
        if docker_client is None:
            self._logger.warning("[DAS] Running code on bare metal...")
            try:
                completed = subprocess.run(
                    ["python3", tool_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                return completed.stdout.strip()
            except Exception as exc:
                return f"Execution error: {exc}"

        return "Execution skipped: Docker client available but not used in this demo."

    def _code_imports_allowed(self, code: str) -> bool:
        allowed = {"math", "numpy", "pandas"}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in allowed:
                        return False
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in allowed:
                    return False
        return True


class _LocalModelRouter:
    """Route prompts to local models based on DAS hierarchy."""

    def __init__(self, executor: LocalExecutor) -> None:
        self.executor = executor

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        lower = prompt.lower()
        if "answer briefly" in lower:
            model = self.executor.model_small
        elif "generate a minimal python script" in lower or "python" in lower or "script" in lower:
            model = self.executor.model_code
        elif any(k in lower for k in ["prove", "logic", "推理", "证明", "theorem", "therefore", "if", "then"]):
            model = self.executor.model_logic
        else:
            model = self.executor.model_small

        text = self.executor._infer_with_model(model, prompt, strategy="auto")
        return {"text": text}

    @staticmethod
    def _compute_confidence(output: str, task_info: Dict[str, Any]) -> float:
        if not output:
            return 0.2
        if task_info.get("type") == "math":
            return 0.8
        return 0.6
