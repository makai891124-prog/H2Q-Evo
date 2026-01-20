"""Local task execution and learning integration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .learning_loop import LearningLoop
from .strategy_manager import StrategyManager
from .feedback_handler import FeedbackHandler
from .knowledge.knowledge_db import KnowledgeDB


class LocalExecutor:
    """Lightweight local executor with learning hooks."""

    def __init__(self) -> None:
        self.learning_loop = LearningLoop()
        self.strategy_mgr = StrategyManager()
        self.feedback_handler = FeedbackHandler()
        self.knowledge_db: Optional[KnowledgeDB] = None
        
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
        try:
            from .h2q_server import inference_api  # type: ignore
            return str(inference_api(task))
        except Exception:
            return f"Processed: {task[:80]} (strategy={strategy})"

    @staticmethod
    def _compute_confidence(output: str, task_info: Dict[str, Any]) -> float:
        if not output:
            return 0.2
        if task_info.get("type") == "math":
            return 0.8
        return 0.6
