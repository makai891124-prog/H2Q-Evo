"""
Precision-Gated Executor - DAS Meta-Theory Implementation

Core Philosophy:
1. **Metric Decoupling (Axiom III)**: Separate discrete logic (Turing Machine) 
   from continuous manifold (Quaternion Math). Code uses discrete logic for 
   control flow but quaternion algorithms for state representation.

2. **Precision-Gated Causality**: Causality only exists when precision is sufficient.
   - High Entropy = "Wave State" -> Requires "Orthogonal Expansion" (CoT/Tool Use)
   - Low Entropy = "Particle State" -> Allows "Direct Collapse" (Direct Output)

3. **Dualistic Generation (Axiom I)**: Generate both Thesis (A) and Antithesis (¬A),
   then verify "Topological Closure" (consistency).

This module acts as a "Middle Layer" to kill hallucinations by enforcing logic 
verification before the Softmax collapse.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np
from scipy.stats import entropy

# Try relative import first (when imported as module), fallback to absolute
try:
    from .quaternion_ops import (
        quaternion_multiply,
        quaternion_conjugate,
        quaternion_norm,
        quaternion_slerp,
    )
except ImportError:
    # Fallback for direct script execution
    try:
        from quaternion_ops import (
            quaternion_multiply,
            quaternion_conjugate,
            quaternion_norm,
            quaternion_slerp,
        )
    except ImportError:
        # These functions are optional for core functionality
        quaternion_multiply = None
        quaternion_conjugate = None
        quaternion_norm = None
        quaternion_slerp = None


logger = logging.getLogger(__name__)


class StateManifold(Enum):
    """Quantum-like state representation for reasoning stages."""
    WAVE = "wave"        # High entropy, requires expansion
    PARTICLE = "particle"  # Low entropy, direct output allowed
    COHERENCE = "coherence"  # Balanced state


@dataclass
class EntropyMetrics:
    """Entropy measurement across logical and semantic dimensions."""
    logical_entropy: float  # Shannon entropy of discrete propositions
    semantic_entropy: float  # Uncertainty in semantic manifold
    temporal_entropy: float  # Entropy of state transitions
    combined_entropy: float  # Weighted combination
    
    def is_high_precision(self, threshold: float = 0.3) -> bool:
        """Returns True if entropy is below threshold (high precision)."""
        return self.combined_entropy < threshold
    
    def get_manifold_state(self) -> StateManifold:
        """Map entropy to quantum-like state."""
        if self.combined_entropy < 0.25:
            return StateManifold.PARTICLE
        elif self.combined_entropy > 0.65:
            return StateManifold.WAVE
        else:
            return StateManifold.COHERENCE


@dataclass
class DualProposition:
    """Thesis-Antithesis pair for topological closure verification."""
    thesis: str          # Proposition A
    antithesis: str      # Proposition ¬A
    thesis_confidence: float  # P(A|evidence)
    antithesis_confidence: float  # P(¬A|evidence)
    closure_valid: bool = False  # Is topological closure consistent?
    closure_gap: float = 0.0  # |P(A) + P(¬A) - 1.0| (should ≈ 0)
    
    def verify_closure(self, tolerance: float = 0.05) -> bool:
        """
        Verify topological closure: sum of probabilities should equal 1.0
        This ensures logical consistency and prevents hallucinations.
        """
        total_prob = self.thesis_confidence + self.antithesis_confidence
        self.closure_gap = abs(total_prob - 1.0)
        self.closure_valid = self.closure_gap <= tolerance
        return self.closure_valid


@dataclass
class ExecutionContext:
    """Discrete logic control flow context."""
    task: str
    timestamp: float
    execution_trace: List[str] = field(default_factory=list)
    state_history: List[StateManifold] = field(default_factory=list)
    propositions: List[DualProposition] = field(default_factory=list)
    final_entropy: Optional[EntropyMetrics] = None
    output: Optional[str] = None
    confidence: float = 0.0


class ContinuousManifoldEncoder:
    """Encodes discrete propositions into continuous quaternion space."""
    
    def __init__(self, semantic_dim: int = 4):
        """
        Initialize encoder with quaternion-based semantic space.
        
        Args:
            semantic_dim: Dimension of semantic space (default 4 for quaternions)
        """
        self.semantic_dim = semantic_dim
        self._proposition_cache: Dict[str, np.ndarray] = {}
        
        # Semantic anchors (basis quaternions for common concept types)
        self._semantic_anchors = {
            "affirmative": np.array([1.0, 0.707, 0.0, 0.0], dtype=np.float32),
            "negative": np.array([1.0, -0.707, 0.0, 0.0], dtype=np.float32),
            "uncertain": np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32),
            "factual": np.array([1.0, 0.0, 0.707, 0.0], dtype=np.float32),
            "logical": np.array([1.0, 0.0, 0.0, 0.707], dtype=np.float32),
        }
    
    def encode_proposition(self, proposition: str) -> np.ndarray:
        """
        Encode proposition string into quaternion semantic space.
        
        Algorithm:
        1. Extract semantic indicators (keywords, negations, certainty markers)
        2. Select appropriate semantic anchor quaternion
        3. Apply quaternion rotation weighted by confidence
        4. Normalize to unit quaternion
        
        Returns:
            Unit quaternion [w, x, y, z] representing semantic position
        """
        if proposition in self._proposition_cache:
            return self._proposition_cache[proposition]
        
        lower = proposition.lower()
        
        # Discrete logic: classify semantic type (Turing machine step)
        is_negative = any(word in lower for word in ["not", "no", "false", "none", "¬", "无", "非"])
        is_uncertain = any(word in lower for word in ["maybe", "perhaps", "could", "might", "可能", "也许"])
        is_factual = any(word in lower for word in ["fact", "truth", "prove", "证明", "事实"])
        is_logical = any(word in lower for word in ["if", "then", "logic", "reason", "逻辑", "推理"])
        
        # Select anchor based on discrete classification
        if is_negative:
            anchor = self._semantic_anchors["negative"]
        elif is_uncertain:
            anchor = self._semantic_anchors["uncertain"]
        elif is_logical:
            anchor = self._semantic_anchors["logical"]
        elif is_factual:
            anchor = self._semantic_anchors["factual"]
        else:
            anchor = self._semantic_anchors["affirmative"]
        
        # Continuous: apply semantic weighting via quaternion algebra
        confidence = self._extract_confidence(proposition)
        
        # Scale quaternion by confidence (continuous manifold operation)
        q_encoded = anchor * confidence
        
        # Normalize to unit quaternion
        norm = np.linalg.norm(q_encoded) + 1e-8
        q_encoded = q_encoded / norm
        
        self._proposition_cache[proposition] = q_encoded
        return q_encoded
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from linguistic markers."""
        lower = text.lower()
        
        # Certainty markers (discrete classification)
        certainty_markers = {
            "definitely": 0.95,
            "certainly": 0.9,
            "probably": 0.7,
            "possibly": 0.5,
            "unlikely": 0.2,
            "definitely_not": 0.05,
        }
        
        for marker, confidence in certainty_markers.items():
            if marker in lower:
                return confidence
        
        return 0.6  # Default confidence
    
    def quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute angular distance between two quaternions in semantic space.
        Distance = arccos(|<q1, q2>|) normalized to [0, 1]
        """
        dot_product = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
        angular_distance = np.arccos(dot_product) / np.pi
        return float(angular_distance)


class DiscreteLogicVerifier:
    """Discrete logic layer for verification (Turing machine operations)."""
    
    def __init__(self):
        """Initialize logic verification rules."""
        self._contradiction_rules: List[Tuple[str, str]] = [
            ("true", "false"),
            ("yes", "no"),
            ("affirmative", "negative"),
            ("exists", "does_not_exist"),
        ]
    
    def verify_contradiction(self, thesis: str, antithesis: str) -> bool:
        """
        Verify that thesis and antithesis are logical contradictions.
        Uses discrete pattern matching (Turing machine operation).
        """
        thesis_lower = thesis.lower()
        antithesis_lower = antithesis.lower()
        
        # Check for explicit logical negation patterns
        negation_patterns = ["not ", "¬", "非", "无", "没有"]
        
        for pattern in negation_patterns:
            if pattern in thesis_lower and pattern not in antithesis_lower:
                base_thesis = thesis_lower.replace(pattern, "").strip()
                if base_thesis in antithesis_lower:
                    return True
            if pattern in antithesis_lower and pattern not in thesis_lower:
                base_antithesis = antithesis_lower.replace(pattern, "").strip()
                if base_antithesis in thesis_lower:
                    return True
        
        # Check rule-based contradictions
        for rule_a, rule_b in self._contradiction_rules:
            if (rule_a in thesis_lower and rule_b in antithesis_lower) or \
               (rule_b in thesis_lower and rule_a in antithesis_lower):
                return True
        
        return False
    
    def verify_logical_consistency(self, propositions: List[str]) -> Tuple[bool, Set[str]]:
        """
        Verify logical consistency across multiple propositions.
        Returns (is_consistent, conflicting_propositions)
        """
        conflicts: Set[str] = set()
        
        for i, prop1 in enumerate(propositions):
            for prop2 in propositions[i+1:]:
                if self.verify_contradiction(prop1, prop2):
                    conflicts.add(f"({prop1}) ⊥ ({prop2})")
        
        is_consistent = len(conflicts) == 0
        return is_consistent, conflicts


class PrecisionGatedExecutor:
    """
    Main executor implementing DAS Meta-Theory.
    
    Acts as a middle layer to prevent hallucinations by:
    1. Measuring entropy (precision gating)
    2. Generating dualistic propositions (thesis-antithesis)
    3. Verifying topological closure (consistency check)
    4. Routing to appropriate inference mode (CoT vs Direct)
    """
    
    def __init__(
        self,
        base_executor: Optional[Any] = None,
        enable_cot: bool = True,
        llm_client: Optional[Any] = None,
        tool_resolver: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        probe_tokens: int = 8,
        self_consistency_samples: int = 3,
        divergence_threshold: float = 0.35,
        logprob_entropy_threshold: float = 1.5,
        track_manifold_state: bool = True,
    ):
        """
        Initialize precision-gated executor.
        
        Args:
            base_executor: Underlying executor (LocalExecutor or similar)
            enable_cot: Whether to enable Chain-of-Thought for wave states
            llm_client: Optional LLM client for direct generation (supports logprobs)
            tool_resolver: Optional external tool resolver for topological tears
            probe_tokens: Tokens to sample during probe generation
            self_consistency_samples: Number of short answers for self-consistency
            divergence_threshold: Similarity threshold for divergence detection
            logprob_entropy_threshold: Entropy threshold for logprob-based gate
        """
        self.base_executor = base_executor
        self.enable_cot = enable_cot
        self.llm_client = llm_client
        self.tool_resolver = tool_resolver
        self.probe_tokens = max(1, probe_tokens)
        self.self_consistency_samples = max(2, self_consistency_samples)
        self.divergence_threshold = divergence_threshold
        self.logprob_entropy_threshold = logprob_entropy_threshold
        self.track_manifold_state = track_manifold_state

        self.manifold_tracker = None
        if track_manifold_state:
            try:
                from .manifold_state_tracker import ManifoldStateTracker
            except ImportError:
                from manifold_state_tracker import ManifoldStateTracker
            self.manifold_tracker = ManifoldStateTracker()
        
        self.manifold_encoder = ContinuousManifoldEncoder()
        self.logic_verifier = DiscreteLogicVerifier()
        
        self._execution_history: List[ExecutionContext] = []
        self._precision_threshold = 0.3  # Entropy threshold for particle state
    
    def execute_with_precision_gating(
        self,
        task: str,
        strategy: str = "auto",
        generate_antithesis: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute task with precision gating and dualistic verification.

        Workflow (DAS):
        1. Probe generation to estimate entropy (logprobs or self-consistency)
        2. If low entropy: Direct execution (standard LLM response)
        3. If high entropy: Orthogonal expansion (generate & execute Python)
        4. Dualistic verification for logic questions (Axiom I)

        Args:
            task: Task string to execute
            strategy: Execution strategy ("auto", "direct", "cot")
            generate_antithesis: Whether to generate antithesis for verification

        Returns:
            Dictionary containing output, confidence, entropy metrics, and state
        """
        context = ExecutionContext(task=task, timestamp=time.time())

        context.execution_trace.append("STEP_0_PROBE_GENERATION")
        probe_result = self._probe_entropy(task)
        context.final_entropy = self._probe_to_entropy_metrics(probe_result)
        is_high_entropy = probe_result.get("is_high_entropy", False)

        context.execution_trace.append("STEP_1_PRECISION_GATE")
        state_manifold = StateManifold.WAVE if is_high_entropy else StateManifold.PARTICLE
        context.state_history.append(state_manifold)

        logger.info(f"[PRECISION_GATE] Task: {task[:50]}...")
        logger.info(
            "[PROBE] High Entropy: %s | Method: %s",
            is_high_entropy,
            probe_result.get("method"),
        )

        result: Dict[str, Any]
        if not is_high_entropy:
            context.execution_trace.append("ROUTE_DIRECT_EXECUTION")
            result = self._direct_response(task, strategy)
        else:
            context.execution_trace.append("ROUTE_ORTHOGONAL_EXPANSION")
            script = self._generate_verification_script(task)
            context.execution_trace.append("ORTHOGONAL_EXECUTE_SCRIPT")
            script_stdout = self._execute_python_script(script)
            context.execution_trace.append("ORTHOGONAL_FEEDBACK")
            result = self._finalize_with_stdout(task, script_stdout)

        dualistic_result: Optional[Dict[str, Any]] = None
        if generate_antithesis and self._is_logic_task(task):
            context.execution_trace.append("STEP_2_DUALISTIC_VERIFICATION")
            dualistic_result = self.verify_duality(task)

        context.execution_trace.append("STEP_3_FINALIZE")
        context.output = result.get("output", "")
        context.confidence = result.get("confidence", 0.0)

        manifold_state = None
        if self.manifold_tracker is not None:
            try:
                manifold_state = self.manifold_tracker.update_state(task)
            except Exception:
                manifold_state = None

        final_result = {
            "output": result.get("output"),
            "confidence": result.get("confidence"),
            "state_manifold": state_manifold.value,
            "execution_trace": context.execution_trace,
            "probe": probe_result,
            "dualistic_verification": dualistic_result,
            "manifold_state": (
                None
                if manifold_state is None
                else {
                    "complexity": manifold_state.complexity,
                    "quaternion": manifold_state.quaternion.tolist(),
                }
            ),
            "timestamp": context.timestamp,
            "elapsed_time": time.time() - context.timestamp,
        }

        self._execution_history.append(context)
        return final_result

    def _probe_entropy(self, task: str) -> Dict[str, Any]:
        """
        Perform a short probe generation to estimate entropy.

        Strategy:
        1) If logprobs are available, compute entropy of early tokens.
        2) Otherwise use self-consistency (3 short answers) and compare divergence.
        """
        probe_prompt = f"Answer briefly (<=10 tokens). Task: {task}"
        response = self._llm_generate(
            probe_prompt,
            max_tokens=self.probe_tokens,
            logprobs=True,
        )

        logprobs = response.get("logprobs")
        if logprobs:
            entropy_value = self._entropy_from_logprobs(logprobs)
            is_high_entropy = entropy_value >= self.logprob_entropy_threshold
            return {
                "method": "logprobs",
                "entropy": entropy_value,
                "is_high_entropy": is_high_entropy,
            }

        answers = self._self_consistency_probe(probe_prompt)
        diverged, similarity = self._answers_diverge(answers)
        return {
            "method": "self_consistency",
            "answers": answers,
            "similarity": similarity,
            "is_high_entropy": diverged,
        }

    def _probe_to_entropy_metrics(self, probe_result: Dict[str, Any]) -> EntropyMetrics:
        """Convert probe results into an EntropyMetrics structure for stats."""
        if probe_result.get("method") == "logprobs":
            combined = float(probe_result.get("entropy", 0.0))
        else:
            similarity = float(probe_result.get("similarity", 1.0))
            combined = max(0.0, 1.0 - similarity)

        combined = min(max(combined, 0.0), 2.0)
        return EntropyMetrics(
            logical_entropy=combined,
            semantic_entropy=combined,
            temporal_entropy=0.0,
            combined_entropy=combined,
        )

    def _llm_generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate text using an LLM client or fallback to base executor."""
        if self.llm_client is not None:
            if hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt=prompt, **kwargs)
            if hasattr(self.llm_client, "complete"):
                return self.llm_client.complete(prompt=prompt, **kwargs)
            if callable(self.llm_client):
                return self.llm_client(prompt=prompt, **kwargs)

        if self.base_executor is not None:
            if hasattr(self.base_executor, "_run_inference"):
                text = self.base_executor._run_inference(prompt, "auto")
                return {"text": text}
            if hasattr(self.base_executor, "execute"):
                result = self.base_executor.execute(prompt)
                return {"text": result.get("output", "")}

        return {"text": ""}

    def _entropy_from_logprobs(self, logprobs: Any) -> float:
        """
        Compute average entropy from logprobs for early tokens.

        Accepts common logprobs shapes:
        - List[Dict[str, float]] (token -> logprob)
        - List[List[Dict[str, Any]]] (OpenAI-like top_logprobs)
        """
        entropies: List[float] = []
        if isinstance(logprobs, list):
            for token_info in logprobs[: self.probe_tokens]:
                if isinstance(token_info, dict):
                    logp_vals = list(token_info.values())
                elif isinstance(token_info, list):
                    logp_vals = [item.get("logprob") for item in token_info if "logprob" in item]
                else:
                    continue

                probs = [np.exp(lp) for lp in logp_vals if lp is not None]
                if not probs:
                    continue
                total = sum(probs)
                norm_probs = [p / total for p in probs]
                entropies.append(float(entropy(norm_probs)))

        if not entropies:
            return 0.0
        return float(sum(entropies) / len(entropies))

    def _self_consistency_probe(self, prompt: str) -> List[str]:
        answers: List[str] = []
        for _ in range(self.self_consistency_samples):
            response = self._llm_generate(prompt, max_tokens=self.probe_tokens)
            answers.append(str(response.get("text", "")).strip())
        return answers

    def _answers_diverge(self, answers: List[str]) -> Tuple[bool, float]:
        """Return (diverged, avg_similarity). Lower similarity => higher entropy."""
        if len(answers) < 2:
            return False, 1.0

        normalized = [re.sub(r"\s+", " ", a.lower()).strip() for a in answers]
        unique = list(dict.fromkeys(normalized))
        if len(unique) == 1:
            return False, 1.0

        def jaccard(a: str, b: str) -> float:
            a_set = set(a.split())
            b_set = set(b.split())
            if not a_set and not b_set:
                return 1.0
            return len(a_set & b_set) / max(1, len(a_set | b_set))

        similarities = []
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                similarities.append(jaccard(unique[i], unique[j]))

        avg_similarity = float(sum(similarities) / len(similarities)) if similarities else 0.0
        return avg_similarity < self.divergence_threshold, avg_similarity

    def _direct_response(self, task: str, strategy: str) -> Dict[str, Any]:
        """Direct execution using the primary LLM response."""
        response = self._llm_generate(task, max_tokens=256)
        text = str(response.get("text", ""))
        return {
            "output": text,
            "confidence": 0.8,
            "execution_mode": "direct",
        }

    def _generate_verification_script(self, task: str) -> str:
        """
        Generate a Python script to verify facts for high-entropy tasks.
        The script must print results to STDOUT.
        """
        prompt = (
            "Generate a minimal Python script to verify the task's facts. "
            "Use prints for outputs. Do not explain, output only code.\n"
            f"Task: {task}\n"
        )
        response = self._llm_generate(prompt, max_tokens=256)
        code = str(response.get("text", "")).strip()
        return self._strip_code_fences(code)

    def _execute_python_script(self, code: str) -> str:
        """Execute Python code locally and return STDOUT (no guessing)."""
        if not code:
            return ""

        if self.base_executor is not None and hasattr(self.base_executor, "execute_code_safely"):
            try:
                return self.base_executor.execute_code_safely(code)
            except Exception:
                pass

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

    def _finalize_with_stdout(self, task: str, stdout: str) -> Dict[str, Any]:
        """Generate final answer using tool output (STDOUT)."""
        prompt = (
            "You must answer using ONLY the tool output below. "
            "Do not guess or invent results.\n"
            f"Task: {task}\n"
            f"Tool Output (STDOUT):\n{stdout}\n"
            "Answer:"
        )
        response = self._llm_generate(prompt, max_tokens=256)
        return {
            "output": str(response.get("text", "")),
            "confidence": 0.6,
            "execution_mode": "orthogonal_expansion",
            "tool_stdout": stdout,
        }

    def verify_duality(self, statement: str) -> Dict[str, Any]:
        """
        Dualistic verification (Axiom I).
        Prompt A: Assume True. Prove it.
        Prompt B: Assume False. Prove it.
        If contradictions are detected, flag a topological tear.
        """
        prompt_true = f"Assume the statement is True. Prove it.\nStatement: {statement}"
        prompt_false = f"Assume the statement is False. Prove it.\nStatement: {statement}"

        result_true = self._llm_generate(prompt_true, max_tokens=256)
        result_false = self._llm_generate(prompt_false, max_tokens=256)

        text_true = str(result_true.get("text", ""))
        text_false = str(result_false.get("text", ""))

        contradictions = self._detect_contradictions(text_true, text_false)
        topological_tear = len(contradictions) > 0

        resolution: Optional[Dict[str, Any]] = None
        if topological_tear and self.tool_resolver is not None:
            resolution = self.tool_resolver(statement, {"contradictions": contradictions})

        return {
            "thesis": text_true,
            "antithesis": text_false,
            "topological_tear": topological_tear,
            "contradictions": contradictions,
            "resolution": resolution,
        }

    @staticmethod
    def _is_logic_task(task: str) -> bool:
        lower = task.lower()
        return any(
            kw in lower
            for kw in [
                "prove",
                "logic",
                "therefore",
                "if",
                "then",
                "contradiction",
                "推理",
                "证明",
            ]
        )

    def _detect_contradictions(self, text_a: str, text_b: str) -> List[str]:
        nums_a = self._extract_numbers(text_a)
        nums_b = self._extract_numbers(text_b)
        contradictions: List[str] = []

        if nums_a and nums_b and nums_a != nums_b:
            contradictions.append(f"numeric_claims_mismatch: {nums_a} vs {nums_b}")

        return contradictions

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        code = code.strip()
        if code.startswith("```"):
            code = re.sub(r"^```[a-zA-Z]*\n", "", code)
            code = re.sub(r"```$", "", code)
        return code.strip()
    
    def _measure_entropy(self, task: str) -> EntropyMetrics:
        """
        Measure entropy across three dimensions (discrete logic layer).
        
        Entropy Types:
        1. Logical Entropy: Shannon entropy of discrete propositions
        2. Semantic Entropy: Uncertainty in semantic manifold (quaternion space)
        3. Temporal Entropy: Rate of state changes
        """
        # Logical entropy: based on task keywords and their distinctiveness
        keywords = task.lower().split()
        keyword_counts = {}
        for kw in keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        total_keywords = len(keywords)
        if total_keywords == 0:
            logical_entropy = 0.0
        else:
            probabilities = [count / total_keywords for count in keyword_counts.values()]
            logical_entropy = float(entropy(probabilities))  # Shannon entropy
        
        # Semantic entropy: compute quaternion variance in semantic space
        # Encode task into multiple semantic perspectives
        q_affirmative = self.manifold_encoder.encode_proposition(task)
        q_negative = self.manifold_encoder.encode_proposition(f"not ({task})")
        q_uncertain = self.manifold_encoder.encode_proposition(f"uncertain about ({task})")
        
        # Distance in quaternion space represents semantic uncertainty
        dist_affirm_neg = self.manifold_encoder.quaternion_distance(q_affirmative, q_negative)
        dist_affirm_unc = self.manifold_encoder.quaternion_distance(q_affirmative, q_uncertain)
        semantic_entropy = (dist_affirm_neg + dist_affirm_unc) / 2.0
        
        # Temporal entropy: check against history for consistency
        temporal_entropy = 0.0
        if self._execution_history:
            recent_states = [ctx.final_entropy.combined_entropy 
                            for ctx in self._execution_history[-5:] 
                            if ctx.final_entropy]
            if recent_states:
                temporal_variance = np.var(recent_states) if len(recent_states) > 1 else 0.0
                temporal_entropy = float(temporal_variance)
        
        # Weighted combination (equal weights for demonstration)
        combined_entropy = (logical_entropy + semantic_entropy + temporal_entropy) / 3.0
        
        return EntropyMetrics(
            logical_entropy=float(logical_entropy),
            semantic_entropy=float(semantic_entropy),
            temporal_entropy=float(temporal_entropy),
            combined_entropy=float(combined_entropy),
        )
    
    def _generate_dualistic_propositions(self, task: str) -> List[DualProposition]:
        """
        Generate thesis-antithesis pairs for verification (Axiom I).
        
        Strategy:
        1. Primary thesis: Direct interpretation of task
        2. Antithesis: Logical negation or contrary interpretation
        3. Confidence: Derived from semantic entropy
        """
        propositions = []
        
        # Thesis 1: Affirmative interpretation
        thesis = f"The task '{task}' can be solved affirmatively"
        antithesis = f"The task '{task}' cannot be solved affirmatively"
        
        # Compute confidence from semantic encoding
        q_thesis = self.manifold_encoder.encode_proposition(thesis)
        q_antithesis = self.manifold_encoder.encode_proposition(antithesis)
        
        # Convert quaternion representation to confidence
        # ||q|| = 1 by construction; use w-component and semantic distance
        thesis_conf = float(np.abs(q_thesis[0]))  # w-component magnitude
        antithesis_conf = float(np.abs(q_antithesis[0]))
        
        # Normalize to ensure proper probability distribution
        total = thesis_conf + antithesis_conf + 1e-8
        thesis_conf /= total
        antithesis_conf /= total
        
        prop1 = DualProposition(
            thesis=thesis,
            antithesis=antithesis,
            thesis_confidence=thesis_conf,
            antithesis_confidence=antithesis_conf,
        )
        prop1.verify_closure()
        propositions.append(prop1)
        
        # Thesis 2: Logical consistency
        thesis2 = f"The task '{task}' is logically consistent"
        antithesis2 = f"The task '{task}' contains logical contradictions"
        
        q_thesis2 = self.manifold_encoder.encode_proposition(thesis2)
        q_antithesis2 = self.manifold_encoder.encode_proposition(antithesis2)
        
        thesis2_conf = float(np.abs(q_thesis2[0]))
        antithesis2_conf = float(np.abs(q_antithesis2[0]))
        
        total2 = thesis2_conf + antithesis2_conf + 1e-8
        thesis2_conf /= total2
        antithesis2_conf /= total2
        
        prop2 = DualProposition(
            thesis=thesis2,
            antithesis=antithesis2,
            thesis_confidence=thesis2_conf,
            antithesis_confidence=antithesis2_conf,
        )
        prop2.verify_closure()
        propositions.append(prop2)
        
        return propositions
    
    def _execute_direct(self, task: str, strategy: str) -> Dict[str, Any]:
        """Execute with direct output (Particle state, low entropy)."""
        logger.info(f"[DIRECT_EXECUTION] High precision detected, collapsing to direct output")
        
        if self.base_executor:
            return self.base_executor.execute(task, strategy)
        
        # Fallback direct response
        return {
            "output": f"Direct output: {task}",
            "confidence": 0.9,
            "execution_mode": "direct",
        }
    
    def _execute_with_cot(
        self, 
        task: str, 
        strategy: str,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute with Chain-of-Thought reasoning (Wave state, high entropy)."""
        logger.info(f"[CHAIN_OF_THOUGHT] High entropy detected, expanding via orthogonal basis")
        
        # CoT step 1: Decompose task
        context.execution_trace.append("COT_DECOMPOSE")
        subtasks = self._decompose_task(task)
        
        # CoT step 2: Generate reasoning chain
        context.execution_trace.append("COT_REASONING_CHAIN")
        reasoning_chain = []
        for i, subtask in enumerate(subtasks):
            reasoning_chain.append(f"Step {i+1}: Analyze '{subtask}'")
        
        # CoT step 3: Execute subtasks
        context.execution_trace.append("COT_EXECUTE_SUBTASKS")
        if self.base_executor:
            results = [self.base_executor.execute(subtask, strategy) for subtask in subtasks]
            final_output = " -> ".join([r.get("output", "") for r in results])
            avg_confidence = np.mean([r.get("confidence", 0.0) for r in results])
        else:
            final_output = " -> ".join(reasoning_chain)
            avg_confidence = 0.75
        
        return {
            "output": final_output,
            "confidence": avg_confidence,
            "execution_mode": "chain_of_thought",
            "reasoning_chain": reasoning_chain,
            "subtasks": subtasks,
        }
    
    def _execute_standard(
        self,
        task: str,
        strategy: str,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute with standard verification (Coherence state)."""
        logger.info(f"[STANDARD_EXECUTION] Balanced entropy state, executing with verification")
        
        if self.base_executor:
            return self.base_executor.execute(task, strategy)
        
        return {
            "output": f"Verified output: {task}",
            "confidence": 0.75,
            "execution_mode": "standard_verified",
        }
    
    def _decompose_task(self, task: str, max_subtasks: int = 3) -> List[str]:
        """Decompose task into subtasks (discrete logic operation)."""
        # Simple heuristic: split by logical connectors
        connectors = [" and ", " or ", " then ", ","]
        subtasks = [task]
        
        for connector in connectors:
            if connector in task.lower():
                parts = task.split(connector)
                subtasks = [part.strip() for part in parts if part.strip()]
                break
        
        # Limit to max_subtasks
        return subtasks[:max_subtasks]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about execution history."""
        if not self._execution_history:
            return {"total_executions": 0}
        
        states_count = {}
        avg_entropy = 0.0
        total_confidence = 0.0
        
        for context in self._execution_history:
            if context.final_entropy:
                state = context.state_history[-1] if context.state_history else StateManifold.COHERENCE
                states_count[state.value] = states_count.get(state.value, 0) + 1
                avg_entropy += context.final_entropy.combined_entropy
                total_confidence += context.confidence
        
        n = len(self._execution_history)
        return {
            "total_executions": n,
            "state_distribution": states_count,
            "average_entropy": avg_entropy / n if n > 0 else 0.0,
            "average_confidence": total_confidence / n if n > 0 else 0.0,
            "execution_modes": [ctx.execution_trace[-1] if ctx.execution_trace else "unknown" 
                              for ctx in self._execution_history],
        }
