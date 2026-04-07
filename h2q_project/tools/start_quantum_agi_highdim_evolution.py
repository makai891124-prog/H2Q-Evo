#!/usr/bin/env python3
"""Start a high-dimensional quantum-AGI evolution instance.

This runner extends the existing long-evolution pipeline with:
1) High-dimensional witness projection.
2) Parallel multi-branch projection scoring.
3) Acceptance gates for capability uplift.

Outputs:
- quantum_agi_highdim_state.json
- quantum_agi_highdim_cycles.jsonl
- quantum_agi_highdim_report.json
- quantum_agi_highdim_report.md
- quantum_agi_highdim_acceptance.json
- quantum_agi_highdim_acceptance.md
- quantum_agi_highdim_acceptance_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from STQ_QuantumSimulator import STQ_QuantumSimulator  # noqa: E402
from h2q_project.h2q.agi.evolution_24h import (  # noqa: E402
    CapabilityTester,
    FractalCompressor,
    KnowledgeAcquirer,
)
from h2q_project.h2q.agi.survival_daemon import (  # noqa: E402
    SurvivalConfig,
    create_survival_daemon,
)
from h2q_project.tools.uplift_metrics import (  # noqa: E402
    RollingUpliftWindow,
    StrategyPersistenceManager,
)


logger = logging.getLogger(__name__)


@dataclass
class HighDimRunnerConfig:
    hours: float
    cycle_seconds: int
    time_points: int
    capability_check_every: int
    max_cycles: int
    max_knowledge_items: int
    compression_threshold: float

    mass_kg: float
    distance_m: float
    gamma_base: float
    lambda_threshold: float
    formula_mode: str

    resume: bool
    output_dir: Path

    projection_dim: int
    parallel_branches: int
    projection_seed: int
    resource_profile: str
    max_parallel_workers: int
    capability_timeout_seconds: float
    important_cycle_every: int
    force_acceptance_prompt: bool

    acceptance_min_cycles: int
    acceptance_composite_min: float
    acceptance_capability_min: float
    acceptance_entanglement_min: float
    acceptance_highdim_consensus_min: float
    acceptance_uplift_min: float
    acceptance_min_capability_measurements: int
    acceptance_min_forced_prompts: int
    strict_acceptance: bool


def _bootstrap_ci(values: List[float], *, seed: int, alpha: float = 0.05) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return {"mean": mean_val, "lower": mean_val, "upper": mean_val}

    rng = np.random.default_rng(seed + arr.size)
    rounds = 1200
    idx = rng.integers(0, arr.size, size=(rounds, arr.size))
    samples = np.mean(arr[idx], axis=1)
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {"mean": mean_val, "lower": lower, "upper": upper}


def _safe_entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-np.sum(prob * np.log(prob)))


def _evaluate_projection_branch(
    branch_id: int,
    features: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> Dict[str, float]:
    projected = np.tanh(features @ matrix + bias)
    abs_proj = np.abs(projected)

    total = float(np.sum(abs_proj))
    if total <= 1e-12:
        probs = np.full(projected.shape, 1.0 / max(1, projected.size), dtype=np.float64)
    else:
        probs = abs_proj / total

    entropy = _safe_entropy(probs) / max(1e-12, math.log(max(2, projected.size)))
    coherence = float(np.abs(np.mean(np.exp(1j * projected))))
    norm = float(np.linalg.norm(projected) / max(1e-12, math.sqrt(projected.size)))
    sparsity = float(np.mean(abs_proj < 0.20))

    score = float(np.clip(0.45 * coherence + 0.30 * entropy + 0.15 * norm + 0.10 * (1.0 - sparsity), 0.0, 1.0))

    return {
        "branch": float(branch_id),
        "score": score,
        "coherence": coherence,
        "entropy": float(entropy),
        "norm": norm,
        "sparsity": sparsity,
    }


class HighDimQuantumAGIRunner:
    def __init__(self, cfg: HighDimRunnerConfig, *, china_mode: Optional[bool]) -> None:
        self.cfg = cfg
        self.start_ts = time.time()
        self.stop_requested = False

        self.quantum_sim = STQ_QuantumSimulator(mass_kg=cfg.mass_kg, distance_m=cfg.distance_m)
        self.tester = CapabilityTester()
        self.acquirer = KnowledgeAcquirer(china_mode=china_mode)
        self.compressor = FractalCompressor(compression_ratio=0.55)

        self.cycles: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.latest_capability_score = 0.0
        self.latest_base_score = 0.0
        self.latest_enhanced_score = 0.0
        self.capability_checked_cycles = 0
        self.capability_check_failures = 0
        self.forced_prompt_count = 0
        self.force_capability_check_next_cycle = False
        self.strategy_topic_boost: Dict[str, float] = {}
        self.strategy_capability_interval_override: Optional[int] = None
        self.strategy_cycles_left: int = 0
        self.slope_alert_active = False
        self.strategy_hold_cross_window = False
        self.uplift_positive_streak = 0

        # Phase 1 Track A: Uplift Window Tracker (Rolling window for slope-alarm detection)
        self.uplift_window = RollingUpliftWindow(window_size=3, alarm_threshold=-0.005)
        self.strategy_persistence = StrategyPersistenceManager()

        self.resource_profile_requested = str(cfg.resource_profile).strip().lower() or "auto"
        self.resource_profile_applied = "balanced"
        self.parallel_worker_limit = 1
        self._apply_resource_profile()

        self.resumed = False
        self.resumed_from_cycle = 0
        self.resumed_at: Optional[str] = None

        self.curriculum = [
            "quantum_mechanics",
            "topological_quantum_computing",
            "machine_learning",
            "mathematics",
            "computer_science",
            "physics",
            "information_theory",
            "causal_inference",
        ]

        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.output_dir / "quantum_agi_highdim_state.json"
        self.cycles_path = self.output_dir / "quantum_agi_highdim_cycles.jsonl"
        self.report_json_path = self.output_dir / "quantum_agi_highdim_report.json"
        self.report_md_path = self.output_dir / "quantum_agi_highdim_report.md"
        self.acceptance_json_path = self.output_dir / "quantum_agi_highdim_acceptance.json"
        self.acceptance_md_path = self.output_dir / "quantum_agi_highdim_acceptance.md"
        self.acceptance_prompts_path = self.output_dir / "quantum_agi_highdim_acceptance_prompts.jsonl"

        self.feature_dim = 20
        rng = np.random.default_rng(cfg.projection_seed)
        self.branch_matrices = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(self.feature_dim),
            size=(self.cfg.parallel_branches, self.feature_dim, self.cfg.projection_dim),
        ).astype(np.float64)
        self.branch_bias = rng.uniform(
            low=-math.pi,
            high=math.pi,
            size=(self.cfg.parallel_branches, self.cfg.projection_dim),
        ).astype(np.float64)

        self.baseline_snapshot = self._load_local_baselines()
        self._restore_if_requested()

    def request_stop(self) -> None:
        self.stop_requested = True

    def _elapsed_hours(self) -> float:
        return (time.time() - self.start_ts) / 3600.0

    def _apply_resource_profile(self) -> None:
        cpu_count = max(1, int(os.cpu_count() or 1))
        requested = self.resource_profile_requested
        if requested not in {"auto", "low", "balanced", "high"}:
            requested = "auto"

        applied = requested
        if requested == "auto":
            applied = "low" if cpu_count <= 4 else "balanced"

        if applied == "low":
            self.cfg.parallel_branches = max(2, min(self.cfg.parallel_branches, max(2, min(3, cpu_count))))
            self.cfg.projection_dim = max(16, min(self.cfg.projection_dim, 64))
            self.cfg.time_points = max(40, min(self.cfg.time_points, 140))
            self.cfg.capability_check_every = max(3, self.cfg.capability_check_every)
            worker_limit = min(self.cfg.parallel_branches, max(1, min(2, cpu_count)))
        elif applied == "balanced":
            self.cfg.parallel_branches = max(2, min(self.cfg.parallel_branches, max(2, min(6, cpu_count))))
            self.cfg.projection_dim = max(16, min(self.cfg.projection_dim, 128))
            self.cfg.time_points = max(40, min(self.cfg.time_points, 220))
            worker_limit = min(self.cfg.parallel_branches, max(1, min(4, cpu_count)))
        else:
            worker_limit = min(self.cfg.parallel_branches, max(1, min(8, cpu_count)))

        manual_limit = int(self.cfg.max_parallel_workers)
        if manual_limit > 0:
            worker_limit = min(worker_limit, manual_limit)

        self.parallel_worker_limit = max(1, int(worker_limit))
        self.resource_profile_requested = requested
        self.resource_profile_applied = applied

    def _composite_uplift_metrics(self, composite_values: List[float]) -> Dict[str, float]:
        if not composite_values:
            return {
                "window_size": 1.0,
                "initial_window_mean": 0.0,
                "final_window_mean": 0.0,
                "uplift": 0.0,
            }

        window = max(1, len(composite_values) // 4)
        initial_mean = float(np.mean(composite_values[:window]))
        final_mean = float(np.mean(composite_values[-window:]))
        uplift = float(final_mean - initial_mean)
        return {
            "window_size": float(window),
            "initial_window_mean": initial_mean,
            "final_window_mean": final_mean,
            "uplift": uplift,
        }

    def _is_important_cycle(self, cycle_idx: int, capability: Dict[str, Any], enhanced_score: float) -> bool:
        if cycle_idx == 1:
            return True
        if bool(capability.get("checked")):
            return True
        if self.slope_alert_active:
            return True

        # Phase 1 Track A: Check if slope-alarm would be triggered with current score
        # This provides early warning for strategy application
        projected_values = [float(c.get("composite_score", 0.0)) for c in self.cycles]
        projected_values.append(float(enhanced_score))
        
        # Check if current window would trigger slope-alarm
        window_snapshot = RollingUpliftWindow(window_size=3, alarm_threshold=-0.005)
        for val in projected_values:
            window_snapshot.push_value(val, 0)
        if window_snapshot.is_alarm_triggered():
            return True

        stride = max(1, int(self.cfg.important_cycle_every))
        return (cycle_idx % stride) == 0

    def _interim_acceptance_status(
        self,
        *,
        enhanced_score: float,
        capability: Dict[str, Any],
        quantum: Dict[str, Any],
        highdim: Dict[str, Any],
    ) -> Dict[str, Any]:
        composite_values = [float(c.get("composite_score", 0.0)) for c in self.cycles]
        composite_values.append(float(enhanced_score))

        ent_values = [float(c.get("quantum", {}).get("entanglement_negative_ratio", 0.0)) for c in self.cycles]
        ent_values.append(float(quantum.get("entanglement_negative_ratio", 0.0)))

        highdim_values = [float(c.get("highdim", {}).get("consensus_score", 0.0)) for c in self.cycles]
        highdim_values.append(float(highdim.get("consensus_score", 0.0)))

        cap_scores = [
            float(c.get("capability", {}).get("overall_score", 0.0))
            for c in self.cycles
            if bool(c.get("capability", {}).get("checked"))
        ]
        if bool(capability.get("checked")):
            cap_scores.append(float(capability.get("overall_score", 0.0)))

        comp_mean = float(np.mean(composite_values)) if composite_values else 0.0
        cap_mean = float(np.mean(cap_scores)) if cap_scores else 0.0
        ent_mean = float(np.mean(ent_values)) if ent_values else 0.0
        highdim_mean = float(np.mean(highdim_values)) if highdim_values else 0.0

        uplift_metrics = self._composite_uplift_metrics(composite_values)
        window = int(uplift_metrics.get("window_size", 1.0))
        uplift = float(uplift_metrics.get("uplift", 0.0))

        cycles_now = len(self.cycles) + 1
        cap_measurements = len(cap_scores)
        forced_prompts_now = int(self.forced_prompt_count)

        criteria = [
            {
                "name": "minimum_cycles",
                "value": int(cycles_now),
                "threshold": int(self.cfg.acceptance_min_cycles),
                "passed": bool(cycles_now >= self.cfg.acceptance_min_cycles),
            },
            {
                "name": "enhanced_composite_mean",
                "value": comp_mean,
                "threshold": float(self.cfg.acceptance_composite_min),
                "passed": bool(comp_mean >= self.cfg.acceptance_composite_min),
            },
            {
                "name": "capability_measurements",
                "value": int(cap_measurements),
                "threshold": int(self.cfg.acceptance_min_capability_measurements),
                "passed": bool(cap_measurements >= self.cfg.acceptance_min_capability_measurements),
            },
            {
                "name": "capability_score_mean",
                "value": cap_mean,
                "threshold": float(self.cfg.acceptance_capability_min),
                "passed": bool(cap_mean >= self.cfg.acceptance_capability_min),
            },
            {
                "name": "entanglement_ratio_mean",
                "value": ent_mean,
                "threshold": float(self.cfg.acceptance_entanglement_min),
                "passed": bool(ent_mean >= self.cfg.acceptance_entanglement_min),
            },
            {
                "name": "highdim_consensus_mean",
                "value": highdim_mean,
                "threshold": float(self.cfg.acceptance_highdim_consensus_min),
                "passed": bool(highdim_mean >= self.cfg.acceptance_highdim_consensus_min),
            },
            {
                "name": "composite_uplift",
                "value": uplift,
                "threshold": float(self.cfg.acceptance_uplift_min),
                "passed": bool(uplift >= self.cfg.acceptance_uplift_min),
            },
        ]

        if bool(self.cfg.strict_acceptance):
            criteria.append(
                {
                    "name": "forced_acceptance_prompts",
                    "value": int(forced_prompts_now),
                    "threshold": int(self.cfg.acceptance_min_forced_prompts),
                    "passed": bool(forced_prompts_now >= self.cfg.acceptance_min_forced_prompts),
                }
            )

        gaps = [str(c["name"]) for c in criteria if not bool(c["passed"])]

        return {
            "criteria": criteria,
            "gaps": gaps,
            "metrics": {
                "cycles": int(cycles_now),
                "enhanced_composite_mean": comp_mean,
                "capability_measurements": int(cap_measurements),
                "forced_acceptance_prompts": int(forced_prompts_now),
                "capability_score_mean": cap_mean,
                "entanglement_ratio_mean": ent_mean,
                "highdim_consensus_mean": highdim_mean,
                "composite_uplift": uplift,
                "slope_alert": bool(uplift < 0.0),
                "window_size": int(window),
            },
        }

    def _write_acceptance_prompt(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self.acceptance_prompts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _derive_prompt_driven_strategy(self, gaps: List[str], *, slope_alert: bool) -> Dict[str, Any]:
        topic_weights: Dict[str, float] = {t: 0.0 for t in self.curriculum}

        def _add(topics: List[str], weight: float) -> None:
            for t in topics:
                if t in topic_weights:
                    topic_weights[t] += weight

        if "composite_uplift" in gaps or "enhanced_composite_mean" in gaps:
            _add(["mathematics", "information_theory", "machine_learning"], 1.5)
        if "capability_score_mean" in gaps:
            _add(["machine_learning", "computer_science", "causal_inference"], 1.4)
        if "highdim_consensus_mean" in gaps:
            _add(["topological_quantum_computing", "quantum_mechanics", "mathematics"], 1.2)
        if "entanglement_ratio_mean" in gaps:
            _add(["quantum_mechanics", "physics"], 1.1)
        if "capability_measurements" in gaps:
            _add(["computer_science", "machine_learning"], 0.8)
        if slope_alert:
            _add(["mathematics", "information_theory", "machine_learning", "causal_inference"], 1.8)

        ranked = sorted(topic_weights.items(), key=lambda kv: kv[1], reverse=True)
        preferred_topics = [k for k, v in ranked if v > 0.0][:4]

        if slope_alert:
            capability_interval = 1
        elif "capability_measurements" in gaps:
            capability_interval = 1
        elif "capability_score_mean" in gaps or "composite_uplift" in gaps:
            capability_interval = max(1, min(2, int(self.cfg.capability_check_every)))
        else:
            capability_interval = int(self.cfg.capability_check_every)

        if slope_alert:
            exploration_mode = "uplift_alert_recovery"
        elif "composite_uplift" in gaps or "enhanced_composite_mean" in gaps:
            exploration_mode = "uplift_recovery"
        elif "highdim_consensus_mean" in gaps:
            exploration_mode = "consensus_stabilize"
        elif "capability_score_mean" in gaps:
            exploration_mode = "capability_focus"
        else:
            exploration_mode = "balanced"

        return {
            "preferred_topics": preferred_topics,
            "topic_boost": {k: float(v) for k, v in topic_weights.items() if v > 0.0},
            "capability_interval": int(capability_interval),
            "exploration_mode": exploration_mode,
            "cross_window_hold": bool(slope_alert),
        }

    def _apply_prompt_driven_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        raw_boost = strategy.get("topic_boost", {})
        topic_boost: Dict[str, float] = {}
        if isinstance(raw_boost, dict):
            for k, v in raw_boost.items():
                if k in self.curriculum:
                    try:
                        topic_boost[k] = max(0.0, float(v))
                    except Exception:
                        continue

        self.strategy_topic_boost = topic_boost

        try:
            interval = int(strategy.get("capability_interval", self.cfg.capability_check_every))
        except Exception:
            interval = int(self.cfg.capability_check_every)
        self.strategy_capability_interval_override = max(1, interval)

        self.strategy_cycles_left = max(1, min(4, int(self.cfg.important_cycle_every)))

        if bool(strategy.get("cross_window_hold", False)):
            self.strategy_hold_cross_window = True
            self.slope_alert_active = True
            self.uplift_positive_streak = 0

        return {
            "topic_boost": self.strategy_topic_boost,
            "preferred_topics": [
                t for t in strategy.get("preferred_topics", []) if isinstance(t, str) and t in self.curriculum
            ],
            "capability_interval": int(self.strategy_capability_interval_override),
            "exploration_mode": str(strategy.get("exploration_mode", "balanced")),
            "strategy_cycles_left": int(self.strategy_cycles_left),
            "cross_window_hold": bool(self.strategy_hold_cross_window),
        }

    def _decay_prompt_driven_strategy(self) -> None:
        if self.strategy_hold_cross_window:
            return

        if self.strategy_cycles_left <= 0:
            self.strategy_topic_boost = {}
            self.strategy_capability_interval_override = None
            self.strategy_cycles_left = 0
            return

        self.strategy_cycles_left -= 1
        if self.strategy_cycles_left <= 0:
            self.strategy_topic_boost = {}
            self.strategy_capability_interval_override = None
            self.strategy_cycles_left = 0

    def _update_uplift_alert_state(self) -> Dict[str, Any]:
        metrics = self._composite_uplift_metrics([float(c.get("composite_score", 0.0)) for c in self.cycles])
        uplift = float(metrics.get("uplift", 0.0))
        window_size = max(1, int(metrics.get("window_size", 1.0)))

        released_now = False
        if uplift < 0.0:
            self.slope_alert_active = True
            self.strategy_hold_cross_window = True
            self.uplift_positive_streak = 0
        elif self.slope_alert_active:
            self.uplift_positive_streak += 1
            recovery_needed = max(2, window_size)
            if self.uplift_positive_streak >= recovery_needed:
                self.slope_alert_active = False
                self.strategy_hold_cross_window = False
                self.uplift_positive_streak = 0
                self.strategy_cycles_left = 0
                self.strategy_topic_boost = {}
                self.strategy_capability_interval_override = None
                released_now = True

        return {
            "slope_alert_active": bool(self.slope_alert_active),
            "strategy_hold_cross_window": bool(self.strategy_hold_cross_window),
            "uplift_positive_streak": int(self.uplift_positive_streak),
            "recovery_released": bool(released_now),
            "window_size": int(window_size),
            "uplift": float(uplift),
        }

    def _choose_topic_for_cycle(self, cycle_idx: int) -> Tuple[str, Dict[str, Any]]:
        if not self.curriculum:
            return "general", {"source": "fallback", "reason": "empty_curriculum"}

        base_idx = (cycle_idx - 1) % len(self.curriculum)
        base_topic = self.curriculum[base_idx]

        if not self.strategy_topic_boost:
            return base_topic, {
                "source": "curriculum",
                "base_topic": base_topic,
                "strategy_cycles_left": int(self.strategy_cycles_left),
            }

        scored: List[Tuple[float, str]] = []
        for i, topic in enumerate(self.curriculum):
            boost = float(self.strategy_topic_boost.get(topic, 0.0))
            base_bonus = 0.20 if i == base_idx else 0.0
            score = boost + base_bonus
            scored.append((score, topic))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[0][1] if scored else base_topic

        return selected, {
            "source": "prompt_strategy",
            "base_topic": base_topic,
            "selected_topic": selected,
            "topic_boost": dict(self.strategy_topic_boost),
            "strategy_cycles_left": int(self.strategy_cycles_left),
        }

    def _build_forced_acceptance_prompt(
        self,
        *,
        cycle_idx: int,
        enhanced_score: float,
        capability: Dict[str, Any],
        quantum: Dict[str, Any],
        highdim: Dict[str, Any],
    ) -> Dict[str, Any]:
        snapshot = self._interim_acceptance_status(
            enhanced_score=enhanced_score,
            capability=capability,
            quantum=quantum,
            highdim=highdim,
        )

        gaps = list(snapshot.get("gaps", []))
        metrics = snapshot.get("metrics", {})
        slope_alert = bool(float(metrics.get("composite_uplift", 0.0)) < 0.0)

        actions: List[str] = []
        if "minimum_cycles" in gaps:
            remain = max(0, int(self.cfg.acceptance_min_cycles) - int(metrics.get("cycles", 0)))
            actions.append(f"继续稳定运行，至少再完成 {remain} 个周期。")
        if "capability_measurements" in gaps:
            actions.append("下一个周期必须执行 capability 测试，补齐验收样本。")
        if "capability_score_mean" in gaps:
            actions.append("优先提升能力测试得分，锁定高收益知识主题并保留当前模型稳定性。")
        if "highdim_consensus_mean" in gaps:
            actions.append("优先提升高维共识：降低分支分歧并维持投影稳定。")
        if "enhanced_composite_mean" in gaps or "composite_uplift" in gaps:
            actions.append("优先提高综合得分与提升斜率，避免无效探索。")
        if slope_alert:
            actions.append("检测到提升斜率为负，立即进入跨窗口恢复策略并持续驱动后续周期。")
        if "entanglement_ratio_mean" in gaps:
            actions.append("保持纠缠见证负值比例，避免退化到经典区间。")
        if not actions:
            actions.append("所有当前门禁已达标，保持策略并继续累积统计显著性。")

        force_cap_next = bool(
            (
                ("capability_measurements" in gaps)
                or ("capability_score_mean" in gaps)
                or slope_alert
            )
            and not bool(capability.get("checked"))
        )

        prompt_lines = [
            f"[FORCE-ACCEPTANCE] cycle={cycle_idx} important_cycle=1",
            (
                "在本地有限资源条件下，必须优先完成验收闭环："
                "持续运行、补齐能力样本、提高高维共识与综合得分。"
            ),
            f"resource_profile={self.resource_profile_applied} workers={self.parallel_worker_limit}",
            f"gaps={','.join(gaps) if gaps else 'none'}",
            f"slope_alert={int(slope_alert)}",
            f"actions={' | '.join(actions)}",
        ]
        prompt_text = "\n".join(prompt_lines)

        payload = {
            "timestamp": datetime.now().isoformat(),
            "cycle": int(cycle_idx),
            "important_cycle": True,
            "gaps": gaps,
            "actions": actions,
            "slope_alert": bool(slope_alert),
            "force_capability_next_cycle": bool(force_cap_next),
            "prompt": prompt_text,
            "metrics": metrics,
            "strategy": self._derive_prompt_driven_strategy(gaps, slope_alert=slope_alert),
        }
        self._write_acceptance_prompt(payload)
        return payload

    def _load_local_baselines(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}

        paths = {
            "paper2206_report": PROJECT_ROOT / "h2q_project/reports/paper2206_ray_structure_space/paper2206_argument_space_report.json",
            "paper2604_report": PROJECT_ROOT / "h2q_project/reports/paper2604_argument_space/paper2604_argument_space_report.json",
            "quantum_equivalence": PROJECT_ROOT / "quantum_equivalence_report.json",
            "agi_benchmark": PROJECT_ROOT / "AGI_BENCHMARK_REPORT.json",
            "agi_eval": PROJECT_ROOT / "agi_evaluation_results.json",
        }

        for name, path in paths.items():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if name in {"paper2206_report", "paper2604_report"}:
                snapshot[name] = {
                    "overall_passed": bool(data.get("overall_passed", False)),
                    "checks": data.get("checks", {}),
                }
            elif name == "quantum_equivalence":
                h2q = data.get("h2q_results", {})
                ghz = h2q.get("GHZ_State_Fidelity", {})
                snapshot[name] = {
                    "timestamp": data.get("timestamp"),
                    "bv": h2q.get("Bernstein_Vazirani"),
                    "ghz_fidelity_mean": float(np.mean(list(ghz.values()))) if ghz else 0.0,
                }
            elif name == "agi_benchmark":
                snapshot[name] = {
                    "overall_score": float(data.get("overall_score", 0.0)),
                    "pass_rate": float(data.get("overall_pass_rate", 0.0)),
                    "verdict": data.get("superiority_verdict", "unknown"),
                }
            elif name == "agi_eval":
                scores = data.get("scores", {})
                snapshot[name] = {
                    "overall_score": float(scores.get("overall_score", 0.0)),
                    "consciousness_score": float(scores.get("consciousness_score", 0.0)),
                    "adaptability_score": float(scores.get("adaptability_score", 0.0)),
                }

        return snapshot

    def _load_existing_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _load_existing_cycles(self) -> List[Dict[str, Any]]:
        if not self.cycles_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(self.cycles_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except Exception:
            return []

        return records

    def _parse_iso_ts(self, raw: Any) -> Optional[float]:
        if not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            return None

    def _restore_knowledge_placeholders(self, count: int) -> None:
        restore_count = max(0, min(count, self.cfg.max_knowledge_items))
        for idx in range(restore_count):
            key = f"restored_knowledge_{idx:06d}"
            self.knowledge_base[key] = {"restored": True}

    def _restore_if_requested(self) -> None:
        if not self.cfg.resume:
            return

        state = self._load_existing_state()
        restored_cycles = self._load_existing_cycles()

        if not state and not restored_cycles:
            return

        if restored_cycles:
            self.cycles = restored_cycles
            self.resumed_from_cycle = len(restored_cycles)
            self.capability_checked_cycles = int(
                sum(1 for c in restored_cycles if bool(c.get("capability", {}).get("checked")))
            )
            self.capability_check_failures = int(
                sum(1 for c in restored_cycles if bool(c.get("capability", {}).get("check_error")))
            )
            self.forced_prompt_count = int(
                sum(1 for c in restored_cycles if bool(c.get("control", {}).get("important_cycle")))
            )

        if state:
            started_ts = self._parse_iso_ts(state.get("started_at"))
            if started_ts is not None:
                self.start_ts = started_ts

            try:
                self.latest_capability_score = float(state.get("latest_capability_score", self.latest_capability_score))
            except Exception:
                pass

            try:
                self.latest_base_score = float(state.get("latest_base_score", self.latest_base_score))
            except Exception:
                pass

            try:
                self.latest_enhanced_score = float(state.get("latest_enhanced_score", self.latest_enhanced_score))
            except Exception:
                pass

            try:
                self.capability_checked_cycles = max(
                    self.capability_checked_cycles,
                    int(state.get("capability_checked_cycles", 0)),
                )
            except Exception:
                pass

            try:
                self.capability_check_failures = max(
                    self.capability_check_failures,
                    int(state.get("capability_check_failures", 0)),
                )
            except Exception:
                pass

            try:
                self.forced_prompt_count = max(
                    self.forced_prompt_count,
                    int(state.get("forced_prompt_count", 0)),
                )
            except Exception:
                pass

            self.force_capability_check_next_cycle = bool(state.get("force_capability_check_next_cycle", False))

            try:
                knowledge_count = int(state.get("knowledge_count", 0))
            except Exception:
                knowledge_count = 0
            self._restore_knowledge_placeholders(knowledge_count)

            try:
                self.acquirer.acquired_count = max(
                    self.acquirer.acquired_count,
                    int(state.get("acquired_count", 0)),
                )
            except Exception:
                pass

            try:
                self.acquirer.failed_count = max(
                    self.acquirer.failed_count,
                    int(state.get("failed_count", 0)),
                )
            except Exception:
                pass

            raw_boost = state.get("strategy_topic_boost", {})
            if isinstance(raw_boost, dict):
                restored_boost: Dict[str, float] = {}
                for k, v in raw_boost.items():
                    key = str(k)
                    if key not in self.curriculum:
                        continue
                    try:
                        restored_boost[key] = max(0.0, float(v))
                    except Exception:
                        continue
                self.strategy_topic_boost = restored_boost

            try:
                raw_interval = state.get("strategy_capability_interval_override", None)
                self.strategy_capability_interval_override = (
                    None if raw_interval is None else max(1, int(raw_interval))
                )
            except Exception:
                self.strategy_capability_interval_override = None

            try:
                self.strategy_cycles_left = max(0, int(state.get("strategy_cycles_left", 0)))
            except Exception:
                self.strategy_cycles_left = 0

            self.slope_alert_active = bool(state.get("slope_alert_active", False))
            self.strategy_hold_cross_window = bool(state.get("strategy_hold_cross_window", False))
            try:
                self.uplift_positive_streak = max(0, int(state.get("uplift_positive_streak", 0)))
            except Exception:
                self.uplift_positive_streak = 0

            if self.resumed_from_cycle == 0:
                try:
                    self.resumed_from_cycle = max(0, int(state.get("cycle_count", 0)))
                except Exception:
                    pass

            # Phase 1 Track A: Restore uplift window state
            window_state = state.get("uplift_window_state")
            if isinstance(window_state, dict):
                try:
                    self.uplift_window = RollingUpliftWindow.from_dict(window_state)
                except Exception as e:
                    logger.warning(f"Failed to restore uplift_window_state: {e}")
                    self.uplift_window = RollingUpliftWindow(window_size=3, alarm_threshold=-0.005)

        self.resumed = True
        self.resumed_at = datetime.now().isoformat()

    def _save_state(self) -> None:
        data = {
            "started_at": datetime.fromtimestamp(self.start_ts).isoformat(),
            "saved_at": datetime.now().isoformat(),
            "elapsed_hours": self._elapsed_hours(),
            "cycle_count": len(self.cycles),
            "latest_capability_score": self.latest_capability_score,
            "latest_base_score": self.latest_base_score,
            "latest_enhanced_score": self.latest_enhanced_score,
            "capability_checked_cycles": int(self.capability_checked_cycles),
            "capability_check_failures": int(self.capability_check_failures),
            "forced_prompt_count": int(self.forced_prompt_count),
            "force_capability_check_next_cycle": bool(self.force_capability_check_next_cycle),
            "strategy_topic_boost": self.strategy_topic_boost,
            "strategy_capability_interval_override": self.strategy_capability_interval_override,
            "strategy_cycles_left": int(self.strategy_cycles_left),
            "slope_alert_active": bool(self.slope_alert_active),
            "strategy_hold_cross_window": bool(self.strategy_hold_cross_window),
            "uplift_positive_streak": int(self.uplift_positive_streak),
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
            "resource_profile_requested": self.resource_profile_requested,
            "resource_profile_applied": self.resource_profile_applied,
            "parallel_worker_limit": int(self.parallel_worker_limit),
            "resume_enabled": self.cfg.resume,
            "resumed": self.resumed,
            "resumed_from_cycle": self.resumed_from_cycle,
            "resumed_at": self.resumed_at,
            # Phase 1 Track A: Uplift window state persistence
            "uplift_window_state": self.uplift_window.to_dict(),
        }
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def _write_cycle(self, record: Dict[str, Any]) -> None:
        with open(self.cycles_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _knowledge_step(self, cycle_idx: int) -> Dict[str, Any]:
        topic, topic_plan = self._choose_topic_for_cycle(cycle_idx)
        acquired = self.acquirer.fetch_summary(topic)
        added = False

        if acquired:
            key = f"{topic}_{cycle_idx:06d}"
            self.knowledge_base[key] = acquired
            added = True

        if len(self.knowledge_base) > int(self.cfg.max_knowledge_items * self.cfg.compression_threshold):
            compressed: Dict[str, Any] = {}
            for key, val in self.knowledge_base.items():
                compressed[key] = self.compressor.compress(val) if isinstance(val, dict) else val
            self.knowledge_base = compressed

        if len(self.knowledge_base) > self.cfg.max_knowledge_items:
            keys = sorted(self.knowledge_base.keys())
            drop = len(self.knowledge_base) - self.cfg.max_knowledge_items
            for key in keys[:drop]:
                self.knowledge_base.pop(key, None)

        return {
            "topic": topic,
            "topic_plan": topic_plan,
            "added": added,
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
        }

    def _quantum_step(self, cycle_idx: int) -> Dict[str, Any]:
        horizon = 2.0 + 0.03 * float(cycle_idx)
        t = np.linspace(0.0, horizon, self.cfg.time_points)

        gamma = self.cfg.gamma_base * (1.0 + 0.01 * min(cycle_idx, 300))
        lambda_threshold = self.cfg.lambda_threshold * (1.0 + 0.02 * math.sin(cycle_idx / 13.0))

        witness = self.quantum_sim.dual_complex_evolution(
            t,
            gamma_decoherence=gamma,
            Lambda_threshold=lambda_threshold,
            formula_mode=self.cfg.formula_mode,
        )
        w = np.asarray(witness, dtype=np.float64)

        w_min = float(np.min(w))
        w_max = float(np.max(w))
        w_mean = float(np.mean(w))
        w_std = float(np.std(w))
        q25, q50, q75 = [float(x) for x in np.quantile(w, [0.25, 0.5, 0.75])]

        centered = w - w_mean
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        if spectrum.size <= 2:
            spec_low = 0.0
            spec_high = 0.0
        else:
            total = float(np.sum(spectrum[1:]))
            if total <= 1e-20:
                spec_low = 0.0
                spec_high = 0.0
            else:
                split = max(2, spectrum.size // 4)
                spec_low = float(np.sum(spectrum[1:split]) / total)
                spec_high = float(np.sum(spectrum[split:]) / total)

        entanglement_ratio = float(np.mean(w < 0.0))

        stride = max(1, int(w.size // 16))
        sample = [float(v) for v in w[::stride][:16]]

        return {
            "horizon": horizon,
            "gamma": gamma,
            "lambda_threshold": lambda_threshold,
            "witness_min": w_min,
            "witness_max": w_max,
            "witness_mean": w_mean,
            "witness_std": w_std,
            "witness_q25": q25,
            "witness_q50": q50,
            "witness_q75": q75,
            "witness_spectral_low": spec_low,
            "witness_spectral_high": spec_high,
            "entanglement_negative_ratio": entanglement_ratio,
            "entanglement_detected": bool(w_min < 0.0),
            "witness_sample": sample,
        }

    def _run_capability_test_with_timeout(self) -> Dict[str, Any]:
        timeout_s = max(0.2, float(self.cfg.capability_timeout_seconds))
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.tester.run_comprehensive_test)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"capability test timeout after {timeout_s:.1f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _capability_step(self, cycle_idx: int) -> Dict[str, Any]:
        interval = max(
            1,
            int(
                self.strategy_capability_interval_override
                if self.strategy_capability_interval_override is not None
                else self.cfg.capability_check_every
            ),
        )
        forced_check = bool(self.force_capability_check_next_cycle)
        self.force_capability_check_next_cycle = False
        # Always run at least one capability check on the first cycle so
        # short smoke runs still produce meaningful capability evidence.
        do_check = forced_check or (cycle_idx == 1) or ((cycle_idx % interval) == 0)
        if not do_check:
            return {
                "checked": False,
                "overall_score": self.latest_capability_score * 100.0,
                "forced": False,
                    "interval": int(interval),
            }

        try:
            result = self._run_capability_test_with_timeout()
        except Exception as exc:
            self.capability_check_failures += 1
            return {
                "checked": False,
                "overall_score": self.latest_capability_score * 100.0,
                "check_error": str(exc),
                "forced": bool(forced_check),
                    "interval": int(interval),
            }

        score = float(result.get("overall_score", 0.0))
        self.latest_capability_score = score / 100.0
        self.capability_checked_cycles += 1

        return {
            "checked": True,
            "overall_score": score,
            "grade": result.get("grade", "unknown"),
            "tests": {k: float(v.get("score", 0.0)) for k, v in result.get("tests", {}).items()},
            "forced": bool(forced_check),
                    "interval": int(interval),
        }

    def _build_feature_vector(
        self,
        *,
        cycle_idx: int,
        quantum: Dict[str, Any],
        capability: Dict[str, Any],
        knowledge: Dict[str, Any],
    ) -> np.ndarray:
        cap_norm = float(np.clip(capability.get("overall_score", 0.0) / 100.0, 0.0, 1.0))
        knowledge_norm = float(
            np.clip(
                np.log1p(float(knowledge["knowledge_count"])) / np.log1p(float(self.cfg.max_knowledge_items)),
                0.0,
                1.0,
            )
        )

        attempts = max(1, int(knowledge.get("acquired_count", 0)) + int(knowledge.get("failed_count", 0)))
        acquisition_success = float(int(knowledge.get("acquired_count", 0)) / attempts)

        phase = float(cycle_idx) / max(1.0, float(self.cfg.capability_check_every))

        features = np.array(
            [
                float(quantum["witness_min"]),
                float(quantum["witness_max"]),
                float(quantum["witness_mean"]),
                float(quantum["witness_std"]),
                float(quantum["witness_q25"]),
                float(quantum["witness_q50"]),
                float(quantum["witness_q75"]),
                float(quantum["witness_spectral_low"]),
                float(quantum["witness_spectral_high"]),
                float(quantum["entanglement_negative_ratio"]),
                float(quantum["gamma"]),
                float(quantum["lambda_threshold"]),
                float(quantum["horizon"]),
                cap_norm,
                knowledge_norm,
                acquisition_success,
                math.sin(phase),
                math.cos(phase),
                float(self.latest_base_score),
                float(self.latest_enhanced_score),
            ],
            dtype=np.float64,
        )

        if features.shape[0] != self.feature_dim:
            raise RuntimeError(f"feature dimension mismatch: got {features.shape[0]}, expected {self.feature_dim}")

        scale = np.maximum(1.0, np.std(features))
        return features / scale

    def _parallel_projection_step(self, features: np.ndarray) -> Dict[str, Any]:
        worker_count = min(self.cfg.parallel_branches, max(1, int(self.parallel_worker_limit)))

        args_list = [
            (idx, features, self.branch_matrices[idx], self.branch_bias[idx])
            for idx in range(self.cfg.parallel_branches)
        ]

        metrics: List[Dict[str, float]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_evaluate_projection_branch, branch_id, feat, mat, bias)
                for (branch_id, feat, mat, bias) in args_list
            ]
            for f in futures:
                metrics.append(f.result())

        metrics = sorted(metrics, key=lambda x: x["branch"])
        scores = np.asarray([m["score"] for m in metrics], dtype=np.float64)
        coherences = np.asarray([m["coherence"] for m in metrics], dtype=np.float64)
        entropies = np.asarray([m["entropy"] for m in metrics], dtype=np.float64)

        top_idx = int(np.argmax(scores)) if scores.size > 0 else 0

        return {
            "consensus_score": float(np.mean(scores)) if scores.size > 0 else 0.0,
            "branch_disagreement": float(np.std(scores)) if scores.size > 0 else 0.0,
            "top_branch": top_idx,
            "top_score": float(scores[top_idx]) if scores.size > 0 else 0.0,
            "mean_coherence": float(np.mean(coherences)) if coherences.size > 0 else 0.0,
            "mean_entropy": float(np.mean(entropies)) if entropies.size > 0 else 0.0,
            "branch_scores": [float(s) for s in scores.tolist()],
            "parallel_branches": int(self.cfg.parallel_branches),
            "projection_dim": int(self.cfg.projection_dim),
        }

    def _composite_score(
        self,
        *,
        quantum: Dict[str, Any],
        capability: Dict[str, Any],
        knowledge: Dict[str, Any],
        highdim: Dict[str, Any],
    ) -> Tuple[float, float]:
        witness_quality = float(np.clip(-quantum["witness_min"], 0.0, 1.0))
        witness_stability = 1.0 / (1.0 + max(0.0, quantum["witness_std"]))
        capability_norm = float(np.clip(capability.get("overall_score", 0.0) / 100.0, 0.0, 1.0))
        knowledge_norm = float(np.clip(knowledge["knowledge_count"] / 800.0, 0.0, 1.0))
        entanglement_ratio = float(np.clip(quantum["entanglement_negative_ratio"], 0.0, 1.0))

        base_score = (
            0.32 * witness_quality
            + 0.20 * witness_stability
            + 0.23 * capability_norm
            + 0.10 * knowledge_norm
            + 0.15 * entanglement_ratio
        )

        enhanced_score = (
            0.72 * base_score
            + 0.28 * float(highdim.get("consensus_score", 0.0))
            - 0.05 * float(np.clip(highdim.get("branch_disagreement", 0.0), 0.0, 1.0))
        )

        return float(np.clip(base_score, 0.0, 1.0)), float(np.clip(enhanced_score, 0.0, 1.0))

    def run(self, daemon: Any) -> Dict[str, Any]:
        print("=" * 72)
        print("High-dimensional Quantum AGI runner started")
        print(f"Output dir: {self.output_dir}")
        print(f"Target duration: {self.cfg.hours:.2f} hours")
        print(f"Resume enabled: {self.cfg.resume}")
        if self.resumed:
            print(f"Resumed from existing history: {self.resumed_from_cycle} cycles")
        print("=" * 72)

        cycle = len(self.cycles)

        while not self.stop_requested:
            elapsed_h = self._elapsed_hours()
            if elapsed_h >= self.cfg.hours:
                break
            if self.cfg.max_cycles > 0 and cycle >= self.cfg.max_cycles:
                break

            cycle += 1
            cycle_started = time.time()

            try:
                knowledge = self._knowledge_step(cycle)
                quantum = self._quantum_step(cycle)
                capability = self._capability_step(cycle)
                features = self._build_feature_vector(
                    cycle_idx=cycle,
                    quantum=quantum,
                    capability=capability,
                    knowledge=knowledge,
                )
                highdim = self._parallel_projection_step(features)

                base_score, enhanced_score = self._composite_score(
                    quantum=quantum,
                    capability=capability,
                    knowledge=knowledge,
                    highdim=highdim,
                )
                self.latest_base_score = base_score
                self.latest_enhanced_score = enhanced_score

                control: Dict[str, Any] = {
                    "important_cycle": False,
                    "forced_prompt_written": False,
                    "force_capability_next_cycle": bool(self.force_capability_check_next_cycle),
                }
                if bool(self.cfg.force_acceptance_prompt):
                    important_cycle = self._is_important_cycle(cycle, capability, enhanced_score)
                    control["important_cycle"] = bool(important_cycle)
                    if important_cycle:
                        forced_payload = self._build_forced_acceptance_prompt(
                            cycle_idx=cycle,
                            enhanced_score=enhanced_score,
                            capability=capability,
                            quantum=quantum,
                            highdim=highdim,
                        )
                        self.forced_prompt_count += 1
                        self.force_capability_check_next_cycle = bool(
                            forced_payload.get("force_capability_next_cycle", False)
                        )
                        strategy_applied = self._apply_prompt_driven_strategy(
                            forced_payload.get("strategy", {}) if isinstance(forced_payload, dict) else {}
                        )
                        control["forced_prompt_written"] = True
                        control["force_capability_next_cycle"] = bool(self.force_capability_check_next_cycle)
                        control["gaps"] = list(forced_payload.get("gaps", []))
                        control["actions"] = list(forced_payload.get("actions", []))
                        control["prompt"] = str(forced_payload.get("prompt", ""))
                        control["strategy"] = strategy_applied

                record = {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_hours": elapsed_h,
                    "quantum": quantum,
                    "capability": capability,
                    "knowledge": knowledge,
                    "highdim": highdim,
                    "control": control,
                    "base_score": base_score,
                    "composite_score": enhanced_score,
                }
                self.cycles.append(record)
                
                # Phase 1 Track A: Push composite score to uplift window and check for slope-alarm
                self.uplift_window.push_value(enhanced_score, cycle)
                if self.uplift_window.is_alarm_triggered(cycle):
                    # Slope-alarm triggered: extend strategy if active
                    if self.strategy_cycles_left > 0:
                        self.strategy_persistence.extend_strategy(additional_cycles=2)
                        self.strategy_cycles_left += 2
                        if len(self.strategy_topic_boost) > 0:
                            logger.info(f"[PHASE1-ALARM] cycle={cycle} strategy extended")
                    else:
                        logger.info(f"[PHASE1-ALARM] cycle={cycle} alarm triggered but no active strategy")
                
                uplift_state = self._update_uplift_alert_state()
                control["slope_alert_active"] = bool(uplift_state.get("slope_alert_active", False))
                control["strategy_hold_cross_window"] = bool(uplift_state.get("strategy_hold_cross_window", False))
                control["uplift_positive_streak"] = int(uplift_state.get("uplift_positive_streak", 0))
                control["uplift_window_size"] = int(uplift_state.get("window_size", 1))
                control["uplift_value"] = float(uplift_state.get("uplift", 0.0))
                control["recovery_released"] = bool(uplift_state.get("recovery_released", False))
                self._write_cycle(record)
                self._save_state()
                self._decay_prompt_driven_strategy()

                daemon.report_task_complete()

                print(
                    f"[cycle {cycle:05d}] witness_min={quantum['witness_min']:.4f} "
                    f"cap={capability.get('overall_score', 0.0):.1f}% "
                    f"consensus={highdim['consensus_score']:.4f} "
                    f"composite={enhanced_score:.4f} "
                    f"important={int(bool(control.get('important_cycle')))} "
                    f"forced={int(bool(control.get('forced_prompt_written')))}"
                )

            except Exception as exc:
                daemon.report_error()
                print(f"[cycle {cycle:05d}] error: {exc}")

            spent = time.time() - cycle_started
            sleep_s = max(1, self.cfg.cycle_seconds - int(spent))
            for _ in range(sleep_s):
                if self.stop_requested:
                    break
                time.sleep(1)

        return self._finalize()

    def _build_acceptance(self, report: Dict[str, Any]) -> Dict[str, Any]:
        composite_values = [float(c.get("composite_score", 0.0)) for c in self.cycles]
        window = max(1, len(composite_values) // 4)

        initial_mean = float(np.mean(composite_values[:window])) if composite_values else 0.0
        final_mean = float(np.mean(composite_values[-window:])) if composite_values else 0.0
        uplift = final_mean - initial_mean

        metrics = report.get("metrics", {})
        composite_mean = float(metrics.get("enhanced_composite_ci95", {}).get("mean", 0.0))
        cap_mean = float(metrics.get("capability_score_ci95", {}).get("mean", 0.0))
        cap_measurements = int(metrics.get("capability_measurements", 0))
        forced_prompt_count = int(metrics.get("forced_prompt_count", 0))
        ent_mean = float(metrics.get("entanglement_ratio_ci95", {}).get("mean", 0.0))
        highdim_mean = float(metrics.get("highdim_consensus_ci95", {}).get("mean", 0.0))

        smoke_like_run = bool(
            (self.cfg.max_cycles > 0 and self.cfg.max_cycles < self.cfg.acceptance_min_cycles)
            or (self._elapsed_hours() < 0.05 and len(self.cycles) < self.cfg.acceptance_min_cycles)
        )

        criteria = [
            {
                "name": "minimum_cycles",
                "value": int(len(self.cycles)),
                "threshold": int(self.cfg.acceptance_min_cycles),
                "passed": bool(len(self.cycles) >= self.cfg.acceptance_min_cycles),
            },
            {
                "name": "enhanced_composite_mean",
                "value": composite_mean,
                "threshold": float(self.cfg.acceptance_composite_min),
                "passed": bool(composite_mean >= self.cfg.acceptance_composite_min),
            },
            {
                "name": "capability_measurements",
                "value": cap_measurements,
                "threshold": int(self.cfg.acceptance_min_capability_measurements),
                "passed": bool(cap_measurements >= self.cfg.acceptance_min_capability_measurements),
            },
            {
                "name": "capability_score_mean",
                "value": cap_mean,
                "threshold": float(self.cfg.acceptance_capability_min),
                "passed": bool(cap_mean >= self.cfg.acceptance_capability_min),
            },
            {
                "name": "entanglement_ratio_mean",
                "value": ent_mean,
                "threshold": float(self.cfg.acceptance_entanglement_min),
                "passed": bool(ent_mean >= self.cfg.acceptance_entanglement_min),
            },
            {
                "name": "highdim_consensus_mean",
                "value": highdim_mean,
                "threshold": float(self.cfg.acceptance_highdim_consensus_min),
                "passed": bool(highdim_mean >= self.cfg.acceptance_highdim_consensus_min),
            },
            {
                "name": "composite_uplift",
                "value": uplift,
                "threshold": float(self.cfg.acceptance_uplift_min),
                "passed": bool(uplift >= self.cfg.acceptance_uplift_min),
            },
        ]

        if bool(self.cfg.strict_acceptance):
            criteria.extend(
                [
                    {
                        "name": "forced_acceptance_prompts",
                        "value": int(forced_prompt_count),
                        "threshold": int(self.cfg.acceptance_min_forced_prompts),
                        "passed": bool(forced_prompt_count >= self.cfg.acceptance_min_forced_prompts),
                    },
                    {
                        "name": "non_smoke_run",
                        "value": int(not smoke_like_run),
                        "threshold": 1,
                        "passed": bool(not smoke_like_run),
                    },
                ]
            )

        passed = all(bool(c["passed"]) for c in criteria)

        recommendations = []
        if not passed:
            for item in criteria:
                if item["passed"]:
                    continue
                recommendations.append(
                    f"Raise {item['name']} above threshold ({item['value']:.6f} < {item['threshold']:.6f})"
                    if isinstance(item["value"], float)
                    else f"Raise {item['name']} above threshold ({item['value']} < {item['threshold']})"
                )

        acceptance = {
            "timestamp": datetime.now().isoformat(),
            "passed": passed,
            "strict_acceptance": bool(self.cfg.strict_acceptance),
            "run_scope": "smoke" if smoke_like_run else "full",
            "criteria": criteria,
            "trend": {
                "initial_window_mean": initial_mean,
                "final_window_mean": final_mean,
                "uplift": uplift,
                "window_size": window,
            },
            "recommendations": recommendations,
        }

        self.acceptance_json_path.write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lines = [
            "# High-Dimensional Quantum AGI Acceptance",
            "",
            f"- timestamp: {acceptance['timestamp']}",
            f"- passed: {acceptance['passed']}",
            "",
            "## Criteria",
            "",
        ]

        for item in criteria:
            lines.append(
                f"- {item['name']}: value={item['value']} threshold={item['threshold']} passed={item['passed']}"
            )

        lines.extend(
            [
                "",
                "## Trend",
                "",
                f"- initial_window_mean: {initial_mean:.6f}",
                f"- final_window_mean: {final_mean:.6f}",
                f"- uplift: {uplift:.6f}",
            ]
        )

        if recommendations:
            lines.extend(["", "## Recommendations", ""])
            for rec in recommendations:
                lines.append(f"- {rec}")

        self.acceptance_md_path.write_text("\n".join(lines), encoding="utf-8")

        return acceptance

    def _finalize(self) -> Dict[str, Any]:
        witness_min_vals = [float(c["quantum"]["witness_min"]) for c in self.cycles]
        ent_ratio_vals = [float(c["quantum"]["entanglement_negative_ratio"]) for c in self.cycles]
        base_vals = [float(c.get("base_score", 0.0)) for c in self.cycles]
        enhanced_vals = [float(c.get("composite_score", 0.0)) for c in self.cycles]
        highdim_consensus_vals = [float(c["highdim"].get("consensus_score", 0.0)) for c in self.cycles]
        highdim_disagree_vals = [float(c["highdim"].get("branch_disagreement", 0.0)) for c in self.cycles]

        capability_scores = [
            float(c["capability"].get("overall_score", 0.0))
            for c in self.cycles
            if bool(c["capability"].get("checked"))
        ]

        report = {
            "started_at": datetime.fromtimestamp(self.start_ts).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_hours": self._elapsed_hours(),
            "cycles": len(self.cycles),
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
            "capability_checked_cycles": int(self.capability_checked_cycles),
            "capability_check_failures": int(self.capability_check_failures),
            "baselines": self.baseline_snapshot,
            "config": {
                "projection_dim": self.cfg.projection_dim,
                "parallel_branches": self.cfg.parallel_branches,
                "projection_seed": self.cfg.projection_seed,
                "formula_mode": self.cfg.formula_mode,
                "resource_profile_requested": self.resource_profile_requested,
                "resource_profile_applied": self.resource_profile_applied,
                "parallel_worker_limit": int(self.parallel_worker_limit),
                "capability_timeout_seconds": float(self.cfg.capability_timeout_seconds),
                "important_cycle_every": int(self.cfg.important_cycle_every),
                "force_acceptance_prompt": bool(self.cfg.force_acceptance_prompt),
            },
            "metrics": {
                "witness_min_ci95": _bootstrap_ci(witness_min_vals, seed=20260406),
                "entanglement_ratio_ci95": _bootstrap_ci(ent_ratio_vals, seed=20260407),
                "base_composite_ci95": _bootstrap_ci(base_vals, seed=20260408),
                "enhanced_composite_ci95": _bootstrap_ci(enhanced_vals, seed=20260409),
                "highdim_consensus_ci95": _bootstrap_ci(highdim_consensus_vals, seed=20260410),
                "highdim_disagreement_ci95": _bootstrap_ci(highdim_disagree_vals, seed=20260411),
                "capability_score_ci95": _bootstrap_ci(capability_scores, seed=20260412),
                "capability_measurements": int(len(capability_scores)),
                "capability_check_failures": int(self.capability_check_failures),
                "forced_prompt_count": int(self.forced_prompt_count),
            },
            "resume": {
                "enabled": self.cfg.resume,
                "resumed": self.resumed,
                "resumed_from_cycle": self.resumed_from_cycle,
                "resumed_at": self.resumed_at,
            },
            "latest_capability_score": self.latest_capability_score,
            "latest_base_score": self.latest_base_score,
            "latest_enhanced_score": self.latest_enhanced_score,
        }

        acceptance = self._build_acceptance(report)
        report["acceptance"] = acceptance

        self.report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics = report["metrics"]
        md = [
            "# High-Dimensional Quantum AGI Report",
            "",
            f"- started_at: {report['started_at']}",
            f"- finished_at: {report['finished_at']}",
            f"- elapsed_hours: {report['elapsed_hours']:.6f}",
            f"- cycles: {report['cycles']}",
            f"- knowledge_count: {report['knowledge_count']}",
            f"- capability_checked_cycles: {report['capability_checked_cycles']}",
            f"- capability_check_failures: {report['capability_check_failures']}",
            f"- acceptance_passed: {acceptance['passed']}",
            f"- acceptance_scope: {acceptance['run_scope']}",
            f"- strict_acceptance: {acceptance['strict_acceptance']}",
            "",
            "## CI Metrics (95%)",
            "",
            f"- capability_measurements: {metrics['capability_measurements']}",
            f"- capability_check_failures: {metrics['capability_check_failures']}",
            f"- forced_prompt_count: {metrics['forced_prompt_count']}",
            (
                "- witness_min mean/lower/upper: "
                f"{metrics['witness_min_ci95']['mean']:.6f} / "
                f"{metrics['witness_min_ci95']['lower']:.6f} / "
                f"{metrics['witness_min_ci95']['upper']:.6f}"
            ),
            (
                "- entanglement_ratio mean/lower/upper: "
                f"{metrics['entanglement_ratio_ci95']['mean']:.6f} / "
                f"{metrics['entanglement_ratio_ci95']['lower']:.6f} / "
                f"{metrics['entanglement_ratio_ci95']['upper']:.6f}"
            ),
            (
                "- base_composite mean/lower/upper: "
                f"{metrics['base_composite_ci95']['mean']:.6f} / "
                f"{metrics['base_composite_ci95']['lower']:.6f} / "
                f"{metrics['base_composite_ci95']['upper']:.6f}"
            ),
            (
                "- enhanced_composite mean/lower/upper: "
                f"{metrics['enhanced_composite_ci95']['mean']:.6f} / "
                f"{metrics['enhanced_composite_ci95']['lower']:.6f} / "
                f"{metrics['enhanced_composite_ci95']['upper']:.6f}"
            ),
            (
                "- highdim_consensus mean/lower/upper: "
                f"{metrics['highdim_consensus_ci95']['mean']:.6f} / "
                f"{metrics['highdim_consensus_ci95']['lower']:.6f} / "
                f"{metrics['highdim_consensus_ci95']['upper']:.6f}"
            ),
            (
                "- capability_score mean/lower/upper: "
                f"{metrics['capability_score_ci95']['mean']:.6f} / "
                f"{metrics['capability_score_ci95']['lower']:.6f} / "
                f"{metrics['capability_score_ci95']['upper']:.6f}"
            ),
        ]

        self.report_md_path.write_text("\n".join(md), encoding="utf-8")
        self._save_state()

        return report


def _parse_china_mode(mode: str) -> Optional[bool]:
    mode = mode.strip().lower()
    if mode == "auto":
        return None
    if mode in {"on", "true", "1"}:
        return True
    if mode in {"off", "false", "0"}:
        return False
    return None


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-dimensional quantum AGI evolution runner")

    parser.add_argument("--hours", type=float, default=0.5, help="Target runtime in hours")
    parser.add_argument("--cycle-seconds", type=int, default=120, help="Seconds per evolution cycle")
    parser.add_argument("--time-points", type=int, default=220, help="Time samples per quantum simulation")
    parser.add_argument("--capability-check-every", type=int, default=3, help="Run capability test every N cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional hard cap for cycles (0 means unlimited)")
    parser.add_argument("--max-knowledge-items", type=int, default=2500, help="Knowledge base size limit")
    parser.add_argument("--compression-threshold", type=float, default=0.80, help="Compression threshold ratio")

    parser.add_argument("--mass-kg", type=float, default=1e-14, help="Quantum simulator mass parameter")
    parser.add_argument("--distance-m", type=float, default=35e-6, help="Quantum simulator distance parameter")
    parser.add_argument("--gamma-base", type=float, default=1e-3, help="Base decoherence gamma")
    parser.add_argument("--lambda-threshold", type=float, default=4.0, help="Quantum witness lock threshold")
    parser.add_argument("--formula-mode", choices=["legacy", "aligned"], default="aligned", help="Witness formula mode")

    parser.add_argument("--projection-dim", type=int, default=128, help="Projected high-dimensional latent size")
    parser.add_argument("--parallel-branches", type=int, default=4, help="Parallel projection branches")
    parser.add_argument("--projection-seed", type=int, default=260401249, help="Random seed for projection matrices")
    parser.add_argument("--resource-profile", choices=["auto", "low", "balanced", "high"], default="auto", help="Resource profile for local machine")
    parser.add_argument("--max-parallel-workers", type=int, default=0, help="Hard cap for projection workers (0 means auto)")
    parser.add_argument("--capability-timeout-seconds", type=float, default=25.0, help="Timeout for one capability test run")
    parser.add_argument("--important-cycle-every", type=int, default=3, help="Force acceptance prompt every N cycles")
    parser.add_argument("--force-acceptance-prompt", choices=["on", "off"], default="on", help="Emit forced acceptance prompt on important cycles")
    parser.add_argument("--strict-acceptance", choices=["on", "off"], default="on", help="Reject smoke-like or over-relaxed runs as final acceptance")

    parser.add_argument("--accept-min-cycles", type=int, default=12, help="Minimum cycles for acceptance")
    parser.add_argument("--accept-composite-min", type=float, default=0.58, help="Minimum enhanced composite mean")
    parser.add_argument("--accept-capability-min", type=float, default=60.0, help="Minimum capability score mean")
    parser.add_argument("--accept-entanglement-min", type=float, default=0.12, help="Minimum entanglement ratio mean")
    parser.add_argument("--accept-highdim-consensus-min", type=float, default=0.55, help="Minimum high-dimensional consensus mean")
    parser.add_argument("--accept-uplift-min", type=float, default=0.02, help="Minimum composite uplift")
    parser.add_argument("--accept-min-capability-measurements", type=int, default=2, help="Minimum capability measurements for final acceptance")
    parser.add_argument("--accept-min-forced-prompts", type=int, default=2, help="Minimum forced acceptance prompts for final acceptance")

    parser.add_argument("--resume", action="store_true", help="Resume from existing state/cycle files in output dir")
    parser.add_argument("--china-mode", choices=["auto", "on", "off"], default="auto", help="Knowledge source mode")

    parser.add_argument(
        "--output-dir",
        default="h2q_project/reports/quantum_agi_highdim_evolution",
        help="Output directory (or base dir when separate-run-dir=on)",
    )
    parser.add_argument("--separate-run-dir", choices=["on", "off"], default="off", help="Use a timestamped subdirectory under output-dir")
    parser.add_argument("--run-name", default="", help="Optional run directory name when separate-run-dir=on")
    parser.add_argument("--latest-link-name", default="latest", help="Symlink name under output-dir base that points to current run dir")

    parser.add_argument("--daemon-heartbeat", type=int, default=30, help="Survival daemon heartbeat interval")
    parser.add_argument("--daemon-timeout", type=int, default=150, help="Survival daemon max no-heartbeat seconds")
    parser.add_argument("--daemon-restart-cooldown", type=int, default=60, help="Survival daemon restart cooldown")

    return parser.parse_args(argv)


def _resolve_output_dir(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = (PROJECT_ROOT / base_output_dir).resolve()
    else:
        base_output_dir = base_output_dir.resolve()

    if args.separate_run_dir == "off":
        return base_output_dir, None

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return (base_output_dir / run_name).resolve(), base_output_dir


def _update_latest_symlink(base_output_dir: Path, output_dir: Path, link_name: str) -> None:
    name = link_name.strip()
    if not name:
        return

    base_output_dir.mkdir(parents=True, exist_ok=True)
    link_path = base_output_dir / name

    try:
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            print(f"Warning: skip latest link update because path exists and is not symlink: {link_path}")
            return

        link_path.symlink_to(output_dir, target_is_directory=True)
    except Exception as exc:
        print(f"Warning: failed to update latest link {link_path}: {exc}")


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    output_dir, base_output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_output_dir is not None:
        _update_latest_symlink(base_output_dir, output_dir, args.latest_link_name)

    cfg = HighDimRunnerConfig(
        hours=max(0.01, args.hours),
        cycle_seconds=max(5, args.cycle_seconds),
        time_points=max(40, args.time_points),
        capability_check_every=max(1, args.capability_check_every),
        max_cycles=max(0, args.max_cycles),
        max_knowledge_items=max(100, args.max_knowledge_items),
        compression_threshold=float(np.clip(args.compression_threshold, 0.2, 0.98)),
        mass_kg=max(1e-20, args.mass_kg),
        distance_m=max(1e-12, args.distance_m),
        gamma_base=max(1e-8, args.gamma_base),
        lambda_threshold=max(1e-6, args.lambda_threshold),
        formula_mode=args.formula_mode,
        resume=bool(args.resume),
        output_dir=output_dir,
        projection_dim=max(16, args.projection_dim),
        parallel_branches=max(2, args.parallel_branches),
        projection_seed=int(args.projection_seed),
        resource_profile=str(args.resource_profile).strip().lower() or "auto",
        max_parallel_workers=max(0, int(args.max_parallel_workers)),
        capability_timeout_seconds=max(0.2, float(args.capability_timeout_seconds)),
        important_cycle_every=max(1, int(args.important_cycle_every)),
        force_acceptance_prompt=(str(args.force_acceptance_prompt).strip().lower() != "off"),
        acceptance_min_cycles=max(1, args.accept_min_cycles),
        acceptance_composite_min=float(np.clip(args.accept_composite_min, 0.0, 1.0)),
        acceptance_capability_min=max(0.0, args.accept_capability_min),
        acceptance_entanglement_min=float(np.clip(args.accept_entanglement_min, 0.0, 1.0)),
        acceptance_highdim_consensus_min=float(np.clip(args.accept_highdim_consensus_min, 0.0, 1.0)),
        acceptance_uplift_min=max(-1.0, args.accept_uplift_min),
        acceptance_min_capability_measurements=max(1, int(args.accept_min_capability_measurements)),
        acceptance_min_forced_prompts=max(1, int(args.accept_min_forced_prompts)),
        strict_acceptance=(str(args.strict_acceptance).strip().lower() != "off"),
    )

    runner = HighDimQuantumAGIRunner(cfg, china_mode=_parse_china_mode(args.china_mode))

    daemon_cfg = SurvivalConfig(
        heartbeat_interval=max(5, args.daemon_heartbeat),
        max_no_heartbeat=max(30, args.daemon_timeout),
        restart_cooldown=max(10, args.daemon_restart_cooldown),
        state_file=str((output_dir / "quantum_agi_highdim_survival_state.json").resolve()),
        heartbeat_file=str((output_dir / "quantum_agi_highdim_heartbeat.json").resolve()),
        log_file=str((output_dir / "quantum_agi_highdim_survival.log").resolve()),
    )
    daemon = create_survival_daemon(work_dir=str(PROJECT_ROOT), config=daemon_cfg)

    def _capability_cb() -> float:
        return float(100.0 * runner.latest_enhanced_score)

    def _restart_cb() -> None:
        runner.quantum_sim = STQ_QuantumSimulator(mass_kg=cfg.mass_kg, distance_m=cfg.distance_m)

    daemon.set_capability_callback(_capability_cb)
    daemon.set_restart_callback(_restart_cb)

    def _signal_handler(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}, requesting graceful stop...")
        runner.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    daemon.start()
    exit_code = 0

    try:
        report = runner.run(daemon)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        daemon.report_error()
        print(f"Fatal error: {exc}")
        exit_code = 2
    finally:
        daemon.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
