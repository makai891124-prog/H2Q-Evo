#!/usr/bin/env python3
"""Fuse karpathy/autoresearch-style loop into H2Q-Evo self-improvement bootstrap.

Design goals:
1) Reuse H2Q-Evo local artifacts and gates (no destructive git operations).
2) Keep/discard/crash loop inspired by autoresearch results tracking.
3) Produce machine-readable experiment ledger and next-step plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.axiom_consistency import evaluate_axiom_consistency, load_axiom_contract
except Exception:
    # Support direct execution from within tools/ when package-style import is unavailable.
    from axiom_consistency import evaluate_axiom_consistency, load_axiom_contract

REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


@dataclass
class ExperimentSpec:
    name: str
    description: str
    cmd: List[str]
    metric_name: str
    metric_reader: Callable[[], Optional[float]]
    higher_is_better: bool = True


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def read_distill_delta() -> Optional[float]:
    p = REPORTS / "self_eval_distillation_pipeline_latest.json"
    obj = _load_json(p)
    if not obj:
        return None
    return _f((obj.get("metrics") or {}).get("delta_schema_valid_rate", 0.0))


def read_research_aggregate() -> Optional[float]:
    p = REPORTS / "research_aggregation_cross_validation_latest.json"
    obj = _load_json(p)
    if not obj:
        return None
    # Historical artifacts may use either aggregate or aggregate_effectiveness.
    score = (obj.get("aggregate_effectiveness") or {}).get("score", None)
    if score is None:
        score = (obj.get("aggregate") or {}).get("score", 0.0)
    return _f(score, 0.0)


def read_systemic_score() -> Optional[float]:
    p = REPORTS / "systemic_platform_joint_capability_latest.json"
    obj = _load_json(p)
    if not obj:
        return None
    score = (obj.get("aggregate_effectiveness") or {}).get("score", None)
    if score is None:
        score = (obj.get("aggregate") or {}).get("score", 0.0)
    return _f(score, 0.0)


def read_trusted_weight_training_signal() -> Dict[str, Any]:
    """Read trusted weight-training loss improvement from latest LoRA/LM report.

    Positive value means loss decreased during weight training.
    """
    candidates = [
        REPORTS / "trusted_nano_lora_training_latest.json",
        REPORTS / "trusted_nano_lm_training_latest.json",
    ]
    obj: Dict[str, Any] = {}
    source = ""
    for p in candidates:
        o = _load_json(p)
        if o:
            obj = o
            source = str(p)
            break

    li = _f(obj.get("loss_initial", 0.0), 0.0)
    lf = _f(obj.get("loss_final", li), li)
    gain = 0.0
    if abs(li) > 1e-12:
        gain = (li - lf) / abs(li)

    # Keep bounded for stable signal scaling.
    gain = max(-1.0, min(1.0, gain))

    return {
        "source": source,
        "loss_initial": li,
        "loss_final": lf,
        "loss_improvement_rate": gain,
    }


def read_lora_replay_signal() -> Dict[str, Any]:
    """Read latest LoRA replay status used as a hard gate in bootstrap keep decisions."""
    p = REPORTS / "trusted_nano_lora_training_latest.json"
    obj = _load_json(p)
    best = obj.get("best_checkpoint") or {}
    replay = best.get("replay") or {}
    quality = replay.get("quality") or {}
    return {
        "source": str(p),
        "exists": bool(obj),
        "replay_pass": bool(best.get("replay_pass", False)),
        "replay_quality_pass": bool(best.get("replay_quality_pass", False)),
        "replay_score": _f(replay.get("score", 0.0), 0.0),
        "replay_quality_score": _f(quality.get("score", 0.0), 0.0),
        "replay_quality_structure_rate": _f(quality.get("structure_rate", 0.0), 0.0),
        "replay_quality_density_rate": _f(quality.get("density_rate", 0.0), 0.0),
        "replay_quality_echo_rate": _f(quality.get("echo_rate", 1.0), 1.0),
        "best_loss": _f(best.get("loss", 0.0), 0.0),
        "best_step": int(best.get("step", 0) or 0),
    }


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    qq = max(0.0, min(1.0, float(q)))
    if len(xs) == 1:
        return xs[0]
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def compute_adaptive_benchmark_gate(
    reports_dir: Path,
    static_threshold: float,
    enabled: bool,
    lookback: int,
    quantile: float,
    safety_floor: float,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "lookback": int(max(1, lookback)),
        "quantile": float(max(0.0, min(1.0, quantile))),
        "safety_floor": float(max(0.0, safety_floor)),
        "static_threshold": float(static_threshold),
        "resolved_threshold": float(static_threshold),
        "history_count": 0,
        "history_used": [],
    }
    if not enabled:
        return info

    candidates: List[Path] = []
    patterns = [
        "local_incremental_benchmark_*.json",
        "local_incremental_benchmark_uplift_before_*.json",
        "local_incremental_benchmark_uplift_after_*.json",
    ]
    for pat in patterns:
        for p in reports_dir.glob(pat):
            if p.name.endswith("_latest.json"):
                continue
            candidates.append(p)

    # Deduplicate and take latest N by modification time.
    unique = {str(p.resolve()): p for p in candidates}
    ordered = sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)

    gains: List[float] = []
    used: List[Dict[str, Any]] = []
    for p in ordered[: max(1, int(lookback))]:
        obj = _load_json(p)
        if not obj:
            continue
        g = _f(obj.get("gain", 0.0), 0.0)
        if not math.isfinite(g):
            continue
        gains.append(g)
        used.append({"file": str(p), "gain": float(g)})

    info["history_count"] = len(gains)
    info["history_used"] = used
    if not gains:
        info["resolved_threshold"] = float(max(static_threshold, safety_floor))
        return info

    qv = _quantile(gains, quantile)
    resolved = max(float(safety_floor), float(qv))
    info["quantile_value"] = float(qv)
    info["resolved_threshold"] = float(resolved)
    return info


def run_incremental_benchmark_probe(py_exec: Path) -> Dict[str, Any]:
    """Run lightweight fixed-seed benchmark probe and return latest gain signals."""
    cmd = [str(py_exec), "tools/run_local_incremental_benchmark.py", "--seed", "42"]
    out = {
        "cmd": cmd,
        "returncode": 1,
        "gain": 0.0,
        "score_base": 0.0,
        "score_adapter": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=240)
        out["returncode"] = int(proc.returncode)
        out["stdout_tail"] = _tail(proc.stdout)
        out["stderr_tail"] = _tail(proc.stderr)
    except Exception as exc:
        out["stderr_tail"] = f"benchmark probe failed: {exc}"
        return out

    latest = _load_json(REPORTS / "local_incremental_benchmark_latest.json")
    out["gain"] = _f(latest.get("gain", 0.0), 0.0)
    out["score_base"] = _f(latest.get("score_base", 0.0), 0.0)
    out["score_adapter"] = _f(latest.get("score_adapter", 0.0), 0.0)
    return out


def parse_autoresearch_tsv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "source": str(path),
            "exists": False,
            "keep_count": 0,
            "discard_count": 0,
            "crash_count": 0,
            "top_keep_descriptions": [],
        }

    keep_rows: List[Dict[str, str]] = []
    discard = 0
    crash = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status == "keep":
                keep_rows.append(row)
            elif status == "discard":
                discard += 1
            elif status == "crash":
                crash += 1

    # Lower val_bpb is better; keep best 5 as inspiration.
    def _bpb(row: Dict[str, str]) -> float:
        try:
            return float(row.get("val_bpb", "999"))
        except Exception:
            return 999.0

    keep_sorted = sorted(keep_rows, key=_bpb)
    top_desc = [str(r.get("description", "")).strip() for r in keep_sorted[:5] if str(r.get("description", "")).strip()]

    return {
        "source": str(path),
        "exists": True,
        "keep_count": len(keep_rows),
        "discard_count": discard,
        "crash_count": crash,
        "top_keep_descriptions": top_desc,
    }


def compute_meaning_components(snapshot: Dict[str, float], experiments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute proxy components for self-bootstrap meaning objective.

    All values are clipped to [0, 1] and derived from local measurable metrics.
    """
    distill = _f(snapshot.get("distill_delta", 0.0), 0.0)
    research = _f(snapshot.get("research_aggregate", 0.0), 0.0)
    systemic = _f(snapshot.get("systemic_score", 0.0), 0.0)

    utility = max(0.0, min(1.0, 0.5 * research + 0.5 * systemic))
    robustness = max(0.0, min(1.0, 0.5 * research + 0.5 * systemic))
    alignment = max(0.0, min(1.0, systemic))

    total = max(1, len(experiments))
    crash_rate = sum(1 for r in experiments if r.get("status") == "crash") / total
    keep_rate = sum(1 for r in experiments if r.get("status") == "keep") / total

    autonomy = max(0.0, min(1.0, 1.0 - crash_rate))
    # Treat positive-distill and keep-rate as the local efficiency proxy.
    distill_signal = max(0.0, min(1.0, distill))
    efficiency = max(0.0, min(1.0, 0.5 * keep_rate + 0.5 * distill_signal))

    return {
        "utility": utility,
        "autonomy": autonomy,
        "robustness": robustness,
        "alignment": alignment,
        "efficiency": efficiency,
    }


def compute_meaning_score(snapshot: Dict[str, float], experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = compute_meaning_components(snapshot, experiments)
    score = (
        0.30 * c["utility"]
        + 0.25 * c["autonomy"]
        + 0.20 * c["robustness"]
        + 0.15 * c["alignment"]
        + 0.10 * c["efficiency"]
    )
    return {
        "score": max(0.0, min(1.0, score)),
        "components": c,
    }


def compute_dual_projection_signals(snapshot: Dict[str, float], omega: float = 0.23) -> Dict[str, float]:
    """Compute AP/GP boundary ratio and dual-conjugate projection acceleration."""
    r0 = 1.0 + _f(snapshot.get("distill_delta", 0.0), 0.0)
    r1 = 1.0 + _f(snapshot.get("research_aggregate", 0.0), 0.0)
    r2 = 1.0 + _f(snapshot.get("systemic_score", 0.0), 0.0)

    g01 = r1 / max(r0, 1e-12)
    g12 = r2 / max(r1, 1e-12)
    boundary_ratio = 0.5 * (g01 + g12)

    def _proj_radius(radius: float, step: float) -> float:
        z_plus = complex(radius * math.cos(omega * step), radius * math.sin(omega * step))
        z_minus = complex(radius * math.cos(-omega * step), radius * math.sin(-omega * step))
        x = (z_plus.real + z_minus.real) / math.sqrt(2.0)
        y = (z_plus.imag - z_minus.imag) / math.sqrt(2.0)
        return math.sqrt(x * x + y * y)

    p0 = _proj_radius(r0, 0.0)
    p1 = _proj_radius(r1, 1.0)
    p2 = _proj_radius(r2, 2.0)
    projection_accel = p2 - 2.0 * p1 + p0

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_alignment = max(0.0, 1.0 - min(1.0, abs(boundary_ratio - phi) / phi))
    accel_score = 0.5 + 0.5 * math.tanh(8.0 * projection_accel)
    geometry_score = max(0.0, min(1.0, 0.65 * phi_alignment + 0.35 * accel_score))

    return {
        "ap_gp_boundary_ratio": boundary_ratio,
        "phi": phi,
        "phi_alignment": phi_alignment,
        "projection_acceleration": projection_accel,
        "geometry_score": geometry_score,
    }


def apply_metric_to_snapshot(snapshot: Dict[str, float], spec_name: str, metric_after: Optional[float]) -> Dict[str, float]:
    out = dict(snapshot)
    if metric_after is None:
        return out
    v = float(metric_after)
    if spec_name == "distillation_uplift":
        out["distill_delta"] = v
    elif spec_name == "research_cross_validation":
        out["research_aggregate"] = v
    elif spec_name == "systemic_joint_capability":
        out["systemic_score"] = v
    return out


def compute_iteration_decision(
    snapshot_before: Dict[str, float],
    snapshot_after: Dict[str, float],
    experiments_so_far: List[Dict[str, Any]],
    delta: float,
    axiom_contract: Dict[str, Any],
    weight_gain_rate: float,
    step_index: int,
    weight_signal_base_weight: float,
    weight_signal_decay: float,
    weight_signal_gate_floor: float,
) -> Dict[str, Any]:
    ms_before = compute_meaning_score(snapshot_before, experiments_so_far)
    ms_after = compute_meaning_score(snapshot_after, experiments_so_far)
    meaning_uplift = _f(ms_after.get("score", 0.0), 0.0) - _f(ms_before.get("score", 0.0), 0.0)

    g_before = compute_dual_projection_signals(snapshot_before)
    g_after = compute_dual_projection_signals(snapshot_after)
    geometry_uplift = _f(g_after.get("geometry_score", 0.0), 0.0) - _f(g_before.get("geometry_score", 0.0), 0.0)

    ax_before = evaluate_axiom_consistency(
        snapshot=snapshot_before,
        signals=g_before,
        meaning_score=_f(ms_before.get("score", 0.0), 0.0),
        experiments=experiments_so_far,
        contract=axiom_contract,
    )
    ax_after = evaluate_axiom_consistency(
        snapshot=snapshot_after,
        signals=g_after,
        meaning_score=_f(ms_after.get("score", 0.0), 0.0),
        experiments=experiments_so_far,
        contract=axiom_contract,
    )
    axiom_uplift = _f(ax_after.get("score", 0.0), 0.0) - _f(ax_before.get("score", 0.0), 0.0)

    delta_signal = 0.5 + 0.5 * math.tanh(220.0 * delta)
    meaning_signal = 0.5 + 0.5 * math.tanh(120.0 * meaning_uplift)
    geometry_signal = 0.5 + 0.5 * math.tanh(120.0 * geometry_uplift)
    axiom_signal = 0.5 + 0.5 * math.tanh(120.0 * axiom_uplift)
    weight_gain_signal = 0.5 + 0.5 * math.tanh(16.0 * _f(weight_gain_rate, 0.0))

    # Gate weight-only uplift by multi-signal consensus and decay it over iteration depth.
    positive_support = 0
    if delta > 0.0:
        positive_support += 1
    if meaning_uplift > 0.0:
        positive_support += 1
    if geometry_uplift > 0.0:
        positive_support += 1
    if axiom_uplift > 0.0:
        positive_support += 1
    support_ratio = positive_support / 4.0
    gate_floor = max(0.0, min(1.0, _f(weight_signal_gate_floor, 0.15)))
    gate_support = gate_floor + (1.0 - gate_floor) * support_ratio
    decay = math.exp(-max(0.0, _f(weight_signal_decay, 0.20)) * max(0, int(step_index) - 1))
    weight_gate = max(0.0, min(1.0, gate_support * decay))

    base_weight = max(0.0, min(0.5, _f(weight_signal_base_weight, 0.15)))
    effective_weight = base_weight * weight_gate

    # Five-way first-class fusion: metric + meaning + geometry + axiom + weight gain.
    # Distill delta is intentionally de-weighted; capability/consistency signals dominate.
    non_weight_sum = 0.15 + 0.30 + 0.20 + 0.20
    remaining = max(0.0, 1.0 - effective_weight)
    k = remaining / max(non_weight_sum, 1e-12)
    decision_score = (
        (0.15 * k) * delta_signal
        + (0.30 * k) * meaning_signal
        + (0.20 * k) * geometry_signal
        + (0.20 * k) * axiom_signal
        + effective_weight * weight_gain_signal
    )
    return {
        "delta_signal": delta_signal,
        "meaning_uplift": meaning_uplift,
        "meaning_signal": meaning_signal,
        "geometry_uplift": geometry_uplift,
        "geometry_signal": geometry_signal,
        "axiom_uplift": axiom_uplift,
        "axiom_signal": axiom_signal,
        "weight_gain_rate": _f(weight_gain_rate, 0.0),
        "weight_gain_signal": weight_gain_signal,
        "weight_gate": weight_gate,
        "weight_support_ratio": support_ratio,
        "weight_decay_factor": decay,
        "weight_effective_weight": effective_weight,
        "decision_score": decision_score,
        "geometry_before": _f(g_before.get("geometry_score", 0.0), 0.0),
        "geometry_after": _f(g_after.get("geometry_score", 0.0), 0.0),
        "boundary_ratio_before": _f(g_before.get("ap_gp_boundary_ratio", 0.0), 0.0),
        "boundary_ratio_after": _f(g_after.get("ap_gp_boundary_ratio", 0.0), 0.0),
        "projection_accel_before": _f(g_before.get("projection_acceleration", 0.0), 0.0),
        "projection_accel_after": _f(g_after.get("projection_acceleration", 0.0), 0.0),
        "axiom_score_before": _f(ax_before.get("score", 0.0), 0.0),
        "axiom_score_after": _f(ax_after.get("score", 0.0), 0.0),
        "axiom_pass_rate_before": _f(ax_before.get("pass_rate", 0.0), 0.0),
        "axiom_pass_rate_after": _f(ax_after.get("pass_rate", 0.0), 0.0),
        "axiom_violations_before": list(ax_before.get("violations", [])),
        "axiom_violations_after": list(ax_after.get("violations", [])),
    }


def build_decision_reason(row: Dict[str, Any]) -> str:
    status = str(row.get("status", "")).strip().lower()
    d = row.get("decision") or {}
    delta = _f(row.get("delta", 0.0), 0.0)
    m_up = _f(d.get("meaning_uplift", 0.0), 0.0)
    g_up = _f(d.get("geometry_uplift", 0.0), 0.0)
    a_up = _f(d.get("axiom_uplift", 0.0), 0.0)
    w_gain = _f(d.get("weight_gain_rate", 0.0), 0.0)
    w_gate = _f(d.get("weight_gate", 0.0), 0.0)

    if status == "crash":
        return "command failed or timed out"

    hg = (d.get("hard_gate") or {}) if isinstance(d, dict) else {}
    if status == "discard" and bool(hg.get("enabled", False)):
        reasons: List[str] = []
        if not bool(hg.get("benchmark_ok", False)):
            reasons.append("hard-gate benchmark gain not met")
        if not bool(hg.get("lora_replay_ok", False)):
            reasons.append("hard-gate lora replay failed")
        if not bool(hg.get("output_quality_ok", False)):
            reasons.append("hard-gate output quality failed")
        if reasons:
            return ", ".join(reasons)

    weak_metric = abs(delta) < 1e-12
    weak_meaning = abs(m_up) < 1e-12
    weak_geometry = abs(g_up) < 1e-12
    weak_axiom = abs(a_up) < 1e-12
    weak_weight = abs(w_gain) < 1e-12

    if status == "keep":
        parts: List[str] = []
        if delta > 0.0:
            parts.append("metric up")
        if m_up > 0.0:
            parts.append("meaning up")
        if g_up > 0.0:
            parts.append("geometry up")
        if a_up > 0.0:
            parts.append("axiom up")
        if w_gain > 0.0:
            if w_gate < 0.25:
                parts.append("weight gain up (gated)")
            else:
                parts.append("weight gain up")
        return ", ".join(parts) if parts else "decision_score above keep threshold"

    # discard reasons
    if weak_metric and weak_meaning and weak_geometry and weak_axiom and weak_weight:
        return "metric no gain and meaning/geometry/axiom/weight flat"
    parts = []
    if delta <= 0.0:
        parts.append("metric not improved")
    if m_up <= 0.0:
        parts.append("meaning not improved")
    if g_up <= 0.0:
        parts.append("geometry not improved")
    if a_up <= 0.0:
        parts.append("axiom not improved")
    if w_gain <= 0.0:
        parts.append("weight gain not improved")
    elif w_gate < 0.25:
        parts.append("weight signal decayed")
    return ", ".join(parts) if parts else "decision_score below keep threshold"


def build_experiments(
    timeout_sec: int,
    distill_sessions: int,
    distill_teacher_provider: str,
    distill_max_samples: int,
    distill_synthetic_min_size: int,
    distill_include_valid_prompts: bool,
    distill_chunks: int,
    distill_chunk_size: int,
    distill_execution_mode: str,
) -> List[ExperimentSpec]:
    if int(distill_chunks) > 1:
        distill_cmd = [
            str(PY),
            "tools/run_self_eval_distillation_pipeline_chunked.py",
            "--chunks",
            str(max(1, int(distill_chunks))),
            "--sessions",
            str(max(1, int(distill_sessions))),
            "--teacher-provider",
            str(distill_teacher_provider),
            "--max-samples-per-chunk",
            str(max(1, int(distill_chunk_size))),
            "--synthetic-min-size-per-chunk",
            str(max(0, int(distill_synthetic_min_size))),
            "--timeout-sec",
            str(max(60, int(timeout_sec))),
        ]
    else:
        distill_cmd = [
            str(PY),
            "tools/run_self_eval_distillation_pipeline.py",
            "--sessions",
            str(max(1, int(distill_sessions))),
            "--teacher-provider",
            str(distill_teacher_provider),
            "--max-samples",
            str(max(1, int(distill_max_samples))),
            "--synthetic-min-size",
            str(max(0, int(distill_synthetic_min_size))),
            "--execution-mode",
            str(distill_execution_mode),
        ]
    if distill_include_valid_prompts:
        distill_cmd.append("--include-valid-prompts")

    return [
        ExperimentSpec(
            name="distillation_uplift",
            description="Run self-eval distillation pipeline and maximize schema-valid uplift.",
            cmd=distill_cmd,
            metric_name="delta_schema_valid_rate",
            metric_reader=read_distill_delta,
            higher_is_better=True,
        ),
        ExperimentSpec(
            name="research_cross_validation",
            description="Aggregate evidence families and maximize cross-validated aggregate score.",
            cmd=[str(PY), "tools/run_research_aggregation_cross_validation.py"],
            metric_name="aggregate_score",
            metric_reader=read_research_aggregate,
            higher_is_better=True,
        ),
        ExperimentSpec(
            name="systemic_joint_capability",
            description="Run systemic joint capability assessment (ci-safe) and maximize trusted aggregate.",
            cmd=[str(PY), "tools/run_systemic_platform_joint_capability_assessment.py", "--ci-safe", "--blueprint-cycles", "1", "--longrun-cycles", "1"],
            metric_name="systemic_score",
            metric_reader=read_systemic_score,
            higher_is_better=True,
        ),
    ]


def choose_schedule(max_iterations: int, snapshot: Dict[str, float]) -> List[str]:
    score = snapshot.get("research_aggregate", 0.0)
    systemic = snapshot.get("systemic_score", 0.0)
    distill = snapshot.get("distill_delta", 0.0)

    priority: List[str] = []
    if distill <= 0.0:
        priority.append("distillation_uplift")
    if score < 0.85:
        priority.append("research_cross_validation")
    if systemic < 0.80:
        priority.append("systemic_joint_capability")

    # Ensure full loop coverage at least once.
    for n in ["distillation_uplift", "research_cross_validation", "systemic_joint_capability"]:
        if n not in priority:
            priority.append(n)

    out: List[str] = []
    while len(out) < max_iterations:
        out.append(priority[len(out) % len(priority)])
    return out[:max_iterations]


def run_experiment(spec: ExperimentSpec, timeout_sec: int, execute: bool, baseline_metric: Optional[float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "cmd": spec.cmd,
        "metric_name": spec.metric_name,
        "baseline_metric": baseline_metric,
        "executed": execute,
        "status": "planned" if not execute else "unknown",
        "metric_after": baseline_metric,
        "delta": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
        "returncode": None,
    }

    if not execute:
        return row

    start = time.time()
    try:
        proc = subprocess.run(
            spec.cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        row["returncode"] = int(proc.returncode)
        row["stdout_tail"] = _tail(proc.stdout)
        row["stderr_tail"] = _tail(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        row["returncode"] = 124
        row["stdout_tail"] = _tail(exc.stdout or "")
        row["stderr_tail"] = _tail((exc.stderr or "") + "\nTIMEOUT")
        row["status"] = "crash"
        row["metric_after"] = baseline_metric
        row["delta"] = 0.0
        row["elapsed_sec"] = time.time() - start
        return row

    metric_after = spec.metric_reader()
    row["metric_after"] = metric_after

    if baseline_metric is None or metric_after is None:
        delta = 0.0
    else:
        delta = float(metric_after - baseline_metric)
    row["delta"] = delta

    if row["returncode"] != 0:
        row["status"] = "crash"
    else:
        improved = delta > 0.0 if spec.higher_is_better else delta < 0.0
        row["status"] = "keep" if improved else "discard"

    row["elapsed_sec"] = time.time() - start
    return row


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Autoresearch-H2Q Bootstrap Fusion")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append(f"- execute: `{payload['meta']['execute']}`")
    lines.append(f"- iterations: `{payload['meta']['max_iterations']}`")
    lines.append(f"- timeout_sec: `{payload['meta']['timeout_sec']}`")
    lines.append("")

    ar = payload["autoresearch_summary"]
    lines.append("## Upstream Autoresearch Summary")
    lines.append(f"- source: `{ar['source']}`")
    lines.append(f"- exists: `{ar['exists']}`")
    lines.append(f"- keep/discard/crash: `{ar['keep_count']}/{ar['discard_count']}/{ar['crash_count']}`")
    if ar.get("top_keep_descriptions"):
        lines.append("- top_keep_descriptions:")
        for d in ar["top_keep_descriptions"]:
            lines.append(f"  - {d}")
    lines.append("")

    lines.append("## Baseline Snapshot")
    base = payload["baseline_snapshot"]
    lines.append(f"- distill_delta: `{base['distill_delta']}`")
    lines.append(f"- research_aggregate: `{base['research_aggregate']}`")
    lines.append(f"- systemic_score: `{base['systemic_score']}`")
    lines.append("")

    lines.append("## Experiment Ledger")
    lines.append("| i | name | status | metric | baseline | after | delta | decision_score | delta_signal | meaning_signal | geometry_signal | axiom_signal | weight_gain_signal | weight_gate | decision_reason |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, row in enumerate(payload["experiments"], start=1):
        d = row.get("decision") or {}
        ds = d.get("decision_score", None)
        dlt = d.get("delta_signal", None)
        ms = d.get("meaning_signal", None)
        gs = d.get("geometry_signal", None)
        ax = d.get("axiom_signal", None)
        ws = d.get("weight_gain_signal", None)
        wg = d.get("weight_gate", None)

        ds_s = "-" if ds is None else f"{_f(ds):.4f}"
        dlt_s = "-" if dlt is None else f"{_f(dlt):.4f}"
        ms_s = "-" if ms is None else f"{_f(ms):.4f}"
        gs_s = "-" if gs is None else f"{_f(gs):.4f}"
        ax_s = "-" if ax is None else f"{_f(ax):.4f}"
        ws_s = "-" if ws is None else f"{_f(ws):.4f}"
        wg_s = "-" if wg is None else f"{_f(wg):.4f}"
        reason = str(row.get("decision_reason", "-")).replace("|", "/")

        lines.append(
            f"| {i} | {row['name']} | {row['status']} | {row['metric_name']} | "
            f"{row['baseline_metric']} | {row['metric_after']} | {row['delta']:+.6f} | {ds_s} | {dlt_s} | {ms_s} | {gs_s} | {ax_s} | {ws_s} | {wg_s} | {reason} |"
        )
    lines.append("")

    lines.append("## Weight Training Signal")
    w0 = payload.get("weight_training_signal_baseline") or {}
    w1 = payload.get("weight_training_signal_final") or {}
    lines.append(f"- source: `{w1.get('source', w0.get('source', ''))}`")
    lines.append(f"- baseline_loss_improvement_rate: `{_f(w0.get('loss_improvement_rate', 0.0)):+.6f}`")
    lines.append(f"- final_loss_improvement_rate: `{_f(w1.get('loss_improvement_rate', 0.0)):+.6f}`")

    wcurve = payload.get("weight_training_curve") or []
    if wcurve:
        lines.append("- curve:")
        for point in wcurve:
            lines.append(
                f"  - step={point.get('step')}, experiment={point.get('experiment')}, gain={_f(point.get('loss_improvement_rate', 0.0)):+.6f}, signal={_f(point.get('weight_gain_signal', 0.0)):.6f}, gate={_f(point.get('weight_gate', 0.0)):.6f}, eff_w={_f(point.get('weight_effective_weight', 0.0)):.6f}"
            )

    lines.append("")

    lines.append("## MeaningScore")
    ms0 = payload.get("meaning_score_baseline") or {}
    ms1 = payload.get("meaning_score_final") or {}
    lines.append(f"- baseline: `{_f(ms0.get('score', 0.0)):.6f}`")
    lines.append(f"- final: `{_f(ms1.get('score', 0.0)):.6f}`")

    curve = payload.get("meaning_curve") or []
    if curve:
        lines.append("- curve:")
        for point in curve:
            lines.append(
                f"  - step={point.get('step')}, experiment={point.get('experiment')}, score={_f(point.get('score', 0.0)):.6f}"
            )

    lines.append("")
    lines.append("## Geometric Signal")
    gs0 = payload.get("geometry_signal_baseline") or {}
    gs1 = payload.get("geometry_signal_final") or {}
    lines.append(f"- baseline_score: `{_f(gs0.get('geometry_score', 0.0)):.6f}`")
    lines.append(f"- final_score: `{_f(gs1.get('geometry_score', 0.0)):.6f}`")
    lines.append(f"- baseline_boundary_ratio: `{_f(gs0.get('ap_gp_boundary_ratio', 0.0)):.6f}`")
    lines.append(f"- final_boundary_ratio: `{_f(gs1.get('ap_gp_boundary_ratio', 0.0)):.6f}`")
    lines.append(f"- baseline_projection_accel: `{_f(gs0.get('projection_acceleration', 0.0)):.6f}`")
    lines.append(f"- final_projection_accel: `{_f(gs1.get('projection_acceleration', 0.0)):.6f}`")

    gcurve = payload.get("geometry_curve") or []
    if gcurve:
        lines.append("- curve:")
        for point in gcurve:
            lines.append(
                f"  - step={point.get('step')}, experiment={point.get('experiment')}, score={_f(point.get('score', 0.0)):.6f}, boundary={_f(point.get('boundary_ratio', 0.0)):.6f}, accel={_f(point.get('projection_accel', 0.0)):.6f}"
            )

    lines.append("")
    lines.append("## Axiom Consistency")
    ax0 = payload.get("axiom_consistency_baseline") or {}
    ax1 = payload.get("axiom_consistency_final") or {}
    lines.append(f"- baseline_score: `{_f(ax0.get('score', 0.0)):.6f}`")
    lines.append(f"- final_score: `{_f(ax1.get('score', 0.0)):.6f}`")
    lines.append(f"- baseline_pass_rate: `{_f(ax0.get('pass_rate', 0.0)):.6f}`")
    lines.append(f"- final_pass_rate: `{_f(ax1.get('pass_rate', 0.0)):.6f}`")

    acurve = payload.get("axiom_curve") or []
    if acurve:
        lines.append("- curve:")
        for point in acurve:
            lines.append(
                f"  - step={point.get('step')}, experiment={point.get('experiment')}, score={_f(point.get('score', 0.0)):.6f}, pass_rate={_f(point.get('pass_rate', 0.0)):.6f}"
            )

    lines.append("")
    lines.append("## Next Bootstrap Plan")
    for item in payload["next_plan"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoresearch x H2Q bootstrap fusion runner")
    parser.add_argument("--autoresearch-results", default="external/autoresearch/results.tsv")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--execute", action="store_true", help="Run experiments. Default is planning only.")
    parser.add_argument("--output-prefix", default="autoresearch_h2q_bootstrap_fusion")
    parser.add_argument("--axiom-contract", default="axiom_contract.json")
    parser.add_argument("--distill-sessions", type=int, default=4)
    parser.add_argument("--distill-teacher-provider", choices=["deepseek", "heuristic"], default="heuristic")
    parser.add_argument("--distill-max-samples", type=int, default=120)
    parser.add_argument("--distill-synthetic-min-size", type=int, default=120)
    parser.add_argument("--distill-include-valid-prompts", action="store_true")
    parser.add_argument("--distill-execution-mode", choices=["full", "compressed"], default="compressed")
    parser.add_argument(
        "--distill-chunks",
        type=int,
        default=0,
        help="Chunk count for distillation. 0 means auto (2 chunks when max-samples>=120).",
    )
    parser.add_argument(
        "--distill-chunk-size",
        type=int,
        default=0,
        help="Per-chunk sample size when --distill-chunks>1 or auto chunking is enabled.",
    )
    parser.add_argument("--weight-signal-base-weight", type=float, default=0.15)
    parser.add_argument("--weight-signal-decay", type=float, default=0.20)
    parser.add_argument("--weight-signal-gate-floor", type=float, default=0.15)
    parser.add_argument("--hard-gate-benchmark-gain", type=float, default=1e-4)
    parser.add_argument("--disable-adaptive-benchmark-gate", action="store_true")
    parser.add_argument("--hard-gate-benchmark-lookback", type=int, default=12)
    parser.add_argument("--hard-gate-benchmark-quantile", type=float, default=0.25)
    parser.add_argument("--hard-gate-benchmark-floor", type=float, default=1e-5)
    parser.add_argument("--disable-hard-gate", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    ar_path = Path(args.autoresearch_results)
    if not ar_path.is_absolute():
        ar_path = ROOT / ar_path
    ar_summary = parse_autoresearch_tsv(ar_path)

    axiom_path = Path(args.axiom_contract)
    if not axiom_path.is_absolute():
        axiom_path = ROOT / axiom_path
    axiom_contract = load_axiom_contract(axiom_path)

    baseline_snapshot = {
        "distill_delta": read_distill_delta() or 0.0,
        "research_aggregate": read_research_aggregate() or 0.0,
        "systemic_score": read_systemic_score() or 0.0,
    }

    # Auto strategy: for high-sample runs, split to 2x60 style batches to reduce timeout risk.
    if int(args.distill_chunks) <= 0:
        if int(args.distill_max_samples) >= 120:
            resolved_distill_chunks = 2
            resolved_chunk_size = max(1, int(args.distill_chunk_size) or int(args.distill_max_samples) // 2)
        else:
            resolved_distill_chunks = 1
            resolved_chunk_size = max(1, int(args.distill_chunk_size) or int(args.distill_max_samples))
    else:
        resolved_distill_chunks = max(1, int(args.distill_chunks))
        if int(args.distill_chunk_size) > 0:
            resolved_chunk_size = int(args.distill_chunk_size)
        else:
            resolved_chunk_size = max(1, int(math.ceil(int(args.distill_max_samples) / float(resolved_distill_chunks))))

    all_specs = {
        s.name: s
        for s in build_experiments(
            timeout_sec=args.timeout_sec,
            distill_sessions=args.distill_sessions,
            distill_teacher_provider=args.distill_teacher_provider,
            distill_max_samples=args.distill_max_samples,
            distill_synthetic_min_size=args.distill_synthetic_min_size,
            distill_include_valid_prompts=bool(args.distill_include_valid_prompts),
            distill_chunks=resolved_distill_chunks,
            distill_chunk_size=resolved_chunk_size,
            distill_execution_mode=args.distill_execution_mode,
        )
    }
    schedule = choose_schedule(max_iterations=max(1, args.max_iterations), snapshot=baseline_snapshot)

    hard_gate_enabled = not bool(args.disable_hard_gate)
    adaptive_gate = compute_adaptive_benchmark_gate(
        reports_dir=REPORTS,
        static_threshold=float(args.hard_gate_benchmark_gain),
        enabled=(hard_gate_enabled and (not bool(args.disable_adaptive_benchmark_gate))),
        lookback=int(args.hard_gate_benchmark_lookback),
        quantile=float(args.hard_gate_benchmark_quantile),
        safety_floor=float(args.hard_gate_benchmark_floor),
    )
    resolved_benchmark_threshold = float(adaptive_gate.get("resolved_threshold", float(args.hard_gate_benchmark_gain)))
    benchmark_probe_before = run_incremental_benchmark_probe(PY) if args.execute else {
        "returncode": 0,
        "gain": 0.0,
        "score_base": 0.0,
        "score_adapter": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    lora_replay_before = read_lora_replay_signal()

    experiments: List[Dict[str, Any]] = []
    current_baseline = dict(baseline_snapshot)
    meaning_curve: List[Dict[str, Any]] = []
    geometry_curve: List[Dict[str, Any]] = []
    axiom_curve: List[Dict[str, Any]] = []
    weight_curve: List[Dict[str, Any]] = []

    w_start = read_trusted_weight_training_signal()
    w_start_rate = _f(w_start.get("loss_improvement_rate", 0.0), 0.0)
    w_start_signal = 0.5 + 0.5 * math.tanh(16.0 * w_start_rate)

    ms_start = compute_meaning_score(current_baseline, experiments)
    gs_start = compute_dual_projection_signals(current_baseline)
    ax_start = evaluate_axiom_consistency(
        snapshot=current_baseline,
        signals=gs_start,
        meaning_score=_f(ms_start.get("score", 0.0), 0.0),
        experiments=experiments,
        contract=axiom_contract,
    )
    meaning_curve.append(
        {
            "step": 0,
            "experiment": "baseline",
            "score": _f(ms_start.get("score", 0.0), 0.0),
        }
    )
    geometry_curve.append(
        {
            "step": 0,
            "experiment": "baseline",
            "score": _f(gs_start.get("geometry_score", 0.0), 0.0),
            "boundary_ratio": _f(gs_start.get("ap_gp_boundary_ratio", 0.0), 0.0),
            "projection_accel": _f(gs_start.get("projection_acceleration", 0.0), 0.0),
        }
    )
    axiom_curve.append(
        {
            "step": 0,
            "experiment": "baseline",
            "score": _f(ax_start.get("score", 0.0), 0.0),
            "pass_rate": _f(ax_start.get("pass_rate", 0.0), 0.0),
        }
    )
    weight_curve.append(
        {
            "step": 0,
            "experiment": "baseline",
            "loss_improvement_rate": w_start_rate,
            "weight_gain_signal": w_start_signal,
        }
    )

    for name in schedule:
        spec = all_specs[name]
        if spec.name == "distillation_uplift":
            b = current_baseline["distill_delta"]
        elif spec.name == "research_cross_validation":
            b = current_baseline["research_aggregate"]
        else:
            b = current_baseline["systemic_score"]

        row = run_experiment(spec, timeout_sec=max(60, args.timeout_sec), execute=bool(args.execute), baseline_metric=b)

        if args.execute and row.get("returncode") == 0 and row.get("metric_after") is not None:
            candidate_snapshot = apply_metric_to_snapshot(current_baseline, spec.name, row.get("metric_after"))
            bench_probe_now = run_incremental_benchmark_probe(PY)
            lora_replay_now = read_lora_replay_signal()
            decision = compute_iteration_decision(
                snapshot_before=current_baseline,
                snapshot_after=candidate_snapshot,
                experiments_so_far=experiments,
                delta=_f(row.get("delta", 0.0), 0.0),
                axiom_contract=axiom_contract,
                weight_gain_rate=_f(read_trusted_weight_training_signal().get("loss_improvement_rate", 0.0), 0.0),
                step_index=len(experiments) + 1,
                weight_signal_base_weight=float(args.weight_signal_base_weight),
                weight_signal_decay=float(args.weight_signal_decay),
                weight_signal_gate_floor=float(args.weight_signal_gate_floor),
            )
            decision["benchmark_gain"] = _f(bench_probe_now.get("gain", 0.0), 0.0)
            decision["benchmark_score_base"] = _f(bench_probe_now.get("score_base", 0.0), 0.0)
            decision["benchmark_score_adapter"] = _f(bench_probe_now.get("score_adapter", 0.0), 0.0)
            decision["benchmark_probe_returncode"] = int(bench_probe_now.get("returncode", 1))
            decision["lora_replay_pass"] = bool(lora_replay_now.get("replay_pass", False))
            decision["lora_replay_score"] = _f(lora_replay_now.get("replay_score", 0.0), 0.0)
            decision["replay_quality_pass"] = bool(lora_replay_now.get("replay_quality_pass", False))
            decision["replay_quality_score"] = _f(lora_replay_now.get("replay_quality_score", 0.0), 0.0)
            decision["replay_quality_structure_rate"] = _f(lora_replay_now.get("replay_quality_structure_rate", 0.0), 0.0)
            decision["replay_quality_density_rate"] = _f(lora_replay_now.get("replay_quality_density_rate", 0.0), 0.0)
            decision["replay_quality_echo_rate"] = _f(lora_replay_now.get("replay_quality_echo_rate", 1.0), 1.0)
            row["decision"] = decision
            row["status"] = "keep" if _f(decision.get("decision_score", 0.0), 0.0) > 0.5 else "discard"
            if hard_gate_enabled and row["status"] == "keep":
                gate_bench_ok = _f(decision.get("benchmark_gain", 0.0), 0.0) >= resolved_benchmark_threshold
                gate_lora_ok = bool(decision.get("lora_replay_pass", False))
                gate_quality_ok = bool(decision.get("replay_quality_pass", False))
                decision["hard_gate"] = {
                    "enabled": True,
                    "benchmark_gain_threshold": resolved_benchmark_threshold,
                    "benchmark_ok": gate_bench_ok,
                    "lora_replay_ok": gate_lora_ok,
                    "output_quality_ok": gate_quality_ok,
                }
                if not (gate_bench_ok and gate_lora_ok and gate_quality_ok):
                    row["status"] = "discard"
            elif hard_gate_enabled:
                gate_bench_ok = _f(decision.get("benchmark_gain", 0.0), 0.0) >= resolved_benchmark_threshold
                decision["hard_gate"] = {
                    "enabled": True,
                    "benchmark_gain_threshold": resolved_benchmark_threshold,
                    "benchmark_ok": gate_bench_ok,
                    "lora_replay_ok": bool(decision.get("lora_replay_pass", False)),
                    "output_quality_ok": bool(decision.get("replay_quality_pass", False)),
                }
        elif args.execute and row.get("returncode") != 0:
            row["decision"] = {"decision_score": 0.0}

        row["decision_reason"] = build_decision_reason(row)

        experiments.append(row)

        # Advance baseline only if keep.
        if row["status"] == "keep" and row.get("metric_after") is not None:
            current_baseline = apply_metric_to_snapshot(current_baseline, spec.name, row.get("metric_after"))

        ms_now = compute_meaning_score(current_baseline, experiments)
        meaning_curve.append(
            {
                "step": len(experiments),
                "experiment": spec.name,
                "score": _f(ms_now.get("score", 0.0), 0.0),
            }
        )

        gs_now = compute_dual_projection_signals(current_baseline)
        geometry_curve.append(
            {
                "step": len(experiments),
                "experiment": spec.name,
                "score": _f(gs_now.get("geometry_score", 0.0), 0.0),
                "boundary_ratio": _f(gs_now.get("ap_gp_boundary_ratio", 0.0), 0.0),
                "projection_accel": _f(gs_now.get("projection_acceleration", 0.0), 0.0),
            }
        )

        ax_now = evaluate_axiom_consistency(
            snapshot=current_baseline,
            signals=gs_now,
            meaning_score=_f(ms_now.get("score", 0.0), 0.0),
            experiments=experiments,
            contract=axiom_contract,
        )
        axiom_curve.append(
            {
                "step": len(experiments),
                "experiment": spec.name,
                "score": _f(ax_now.get("score", 0.0), 0.0),
                "pass_rate": _f(ax_now.get("pass_rate", 0.0), 0.0),
            }
        )

        w_now = read_trusted_weight_training_signal()
        w_rate = _f(w_now.get("loss_improvement_rate", 0.0), 0.0)
        weight_curve.append(
            {
                "step": len(experiments),
                "experiment": spec.name,
                "loss_improvement_rate": w_rate,
                "weight_gain_signal": 0.5 + 0.5 * math.tanh(16.0 * w_rate),
                "weight_gate": _f((row.get("decision") or {}).get("weight_gate", 0.0), 0.0),
                "weight_effective_weight": _f((row.get("decision") or {}).get("weight_effective_weight", 0.0), 0.0),
            }
        )

    keep_count = sum(1 for r in experiments if r["status"] == "keep")
    crash_count = sum(1 for r in experiments if r["status"] == "crash")
    w_final = read_trusted_weight_training_signal()
    w_final_signal = 0.5 + 0.5 * math.tanh(16.0 * _f(w_final.get("loss_improvement_rate", 0.0), 0.0))

    next_plan = [
        "Increase experiment breadth around components marked 'discard' with smaller perturbations.",
        "Prioritize distillation + consistency coupling before scaling systemic gate complexity.",
        "Run overnight execute mode with max-iterations >= 12 for true autoresearch-style cadence.",
    ]
    if crash_count > 0:
        next_plan.insert(0, "Investigate crash experiments first; enforce smaller timeout and safer flags.")
    if keep_count == 0:
        next_plan.insert(0, "No improvements found; reduce step size and test single-factor mutations.")

    benchmark_probe_failures = sum(
        1
        for r in experiments
        if int((r.get("decision") or {}).get("benchmark_probe_returncode", 0)) != 0
    )

    payload: Dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "meta": {
            "execute": bool(args.execute),
            "max_iterations": int(max(1, args.max_iterations)),
            "timeout_sec": int(max(60, args.timeout_sec)),
            "output_prefix": args.output_prefix,
            "weight_signal_base_weight": float(args.weight_signal_base_weight),
            "weight_signal_decay": float(args.weight_signal_decay),
            "weight_signal_gate_floor": float(args.weight_signal_gate_floor),
            "hard_gate_enabled": hard_gate_enabled,
            "hard_gate_benchmark_gain": float(args.hard_gate_benchmark_gain),
            "hard_gate_benchmark_gain_resolved": resolved_benchmark_threshold,
            "hard_gate_benchmark_adaptive": adaptive_gate,
        },
        "autoresearch_summary": ar_summary,
        "baseline_snapshot": baseline_snapshot,
        "final_snapshot": current_baseline,
        "experiments": experiments,
        "summary": {
            "keep": keep_count,
            "discard": sum(1 for r in experiments if r["status"] == "discard"),
            "crash": crash_count,
            "benchmark_probe_failures": benchmark_probe_failures,
        },
        "meaning_score_baseline": ms_start,
        "meaning_score_final": compute_meaning_score(current_baseline, experiments),
        "meaning_curve": meaning_curve,
        "geometry_signal_baseline": gs_start,
        "geometry_signal_final": compute_dual_projection_signals(current_baseline),
        "geometry_curve": geometry_curve,
        "axiom_contract": {
            "path": str(axiom_path),
            "version": axiom_contract.get("version", "unknown"),
        },
        "axiom_consistency_baseline": ax_start,
        "axiom_consistency_final": evaluate_axiom_consistency(
            snapshot=current_baseline,
            signals=compute_dual_projection_signals(current_baseline),
            meaning_score=_f(compute_meaning_score(current_baseline, experiments).get("score", 0.0), 0.0),
            experiments=experiments,
            contract=axiom_contract,
        ),
        "axiom_curve": axiom_curve,
        "weight_training_signal_baseline": {
            **w_start,
            "weight_gain_signal": w_start_signal,
        },
        "weight_training_signal_final": {
            **w_final,
            "weight_gain_signal": w_final_signal,
        },
        "weight_training_curve": weight_curve,
        "incremental_benchmark_baseline": benchmark_probe_before,
        "lora_replay_baseline": lora_replay_before,
        "next_plan": next_plan,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"
    ledger_tsv = REPORTS / "autoresearch_h2q_experiment_ledger_latest.tsv"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    out_md.write_text(render_markdown(payload), encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    # Flatten ledger for quick diff-friendly tracking.
    with ledger_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "name", "status", "metric_name", "baseline_metric", "metric_after", "delta", "returncode"])
        for i, row in enumerate(experiments, start=1):
            writer.writerow([
                i,
                row.get("name", ""),
                row.get("status", ""),
                row.get("metric_name", ""),
                row.get("baseline_metric", ""),
                row.get("metric_after", ""),
                f"{float(row.get('delta', 0.0)):+.6f}",
                row.get("returncode", ""),
            ])

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Ledger TSV: {ledger_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
