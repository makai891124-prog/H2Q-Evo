#!/usr/bin/env python3
"""Generate one-click KPI dashboard from latest run artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_latest_session(min_ts: Optional[int]) -> Optional[Path]:
    sessions = sorted(REPORTS.glob("trusted_local_agi_chat_session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if min_ts is None:
        return sessions[0] if sessions else None

    for p in sessions:
        try:
            stamp = int(p.stem.rsplit("_", 1)[-1])
        except Exception:
            stamp = 0
        if stamp >= min_ts:
            return p
    return sessions[0] if sessions else None


def _find_latest_release_gate(prefer_deepseek: bool) -> Optional[Path]:
    candidates: List[Path] = []
    if prefer_deepseek:
        preferred = REPORTS / "deepseek_assisted_release_gate_latest.json"
        if preferred.exists():
            return preferred
    for name in [
        "deepseek_assisted_release_gate_latest.json",
        "release_gate_latest.json",
        "openclaw_bridge_release_gate_latest.json",
    ]:
        p = REPORTS / name
        if p.exists():
            candidates.append(p)
    return _latest_by_mtime(candidates)


def _collect_distillation_metrics() -> Dict[str, Any]:
    pipeline_path = REPORTS / "self_eval_distillation_pipeline_latest.json"
    distilled_path = REPORTS / "self_model_consistency_distilled_latest.json"

    baseline = 0.0
    distilled = 0.0
    delta = 0.0
    positive = False
    total_runs = 0
    adapter_enabled = False

    if pipeline_path.exists():
        try:
            payload = _load_json(pipeline_path)
            metrics = payload.get("metrics") or {}
            baseline = float(metrics.get("baseline_schema_valid_rate", baseline) or baseline)
            distilled = float(metrics.get("distilled_schema_valid_rate", distilled) or distilled)
            delta = float(metrics.get("delta_schema_valid_rate", distilled - baseline) or (distilled - baseline))
            positive = bool(metrics.get("schema_valid_rate_positive", distilled > 0.0))
        except Exception:
            pass

    if distilled_path.exists():
        try:
            payload = _load_json(distilled_path)
            meta = payload.get("meta") or {}
            metrics = payload.get("metrics") or {}
            total_runs = int(meta.get("total_runs", 0) or 0)
            adapter_enabled = bool(meta.get("distill_adapter_enabled", False))
            distilled = float(metrics.get("schema_valid_rate", distilled) or distilled)
            if abs(delta) < 1e-12:
                delta = distilled - baseline
            positive = distilled > 0.0
        except Exception:
            pass

    return {
        "pipeline_source": str(pipeline_path) if pipeline_path.exists() else None,
        "distilled_benchmark_source": str(distilled_path) if distilled_path.exists() else None,
        "baseline_schema_valid_rate": baseline,
        "distilled_schema_valid_rate": distilled,
        "delta_schema_valid_rate": delta,
        "schema_valid_rate_positive": positive,
        "distill_total_runs": total_runs,
        "distill_adapter_enabled": adapter_enabled,
    }


def _collect_distillation_robustness_compare(target_sessions: List[int]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    by_session: Dict[int, Dict[str, Any]] = {}

    for p in REPORTS.glob("self_model_consistency_distilled_*.json"):
        if p.name.endswith("_latest.json"):
            continue
        try:
            payload = _load_json(p)
            meta = payload.get("meta") or {}
            metrics = payload.get("metrics") or {}
            sessions = int(meta.get("sessions", 0) or 0)
            total_runs = int(meta.get("total_runs", 0) or 0)
            schema_valid_rate = float(metrics.get("schema_valid_rate", 0.0) or 0.0)
            score = float(metrics.get("overall_score", 0.0) or 0.0)
            item = {
                "path": str(p),
                "mtime": float(p.stat().st_mtime),
                "sessions": sessions,
                "total_runs": total_runs,
                "schema_valid_rate": schema_valid_rate,
                "overall_score": score,
            }
            runs.append(item)
        except Exception:
            continue

    for s in target_sessions:
        candidates = [x for x in runs if x["sessions"] == s]
        if not candidates:
            continue
        by_session[s] = sorted(candidates, key=lambda x: x["mtime"], reverse=True)[0]

    compared = []
    for s in target_sessions:
        if s in by_session:
            compared.append(by_session[s])

    comparison: Dict[str, Any] = {
        "targets": target_sessions,
        "available": [x["sessions"] for x in compared],
        "runs": compared,
    }

    if len(compared) >= 2:
        first = compared[0]
        last = compared[-1]
        comparison["delta_schema_valid_rate"] = float(last["schema_valid_rate"] - first["schema_valid_rate"])
        comparison["delta_overall_score"] = float(last["overall_score"] - first["overall_score"])
    else:
        comparison["delta_schema_valid_rate"] = 0.0
        comparison["delta_overall_score"] = 0.0

    return comparison


def _collect_formal_assessment_summary() -> Dict[str, Any]:
    formal_path = REPORTS / "distill_evo_public_formal_assessment_latest.json"
    if not formal_path.exists():
        return {
            "source": None,
            "available": False,
        }

    try:
        payload = _load_json(formal_path)
        logic = payload.get("logic_closure") or {}
        facts = logic.get("facts") or {}
        distill = payload.get("distillation") or {}
        public = payload.get("public_validation") or {}
        robust = payload.get("robustness_compare") or {}
        return {
            "source": str(formal_path),
            "available": True,
            "generated_at_utc": payload.get("generated_at_utc"),
            "lean_compile_success": bool(logic.get("lean_compile_success", False)),
            "facts": {
                "distill_pipeline_all_steps_ok": bool(facts.get("distill_pipeline_all_steps_ok", False)),
                "public_validation_all_steps_ok": bool(facts.get("public_validation_all_steps_ok", False)),
                "distilled_schema_positive": bool(facts.get("distilled_schema_positive", False)),
                "baseline_gate_ok": bool(facts.get("baseline_gate_ok", False)),
                "longrun_gate_ok": bool(facts.get("longrun_gate_ok", False)),
            },
            "distillation": {
                "sessions": int(distill.get("sessions", 0) or 0),
                "schema_valid_rate": float(distill.get("schema_valid_rate", 0.0) or 0.0),
                "overall_score": float(distill.get("overall_score", 0.0) or 0.0),
                "pipeline_delta_schema_valid_rate": float(distill.get("pipeline_delta_schema_valid_rate", 0.0) or 0.0),
            },
            "public_validation": {
                "baseline_gate_ok": bool(public.get("baseline_gate_ok", False)),
                "longrun_gate_ok": bool(public.get("longrun_gate_ok", False)),
                "baseline_alignment_overall": float(public.get("baseline_alignment_overall", 0.0) or 0.0),
                "longrun_alignment_overall": float(public.get("longrun_alignment_overall", 0.0) or 0.0),
            },
            "robustness_compare": {
                "delta_schema_valid_rate_50_minus_30": float(
                    robust.get("delta_schema_valid_rate_50_minus_30", 0.0) or 0.0
                ),
                "delta_overall_score_50_minus_30": float(robust.get("delta_overall_score_50_minus_30", 0.0) or 0.0),
            },
        }
    except Exception:
        return {
            "source": str(formal_path),
            "available": False,
        }


def _collect_systemic_joint_summary() -> Dict[str, Any]:
    joint_path = REPORTS / "systemic_platform_joint_capability_latest.json"
    if not joint_path.exists():
        return {
            "source": None,
            "available": False,
        }

    try:
        payload = _load_json(joint_path)
        agg = payload.get("aggregate_effectiveness") or {}
        cv = payload.get("cross_validation") or {}
        conclusion = payload.get("conclusion") or {}
        return {
            "source": str(joint_path),
            "available": True,
            "generated_at_utc": payload.get("generated_at_utc"),
            "self_problem": payload.get("self_problem"),
            "aggregate_score": float(agg.get("score", 0.0) or 0.0),
            "cv_min_score": float(cv.get("min_score", 0.0) or 0.0),
            "cv_std_score": float(cv.get("std_score", 0.0) or 0.0),
            "all_steps_ok": bool(conclusion.get("all_steps_ok", False)),
            "robust_claim": bool(conclusion.get("robust_claim", False)),
            "statement": conclusion.get("statement"),
        }
    except Exception:
        return {
            "source": str(joint_path),
            "available": False,
        }


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(1.0, float(value)))
    fill = int(round(v * width))
    return "#" * fill + "-" * (width - fill)


def _extract_epoch_from_name(path: Path, prefix: str) -> int:
    stem = path.stem
    key = f"{prefix}_"
    if not stem.startswith(key):
        return 0
    try:
        return int(stem[len(key) :])
    except Exception:
        return 0


def _load_trend_history(output_prefix: str, limit: int = 20) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    latest_name = f"{output_prefix}_latest.json"
    for p in REPORTS.glob(f"{output_prefix}_*.json"):
        if p.name == latest_name:
            continue
        epoch = _extract_epoch_from_name(p, output_prefix)
        if epoch <= 0:
            continue
        try:
            payload = _load_json(p)
            kpis = payload.get("kpis") or {}
            points.append(
                {
                    "epoch": epoch,
                    "label": datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%m-%d %H:%M"),
                    "strict_json_attempt_count": float(kpis.get("strict_json_attempt_count", 0.0) or 0.0),
                    "hard_fail_count": float(kpis.get("hard_fail_count", 0.0) or 0.0),
                    "fallback_ratio_self_eval": float(kpis.get("fallback_ratio_self_eval", 0.0) or 0.0),
                    "teacher_assist_dependency_ratio": float(kpis.get("teacher_assist_dependency_ratio", 0.0) or 0.0),
                    "distilled_schema_valid_rate": float(kpis.get("distilled_schema_valid_rate", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    points.sort(key=lambda x: x["epoch"])
    if len(points) > limit:
        points = points[-limit:]
    return points


def _plot_dashboard_chart(history: List[Dict[str, Any]], out_png: Path, out_latest_png: Path) -> None:
    labels = [p["label"] for p in history]
    fallback = [p["fallback_ratio_self_eval"] for p in history]
    teacher = [p["teacher_assist_dependency_ratio"] for p in history]
    distilled_valid = [p["distilled_schema_valid_rate"] for p in history]
    strict_counts = [p["strict_json_attempt_count"] for p in history]
    hard_fail_counts = [p["hard_fail_count"] for p in history]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    ax1 = axes[0]
    x = range(len(labels))
    ax1.plot(x, fallback, marker="o", linewidth=2.0, color="#c0392b", label="fallback_ratio_self_eval")
    ax1.plot(x, teacher, marker="o", linewidth=2.0, color="#2e86c1", label="teacher_assist_dependency_ratio")
    ax1.plot(x, distilled_valid, marker="o", linewidth=2.0, color="#117a65", label="distilled_schema_valid_rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("ratio")
    ax1.set_title("One-Click KPI Trend (Ratios)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax2 = axes[1]
    width = 0.38
    x2 = list(x)
    ax2.bar([i - width / 2 for i in x2], strict_counts, width=width, color="#27ae60", label="strict_json_attempt_count")
    ax2.bar([i + width / 2 for i in x2], hard_fail_counts, width=width, color="#8e44ad", label="hard_fail_count")
    ax2.set_ylabel("count")
    ax2.set_title("One-Click KPI Trend (Counts)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.savefig(out_png, dpi=160)
    shutil.copy2(out_png, out_latest_png)
    plt.close(fig)


def _collect_session_kpis(session_payload: Dict[str, Any]) -> Dict[str, Any]:
    transcript = session_payload.get("transcript") or []
    strict_json_attempt_count = 0
    hard_fail_count = 1 if bool(((session_payload.get("meta") or {}).get("hard_fail_triggered", False))) else 0

    self_eval_total = 0
    self_eval_fallback = 0

    for turn in transcript:
        runtime = turn.get("runtime") or {}
        route = str(runtime.get("route", ""))
        schema = runtime.get("schema") or {}

        if bool(schema.get("strict_json_pre_fallback_attempted", False)):
            strict_json_attempt_count += 1

        if bool(schema.get("required", False)):
            self_eval_total += 1
            if route == "local_fallback_template":
                self_eval_fallback += 1

    fallback_ratio_self_eval = _safe_ratio(self_eval_fallback, self_eval_total)
    return {
        "strict_json_attempt_count": strict_json_attempt_count,
        "hard_fail_count": hard_fail_count,
        "self_eval_total": self_eval_total,
        "self_eval_fallback_count": self_eval_fallback,
        "fallback_ratio_self_eval": fallback_ratio_self_eval,
    }


def _collect_teacher_assist_dependency(release_payload: Dict[str, Any]) -> Dict[str, Any]:
    signals = release_payload.get("signals") or {}
    meta = release_payload.get("meta") or {}
    assist_enabled = bool(signals.get("assist_enabled", False))
    assist_calls = int(signals.get("assist_calls", 0) or 0)
    assist_success_calls = int(signals.get("assist_success_calls", 0) or 0)

    if not assist_enabled:
        ratio = 0.0
    else:
        # Dependency proxy: use success-call ratio to estimate effective reliance.
        ratio = _safe_ratio(assist_success_calls, max(assist_calls, 1))

    return {
        "assist_provider": str(meta.get("assist_provider", "none")),
        "assist_enabled": assist_enabled,
        "assist_calls": assist_calls,
        "assist_success_calls": assist_success_calls,
        "teacher_assist_dependency_ratio": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one-click KPI dashboard from latest artifacts")
    parser.add_argument("--output-prefix", default="one_click_kpi_dashboard")
    parser.add_argument("--run-ts", type=int, default=0, help="Prefer chat session files whose timestamp >= run-ts")
    parser.add_argument("--prefer-deepseek-assist", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    min_ts = args.run_ts if args.run_ts > 0 else None

    session_path = _find_latest_session(min_ts=min_ts)
    release_gate_path = _find_latest_release_gate(prefer_deepseek=args.prefer_deepseek_assist)

    if session_path is None:
        raise SystemExit("No trusted_local_agi_chat_session_*.json found in reports/")

    session_payload = _load_json(session_path)
    session_kpis = _collect_session_kpis(session_payload)

    assist_kpis: Dict[str, Any] = {
        "assist_provider": "none",
        "assist_enabled": False,
        "assist_calls": 0,
        "assist_success_calls": 0,
        "teacher_assist_dependency_ratio": 0.0,
    }
    if release_gate_path is not None:
        try:
            assist_kpis = _collect_teacher_assist_dependency(_load_json(release_gate_path))
        except Exception:
            pass

    distill_kpis = _collect_distillation_metrics()
    distill_compare = _collect_distillation_robustness_compare(target_sessions=[30, 50])
    formal_summary = _collect_formal_assessment_summary()
    systemic_summary = _collect_systemic_joint_summary()

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "session": str(session_path),
            "release_gate": str(release_gate_path) if release_gate_path else None,
            "distillation_pipeline": distill_kpis["pipeline_source"],
            "distilled_benchmark": distill_kpis["distilled_benchmark_source"],
            "formal_assessment": formal_summary.get("source"),
            "systemic_joint_assessment": systemic_summary.get("source"),
        },
        "kpis": {
            "strict_json_attempt_count": int(session_kpis["strict_json_attempt_count"]),
            "hard_fail_count": int(session_kpis["hard_fail_count"]),
            "fallback_ratio_self_eval": float(session_kpis["fallback_ratio_self_eval"]),
            "teacher_assist_dependency_ratio": float(assist_kpis["teacher_assist_dependency_ratio"]),
            "distilled_schema_valid_rate": float(distill_kpis["distilled_schema_valid_rate"]),
            "distill_schema_valid_rate_delta": float(distill_kpis["delta_schema_valid_rate"]),
            "distill_schema_valid_rate_positive": bool(distill_kpis["schema_valid_rate_positive"]),
            "systemic_joint_score": float(systemic_summary.get("aggregate_score", 0.0) or 0.0),
            "systemic_joint_robust_claim": bool(systemic_summary.get("robust_claim", False)),
        },
        "supporting": {
            "self_eval_total": int(session_kpis["self_eval_total"]),
            "self_eval_fallback_count": int(session_kpis["self_eval_fallback_count"]),
            "assist_provider": assist_kpis["assist_provider"],
            "assist_enabled": bool(assist_kpis["assist_enabled"]),
            "assist_calls": int(assist_kpis["assist_calls"]),
            "assist_success_calls": int(assist_kpis["assist_success_calls"]),
            "distill_baseline_schema_valid_rate": float(distill_kpis["baseline_schema_valid_rate"]),
            "distill_total_runs": int(distill_kpis["distill_total_runs"]),
            "distill_adapter_enabled": bool(distill_kpis["distill_adapter_enabled"]),
            "distill_robustness_compare": distill_compare,
            "formal_assessment_summary": formal_summary,
            "systemic_joint_summary": systemic_summary,
        },
    }

    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_png = REPORTS / f"{args.output_prefix}_{ts}.png"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"
    latest_png = REPORTS / f"{args.output_prefix}_latest.png"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    history = _load_trend_history(args.output_prefix, limit=20)
    # Ensure current run is included in the chart even when history was empty.
    history.append(
        {
            "epoch": ts,
            "label": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M"),
            "strict_json_attempt_count": float(payload["kpis"]["strict_json_attempt_count"]),
            "hard_fail_count": float(payload["kpis"]["hard_fail_count"]),
            "fallback_ratio_self_eval": float(payload["kpis"]["fallback_ratio_self_eval"]),
            "teacher_assist_dependency_ratio": float(payload["kpis"]["teacher_assist_dependency_ratio"]),
            "distilled_schema_valid_rate": float(payload["kpis"]["distilled_schema_valid_rate"]),
        }
    )
    history = sorted(history, key=lambda x: x["epoch"])[-20:]
    _plot_dashboard_chart(history=history, out_png=out_png, out_latest_png=latest_png)

    k = payload["kpis"]
    lines = [
        "# One-Click KPI Dashboard",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- session: `{payload['sources']['session']}`",
        f"- release_gate: `{payload['sources']['release_gate']}`",
        f"- distillation_pipeline: `{payload['sources']['distillation_pipeline']}`",
        f"- distilled_benchmark: `{payload['sources']['distilled_benchmark']}`",
        f"- formal_assessment: `{payload['sources']['formal_assessment']}`",
        f"- systemic_joint_assessment: `{payload['sources']['systemic_joint_assessment']}`",
        "",
        "## KPI Metrics",
        f"- strict_json_attempt_count: `{k['strict_json_attempt_count']}`",
        f"- hard_fail_count: `{k['hard_fail_count']}`",
        f"- fallback_ratio_self_eval: `{k['fallback_ratio_self_eval']:.6f}`",
        f"- teacher_assist_dependency_ratio: `{k['teacher_assist_dependency_ratio']:.6f}`",
        f"- distilled_schema_valid_rate: `{k['distilled_schema_valid_rate']:.6f}`",
        f"- distill_schema_valid_rate_delta: `{k['distill_schema_valid_rate_delta']:+.6f}`",
        f"- distill_schema_valid_rate_positive: `{k['distill_schema_valid_rate_positive']}`",
        f"- systemic_joint_score: `{k['systemic_joint_score']:.6f}`",
        f"- systemic_joint_robust_claim: `{k['systemic_joint_robust_claim']}`",
        "",
        "## Quick Visual",
        "- fallback_ratio_self_eval",
        f"  `{_bar(k['fallback_ratio_self_eval'])}`",
        "- teacher_assist_dependency_ratio",
        f"  `{_bar(k['teacher_assist_dependency_ratio'])}`",
        "- distilled_schema_valid_rate",
        f"  `{_bar(k['distilled_schema_valid_rate'])}`",
        "- systemic_joint_score",
        f"  `{_bar(k['systemic_joint_score'])}`",
        "",
        "## Trend Chart",
        "![One-Click KPI Trend](one_click_kpi_dashboard_latest.png)",
        "",
        "## Supporting Signals",
        f"- self_eval_total: `{payload['supporting']['self_eval_total']}`",
        f"- self_eval_fallback_count: `{payload['supporting']['self_eval_fallback_count']}`",
        f"- assist_provider: `{payload['supporting']['assist_provider']}`",
        f"- assist_enabled: `{payload['supporting']['assist_enabled']}`",
        f"- assist_calls: `{payload['supporting']['assist_calls']}`",
        f"- assist_success_calls: `{payload['supporting']['assist_success_calls']}`",
        f"- distill_baseline_schema_valid_rate: `{payload['supporting']['distill_baseline_schema_valid_rate']:.6f}`",
        f"- distill_total_runs: `{payload['supporting']['distill_total_runs']}`",
        f"- distill_adapter_enabled: `{payload['supporting']['distill_adapter_enabled']}`",
    ]

    compare = payload["supporting"].get("distill_robustness_compare") or {}
    runs = compare.get("runs") or []
    targets = compare.get("targets") or [30, 50]
    lines.extend([
        "",
        f"## Distillation Robustness ({targets[0]} vs {targets[-1]})",
        f"- available_sessions: `{compare.get('available', [])}`",
    ])
    for item in runs:
        lines.append(
            "- "
            f"sessions={int(item.get('sessions', 0))}, "
            f"total_runs={int(item.get('total_runs', 0))}, "
            f"schema_valid_rate={float(item.get('schema_valid_rate', 0.0)):.6f}, "
            f"overall_score={float(item.get('overall_score', 0.0)):.6f}"
        )
    lines.append(f"- delta_schema_valid_rate: `{float(compare.get('delta_schema_valid_rate', 0.0)):+.6f}`")
    lines.append(f"- delta_overall_score: `{float(compare.get('delta_overall_score', 0.0)):+.6f}`")

    formal = payload["supporting"].get("formal_assessment_summary") or {}
    lines.extend([
        "",
        "## Formal Assessment Summary",
        f"- available: `{bool(formal.get('available', False))}`",
        f"- generated_at_utc: `{formal.get('generated_at_utc')}`",
        f"- lean_compile_success: `{bool(formal.get('lean_compile_success', False))}`",
    ])
    facts = formal.get("facts") or {}
    if facts:
        lines.append(f"- facts: `{facts}`")
    distill = formal.get("distillation") or {}
    if distill:
        lines.append(
            "- "
            f"distill sessions={int(distill.get('sessions', 0))}, "
            f"schema_valid_rate={float(distill.get('schema_valid_rate', 0.0)):.6f}, "
            f"overall_score={float(distill.get('overall_score', 0.0)):.6f}, "
            f"delta_schema_valid_rate={float(distill.get('pipeline_delta_schema_valid_rate', 0.0)):+.6f}"
        )
    pub = formal.get("public_validation") or {}
    if pub:
        lines.append(
            "- "
            f"public baseline_gate_ok={bool(pub.get('baseline_gate_ok', False))}, "
            f"longrun_gate_ok={bool(pub.get('longrun_gate_ok', False))}, "
            f"baseline_alignment={float(pub.get('baseline_alignment_overall', 0.0)):.6f}, "
            f"longrun_alignment={float(pub.get('longrun_alignment_overall', 0.0)):.6f}"
        )

    systemic = payload["supporting"].get("systemic_joint_summary") or {}
    lines.extend([
        "",
        "## Systemic Joint Capability Summary",
        f"- available: `{bool(systemic.get('available', False))}`",
        f"- generated_at_utc: `{systemic.get('generated_at_utc')}`",
        f"- self_problem: `{systemic.get('self_problem')}`",
        f"- aggregate_score: `{float(systemic.get('aggregate_score', 0.0)):.6f}`",
        f"- cv_min_score: `{float(systemic.get('cv_min_score', 0.0)):.6f}`",
        f"- cv_std_score: `{float(systemic.get('cv_std_score', 0.0)):.6f}`",
        f"- all_steps_ok: `{bool(systemic.get('all_steps_ok', False))}`",
        f"- robust_claim: `{bool(systemic.get('robust_claim', False))}`",
    ])
    if systemic.get("statement"):
        lines.append(f"- statement: {systemic.get('statement')}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")
    print(f"PNG: {out_png}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest MD: {latest_md}")
    print(f"Latest PNG: {latest_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
