#!/usr/bin/env python3
"""Unified capability framework for H2Q-Evo.

This script does not replace existing modules. It integrates them by:
1) Registering capabilities with clear ownership and acceptance criteria.
2) Mapping each capability to public and reproducible task patterns.
3) Collecting latest evidence artifacts from reports/.
4) Producing a unified robustness assessment in JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@dataclass
class Capability:
    capability_id: str
    name: str
    layer: str
    owner_module: str
    entrypoint: str
    description: str
    acceptance_metric: str
    acceptance_target: str
    evidence_glob: str
    public_task: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_file(glob_pat: str) -> Optional[Path]:
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_optional_trust_center(profile: str, skip_rsa: bool) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "trusted_joint_agi_quantum_center.py"),
        "--profile",
        profile,
    ]
    if skip_rsa:
        cmd.append("--skip-rsa")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def registry() -> List[Capability]:
    return [
        Capability(
            capability_id="svc_local_infer",
            name="Local Inference Service",
            layer="serving",
            owner_module="h2q_project/h2q_server.py",
            entrypoint="/chat, /generate, /health",
            description="Provide local model inference endpoints with DAS and legacy fallback paths.",
            acceptance_metric="service availability + non-empty response",
            acceptance_target="health endpoint active and generation returns effective text",
            evidence_glob="agi_self_evolution_round_*.json",
            public_task="Prompted plan generation and structured JSON generation",
        ),
        Capability(
            capability_id="trust_joint_center",
            name="Trusted Joint Orchestration",
            layer="validation",
            owner_module="tools/trusted_joint_agi_quantum_center.py",
            entrypoint="run_center(profile, include_rsa, rsa_folds)",
            description="Aggregate multi-stage trust gates over implemented modules.",
            acceptance_metric="trust_score + gate vector",
            acceptance_target="trusted_ready=true and trust_score>=0.75",
            evidence_glob="trusted_joint_agi_quantum_center_*.json",
            public_task="Reproducible integrated technical validation report",
        ),
        Capability(
            capability_id="self_evolution_loop",
            name="Self Evolution Daemon",
            layer="autonomy",
            owner_module="tools/agi_self_evolution_daemon.py",
            entrypoint="main()",
            description="Continuous rounds with dual-threshold acceptance and adaptive controls.",
            acceptance_metric="overall_ratio + core_ratio",
            acceptance_target="overall>=0.75 and core>=1.0 (configurable)",
            evidence_glob="agi_self_evolution_round_*.json",
            public_task="Continuous autonomous iteration with daily report",
        ),
        Capability(
            capability_id="external_assist",
            name="External LLM Assist (DeepSeek)",
            layer="augmentation",
            owner_module="tools/agi_self_evolution_daemon.py",
            entrypoint="_external_assist()",
            description="Optional external API assist with traffic budget and fallback logic.",
            acceptance_metric="assist success_rate + token budget discipline",
            acceptance_target="success_rate stable and no uncontrolled budget overflow",
            evidence_glob="agi_self_evolution_daily_*.json",
            public_task="External-assisted task completion under budget constraints",
        ),
        Capability(
            capability_id="docker_consistency",
            name="Local-vs-Docker Consistency Check",
            layer="reliability",
            owner_module="tools/agi_self_evolution_daemon.py",
            entrypoint="_docker_consistency_check()",
            description="Cross-runtime consistency sampling with overlap threshold.",
            acceptance_metric="docker_consistency.ok + overlap",
            acceptance_target="ok=true and overlap>=configured threshold",
            evidence_glob="agi_self_evolution_round_*.json",
            public_task="Runtime reproducibility across local and container execution",
        ),
        Capability(
            capability_id="realtime_monitor",
            name="Realtime Evolution Monitoring",
            layer="observability",
            owner_module="tools/agi_realtime_monitor.py",
            entrypoint="main()",
            description="Periodic snapshots and lookback-based aggregate metrics.",
            acceptance_metric="snapshot freshness + metric completeness",
            acceptance_target="latest monitor artifacts generated on schedule",
            evidence_glob="agi_realtime_monitor_latest.json",
            public_task="Operational telemetry for long-running autonomous loops",
        ),
        Capability(
            capability_id="hourly_diagnosis",
            name="Hourly Trend Diagnosis",
            layer="observability",
            owner_module="tools/agi_realtime_monitor.py",
            entrypoint="_compute_hourly_diagnosis()",
            description="Hourly bucket trend deltas for success and token efficiency.",
            acceptance_metric="hour buckets + deltas present",
            acceptance_target="continuous hourly diagnosis artifact",
            evidence_glob="agi_realtime_monitor_hourly_diagnosis_latest.json",
            public_task="Time-series diagnosis and anomaly interpretation",
        ),
        Capability(
            capability_id="quantum_crossval",
            name="Quantum Supremacy Cross Validation",
            layer="algorithm-benchmark",
            owner_module="tools/quantum_supremacy_crossval_analysis.py",
            entrypoint="main()",
            description="Large-scale cross-validation for quantum supremacy style claims.",
            acceptance_metric="cross-validation stability and reproducibility",
            acceptance_target="public benchmark result artifacts reproducible",
            evidence_glob="*quantum*report*.json",
            public_task="Public benchmark style reproducible analysis",
        ),
        Capability(
            capability_id="np_hard_suite",
            name="NP-Hard MAX-CUT Public Suite",
            layer="algorithm-benchmark",
            owner_module="tools/np_hard_maxcut_quantum_advantage.py",
            entrypoint="main()",
            description="Deterministic public MAX-CUT benchmark instances.",
            acceptance_metric="fixed-seed instance reproducibility",
            acceptance_target="same seed produces same metrics and verdict",
            evidence_glob="*maxcut*.json",
            public_task="公开 NP-hard 基准对照实验",
        ),
        Capability(
            capability_id="unified_audit_chain",
            name="Unified Audit Chain",
            layer="governance",
            owner_module="tools/unified_audit.py",
            entrypoint="main()",
            description="Single command chain for architecture and integration audits.",
            acceptance_metric="all audits pass",
            acceptance_target="return code 0",
            evidence_glob="*audit*report*.json",
            public_task="Release-gate audit before external demonstration",
        ),
    ]


def collect_evidence(capabilities: List[Capability]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cap in capabilities:
        ev = latest_file(cap.evidence_glob)
        rows.append(
            {
                "capability_id": cap.capability_id,
                "evidence_glob": cap.evidence_glob,
                "evidence_path": str(ev) if ev else "",
                "evidence_exists": bool(ev and ev.exists()),
                "evidence_mtime_utc": datetime.fromtimestamp(ev.stat().st_mtime, tz=timezone.utc).isoformat() if ev else "",
            }
        )
    return rows


def _score_from_bool(value: bool) -> float:
    return 1.0 if value else 0.0


def evaluate_robustness() -> Dict[str, Any]:
    latest_round = latest_file("agi_self_evolution_round_*.json")
    latest_daily = latest_file("agi_self_evolution_daily_*.json")
    latest_monitor = latest_file("agi_realtime_monitor_latest.json")
    latest_hourly = latest_file("agi_realtime_monitor_hourly_diagnosis_latest.json")
    latest_trust = latest_file("trusted_joint_agi_quantum_center_*.json")

    round_obj = _safe_load_json(latest_round).get("round", {}) if latest_round else {}
    daily_obj = _safe_load_json(latest_daily) if latest_daily else {}
    monitor_obj = _safe_load_json(latest_monitor) if latest_monitor else {}
    trust_obj = _safe_load_json(latest_trust).get("aggregate", {}) if latest_trust else {}

    acceptance_ok = bool(round_obj.get("acceptance", {}).get("success", False))
    docker_ok = bool(round_obj.get("docker_consistency", {}).get("ok", False))
    trust_ok = bool(trust_obj.get("trusted_ready", False))
    monitor_ok = bool(latest_monitor and latest_monitor.exists())
    hourly_ok = bool(latest_hourly and latest_hourly.exists())

    assist_sum = daily_obj.get("assist_summary", {})
    assist_rate = float(assist_sum.get("success_rate", 0.0) or 0.0)
    assist_tokens = int(assist_sum.get("total_tokens", 0) or 0)

    success_rate = float(monitor_obj.get("metrics", {}).get("success_rate", 0.0) or 0.0)

    availability = (_score_from_bool(acceptance_ok) + _score_from_bool(monitor_ok)) / 2.0
    consistency = (_score_from_bool(docker_ok) + _score_from_bool(trust_ok)) / 2.0
    observability = (_score_from_bool(monitor_ok) + _score_from_bool(hourly_ok)) / 2.0
    control = min(1.0, max(0.0, 0.5 * assist_rate + 0.5 * success_rate))

    # If token volume is very high while control is low, penalize budget discipline.
    budget_penalty = 0.0
    if assist_tokens > 100000 and control < 0.6:
        budget_penalty = 0.15

    weighted = 0.30 * availability + 0.30 * consistency + 0.20 * observability + 0.20 * control
    overall = max(0.0, min(1.0, weighted - budget_penalty))

    grade = "A" if overall >= 0.85 else "B" if overall >= 0.70 else "C" if overall >= 0.55 else "D"

    return {
        "availability": availability,
        "consistency": consistency,
        "observability": observability,
        "control": control,
        "budget_penalty": budget_penalty,
        "overall_score": overall,
        "grade": grade,
        "signals": {
            "acceptance_ok": acceptance_ok,
            "docker_ok": docker_ok,
            "trust_ok": trust_ok,
            "monitor_ok": monitor_ok,
            "hourly_ok": hourly_ok,
            "assist_rate": assist_rate,
            "assist_tokens": assist_tokens,
            "monitor_success_rate": success_rate,
        },
    }


def build_recommendations(robustness: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    sig = robustness.get("signals", {})

    if not sig.get("docker_ok", False):
        recs.append("Prioritize runtime consistency guardrail: keep docker consistency check enabled every round in release gate.")
    if float(sig.get("monitor_success_rate", 0.0)) < 0.8:
        recs.append("Stabilize success-rate before scaling goals: pin goal_mode=basic for at least one full lookback window.")
    if float(sig.get("assist_rate", 0.0)) < 0.9:
        recs.append("Improve external assist reliability: enforce retry with bounded backoff and classify provider-side failures explicitly.")
    if int(sig.get("assist_tokens", 0)) > 100000:
        recs.append("Tighten token governance: dynamic max_calls_per_round with hard stop when marginal success gain plateaus.")
    if not recs:
        recs.append("Current integrated framework is stable; next step is adding deterministic regression suites per capability.")
    recs.append("Create a CI profile that runs: trust center quick + one daemon round + docker consistency + monitor snapshot generation.")
    return recs


def render_markdown(payload: Dict[str, Any]) -> str:
    rows = payload["capabilities"]
    evidence = {r["capability_id"]: r for r in payload["evidence"]}
    robust = payload["robustness"]

    lines = [
        "# H2Q-Evo Unified System Framework",
        "",
        "## 1) Integration Goal",
        "",
        "Build a usable system by decoupling module ownership while unifying orchestration, acceptance, and evidence.",
        "",
        "## 2) Capability Matrix",
        "",
        "| ID | Layer | Capability | Module | Public Task | Acceptance Target | Evidence |",
        "|---|---|---|---|---|---|---|",
    ]

    for cap in rows:
        ev = evidence.get(cap["capability_id"], {})
        ev_path = ev.get("evidence_path", "") or "(missing)"
        lines.append(
            "| {id} | {layer} | {name} | `{mod}` | {task} | {target} | `{evidence}` |".format(
                id=cap["capability_id"],
                layer=cap["layer"],
                name=cap["name"],
                mod=cap["owner_module"],
                task=cap["public_task"],
                target=cap["acceptance_target"],
                evidence=ev_path,
            )
        )

    lines.extend(
        [
            "",
            "## 3) Robustness Assessment",
            "",
            f"- overall_score: `{robust['overall_score']:.3f}`",
            f"- grade: `{robust['grade']}`",
            f"- availability: `{robust['availability']:.3f}`",
            f"- consistency: `{robust['consistency']:.3f}`",
            f"- observability: `{robust['observability']:.3f}`",
            f"- control: `{robust['control']:.3f}`",
            "",
            "## 4) Improvement Priorities",
            "",
        ]
    )

    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 5) Decoupling and Unified Integration Pattern",
            "",
            "- Module Layer: each tool keeps its own logic and release cycle.",
            "- Capability Layer: this framework binds modules to explicit acceptance contracts.",
            "- Evidence Layer: all outputs converge to versioned report artifacts.",
            "- Governance Layer: unified audit and readiness score gate external claims.",
            "",
            "This avoids a fake monolith while still giving one operable system interface.",
            "",
        ]
    )

    return "\n".join(lines)


def write_outputs(payload: Dict[str, Any], prefix: str) -> Tuple[Path, Path, Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_json = REPORTS / f"{prefix}_{ts}.json"
    out_md = REPORTS / f"{prefix}_{ts}.md"
    out_latest_json = REPORTS / f"{prefix}_latest.json"
    out_latest_md = REPORTS / f"{prefix}_latest.md"

    md = render_markdown(payload)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    out_latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest_md.write_text(md, encoding="utf-8")
    return out_json, out_md, out_latest_json, out_latest_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified capability integration framework")
    parser.add_argument("--output-prefix", default="unified_system_framework")
    parser.add_argument("--refresh-trust-center", action="store_true")
    parser.add_argument("--trust-profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--trust-skip-rsa", action="store_true")
    args = parser.parse_args()

    trust_refresh: Dict[str, Any] = {"executed": False}
    if args.refresh_trust_center:
        trust_refresh = {
            "executed": True,
            "result": run_optional_trust_center(profile=args.trust_profile, skip_rsa=args.trust_skip_rsa),
        }

    caps = registry()
    evidence = collect_evidence(caps)
    robustness = evaluate_robustness()
    recommendations = build_recommendations(robustness)

    payload = {
        "meta": {
            "generated_at_utc": now_utc(),
            "framework_version": "v1",
            "workspace_root": str(ROOT),
        },
        "trust_refresh": trust_refresh,
        "capabilities": [asdict(c) for c in caps],
        "evidence": evidence,
        "robustness": robustness,
        "recommendations": recommendations,
    }

    out_json, out_md, out_latest_json, out_latest_md = write_outputs(payload, args.output_prefix)
    print("Unified framework generated")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")
    print(f"Latest JSON: {out_latest_json}")
    print(f"Latest MD:   {out_latest_md}")


if __name__ == "__main__":
    main()
