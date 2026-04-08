#!/usr/bin/env python3
"""Reality-first project finalization pipeline.

Goals:
1) Capture Git sync status before execution.
2) Evaluate expected capabilities against current latest artifacts.
3) Map unmet capabilities to concrete instance-development actions.
4) Optionally execute selected instances and re-evaluate.
5) Emit timestamped + latest JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _safe_run(
    cmd: Sequence[str],
    cwd: Path,
    timeout: int,
) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout)),
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -2,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTIMEOUT",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def _run_git(args: Sequence[str]) -> Tuple[int, str, str]:
    result = _safe_run(["git", *args], cwd=ROOT, timeout=120)
    return int(result["returncode"]), str(result.get("stdout", "")), str(result.get("stderr", ""))


def _parse_divergence(text: str) -> Tuple[int, int]:
    chunks = text.strip().split()
    if len(chunks) < 2:
        return (0, 0)
    try:
        left = int(chunks[0])
        right = int(chunks[1])
        return (left, right)
    except Exception:
        return (0, 0)


def _collect_git_sync(fetch_first: bool, base_branch: str) -> Dict[str, Any]:
    fetch = {"attempted": False, "ok": False, "stdout_tail": "", "stderr_tail": ""}
    if fetch_first:
        fetch["attempted"] = True
        rc, out, err = _run_git(["fetch", "--all", "--prune"])
        fetch["ok"] = rc == 0
        fetch["stdout_tail"] = _tail(out)
        fetch["stderr_tail"] = _tail(err)

    rc_branch, out_branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = out_branch.strip() if rc_branch == 0 else ""

    rc_head, out_head, _ = _run_git(["rev-parse", "HEAD"])
    head = out_head.strip() if rc_head == 0 else ""

    rc_upstream, out_upstream, _ = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    upstream = out_upstream.strip() if rc_upstream == 0 else ""

    rc_status, out_status, _ = _run_git(["status", "--porcelain"])
    dirty_lines = [x for x in out_status.splitlines() if x.strip()] if rc_status == 0 else []

    base_behind, base_ahead = (0, 0)
    if base_branch:
        rc_div, out_div, _ = _run_git(["rev-list", "--left-right", "--count", f"{base_branch}...HEAD"])
        if rc_div == 0:
            base_behind, base_ahead = _parse_divergence(out_div)

    up_behind, up_ahead = (0, 0)
    if upstream:
        rc_up_div, out_up_div, _ = _run_git(["rev-list", "--left-right", "--count", f"{upstream}...HEAD"])
        if rc_up_div == 0:
            up_behind, up_ahead = _parse_divergence(out_up_div)

    return {
        "checked_at_utc": _now_utc(),
        "fetch": fetch,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "dirty_file_count": len(dirty_lines),
        "dirty_preview": dirty_lines[:40],
        "base_branch": base_branch,
        "divergence_vs_base": {
            "behind": base_behind,
            "ahead": base_ahead,
            "synced_or_ahead": base_behind == 0,
        },
        "divergence_vs_upstream": {
            "behind": up_behind,
            "ahead": up_ahead,
            "fully_synced": (up_behind == 0 and up_ahead == 0) if upstream else False,
        },
    }


def _artifact_paths() -> Dict[str, List[Path]]:
    return {
        "release_gate": [
            REPORTS / "release_gate_latest.json",
            REPORTS / "release_gate_instantiation_latest.json",
            REPORTS / "release_gate_post_longrun_latest.json",
        ],
        "capability_registry": [
            REPORTS / "capability_registry_latest.json",
        ],
        "public_alignment": [
            REPORTS / "public_alignment_report_latest.json",
            REPORTS / "public_alignment_instantiation_latest.json",
            REPORTS / "public_alignment_post_longrun_latest.json",
        ],
        "interactive_reasoning": [
            REPORTS / "interactive_reasoning_benchmark_latest.json",
            REPORTS / "interactive_reasoning_benchmark_model_public_latest.json",
        ],
        "systemic_joint": [
            REPORTS / "systemic_platform_joint_capability_latest.json",
        ],
        "research_cv": [
            REPORTS / "research_aggregation_cross_validation_latest.json",
        ],
        "formal_assessment": [
            REPORTS / "distill_evo_public_formal_assessment_latest.json",
        ],
        "integrated_validation": [
            REPORTS / "agi_integrated_validation_latest.json",
            REPORTS / "distill_evo_public_validation_latest.json",
        ],
        "distill_pipeline": [
            REPORTS / "self_eval_distillation_pipeline_latest.json",
        ],
        "fusion_pathway": [
            REPORTS / "open_source_model_fusion_pathway_latest.json",
        ],
    }


def _collect_artifacts() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, options in _artifact_paths().items():
        p = _first_existing(options)
        out[key] = {
            "path": str(p) if p else "",
            "exists": bool(p and p.exists()),
            "data": _load_json(p) if p else {},
        }
    return out


def _extract_signals(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    release = artifacts.get("release_gate", {}).get("data", {})
    release_signals = release.get("signals", {}) if isinstance(release, dict) else {}

    capability = artifacts.get("capability_registry", {}).get("data", {})
    capability_caps = capability.get("capabilities", {}) if isinstance(capability, dict) else {}

    interactive = artifacts.get("interactive_reasoning", {}).get("data", {})
    interactive_metrics = interactive.get("metrics", {}) if isinstance(interactive, dict) else {}

    public = artifacts.get("public_alignment", {}).get("data", {})
    public_alignment = public.get("alignment", {}) if isinstance(public, dict) else {}

    integrated = artifacts.get("integrated_validation", {}).get("data", {})
    integrated_base = integrated.get("baseline_metrics", {}) if isinstance(integrated, dict) else {}
    integrated_long = integrated.get("longrun_metrics", {}) if isinstance(integrated, dict) else {}

    systemic = artifacts.get("systemic_joint", {}).get("data", {})
    systemic_aggregate = systemic.get("aggregate_effectiveness", {}) if isinstance(systemic, dict) else {}
    systemic_cv = systemic.get("cross_validation", {}) if isinstance(systemic, dict) else {}

    research = artifacts.get("research_cv", {}).get("data", {})
    research_aggregate = research.get("aggregate_effectiveness", {}) if isinstance(research, dict) else {}
    research_cv = research.get("cross_validation", {}) if isinstance(research, dict) else {}

    formal = artifacts.get("formal_assessment", {}).get("data", {})
    logic = formal.get("logic_closure", {}) if isinstance(formal, dict) else {}

    distill = artifacts.get("distill_pipeline", {}).get("data", {})
    distill_metrics = distill.get("metrics", {}) if isinstance(distill, dict) else {}

    fusion = artifacts.get("fusion_pathway", {}).get("data", {})
    tier_maturity = (
        fusion.get("fusion_plan", {}).get("tier_maturity", {}) if isinstance(fusion, dict) else {}
    )

    public_overall = _f(public_alignment.get("overall", 0.0), 0.0)
    if public_overall <= 0.0:
        if integrated_long:
            public_overall = 0.5 * (
                _f(integrated_base.get("alignment_overall", 0.0), 0.0)
                + _f(integrated_long.get("alignment_overall", 0.0), 0.0)
            )
        else:
            public_overall = _f(integrated_base.get("alignment_overall", 0.0), 0.0)

    breadth = _f(release_signals.get("breadth", capability_caps.get("breadth", 0.0)), 0.0)
    horizon = _f(release_signals.get("horizon", capability_caps.get("horizon", 0.0)), 0.0)
    robustness = _f(release_signals.get("robustness", capability_caps.get("robustness", 0.0)), 0.0)

    interactive_success = _f(
        interactive_metrics.get("success_rate", release_signals.get("interactive_success_rate", 0.0)),
        0.0,
    )

    stable_tiers = 0
    for tier_name in ["edge", "bridge", "core"]:
        if str(tier_maturity.get(tier_name, "")).strip().lower() == "stable":
            stable_tiers += 1
    tier_stability_ratio = stable_tiers / 3.0

    return {
        "release_gate_pass": _b(release.get("gate_ok", False), False),
        "breadth": _clip01(breadth),
        "horizon": _clip01(horizon),
        "robustness": _clip01(robustness),
        "interactive_success_rate": _clip01(interactive_success),
        "public_alignment_overall": _clip01(public_overall),
        "systemic_aggregate_score": _clip01(_f(systemic_aggregate.get("score", 0.0), 0.0)),
        "systemic_cv_min_score": _clip01(_f(systemic_cv.get("min_score", 0.0), 0.0)),
        "research_aggregate_score": _clip01(_f(research_aggregate.get("score", 0.0), 0.0)),
        "research_loo_min_score": _clip01(_f(research_cv.get("min_score", 0.0), 0.0)),
        "formal_logic_closed": _b(logic.get("all_true", logic.get("lean_compile_success", False)), False),
        "distill_schema_uplift": _clip01(max(0.0, _f(distill_metrics.get("delta_schema_valid_rate", 0.0), 0.0))),
        "fusion_tier_stability_ratio": _clip01(tier_stability_ratio),
    }


@dataclass
class CapabilityTarget:
    capability_id: str
    title: str
    kind: str
    target: Any
    weight: float
    source_hint: str
    notes: str


def _capability_targets() -> List[CapabilityTarget]:
    return [
        CapabilityTarget(
            capability_id="release_gate_pass",
            title="Release gate pass",
            kind="bool",
            target=True,
            weight=0.10,
            source_hint="reports/release_gate_latest.json",
            notes="Primary release safety and readiness gate.",
        ),
        CapabilityTarget(
            capability_id="breadth",
            title="Capability breadth",
            kind="numeric",
            target=0.70,
            weight=0.08,
            source_hint="release_gate.signals.breadth",
            notes="Coverage across heterogeneous task families.",
        ),
        CapabilityTarget(
            capability_id="horizon",
            title="Long-horizon stability",
            kind="numeric",
            target=0.85,
            weight=0.08,
            source_hint="release_gate.signals.horizon",
            notes="Ability to sustain quality over longer windows.",
        ),
        CapabilityTarget(
            capability_id="robustness",
            title="Robustness",
            kind="numeric",
            target=0.70,
            weight=0.08,
            source_hint="release_gate.signals.robustness",
            notes="Resilience against drift and controller changes.",
        ),
        CapabilityTarget(
            capability_id="interactive_success_rate",
            title="Interactive reasoning success",
            kind="numeric",
            target=0.65,
            weight=0.08,
            source_hint="interactive_reasoning.metrics.success_rate",
            notes="Tool-using and iterative reasoning quality.",
        ),
        CapabilityTarget(
            capability_id="public_alignment_overall",
            title="Public benchmark alignment",
            kind="numeric",
            target=0.65,
            weight=0.09,
            source_hint="public_alignment.alignment.overall",
            notes="Alignment against ARC/SWE/METR style dimensions.",
        ),
        CapabilityTarget(
            capability_id="systemic_aggregate_score",
            title="Systemic aggregate score",
            kind="numeric",
            target=0.85,
            weight=0.10,
            source_hint="systemic_platform_joint_capability.aggregate_effectiveness.score",
            notes="Cross-controller solution quality level.",
        ),
        CapabilityTarget(
            capability_id="systemic_cv_min_score",
            title="Systemic CV floor",
            kind="numeric",
            target=0.80,
            weight=0.08,
            source_hint="systemic_platform_joint_capability.cross_validation.min_score",
            notes="Minimum score under evidence-family ablation.",
        ),
        CapabilityTarget(
            capability_id="research_aggregate_score",
            title="Research aggregate score",
            kind="numeric",
            target=0.85,
            weight=0.08,
            source_hint="research_aggregation_cross_validation.aggregate_effectiveness.score",
            notes="Research-to-architecture integrated effectiveness.",
        ),
        CapabilityTarget(
            capability_id="research_loo_min_score",
            title="Research LOO floor",
            kind="numeric",
            target=0.80,
            weight=0.08,
            source_hint="research_aggregation_cross_validation.cross_validation.min_score",
            notes="Stability when each evidence family is removed once.",
        ),
        CapabilityTarget(
            capability_id="formal_logic_closed",
            title="Formal logic closure",
            kind="bool",
            target=True,
            weight=0.09,
            source_hint="distill_evo_public_formal_assessment.logic_closure",
            notes="Formal closure should remain valid for final claims.",
        ),
        CapabilityTarget(
            capability_id="distill_schema_uplift",
            title="Distillation schema uplift",
            kind="numeric",
            target=0.02,
            weight=0.07,
            source_hint="self_eval_distillation_pipeline.metrics.delta_schema_valid_rate",
            notes="Expected net uplift from self-eval distillation.",
        ),
        CapabilityTarget(
            capability_id="fusion_tier_stability_ratio",
            title="Fusion tier stability",
            kind="numeric",
            target=(2.0 / 3.0),
            weight=0.07,
            source_hint="open_source_model_fusion_pathway.fusion_plan.tier_maturity",
            notes="At least two tiers should be marked stable.",
        ),
    ]


def _evaluate_capabilities(signals: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_weight = 0.0
    weighted_completion = 0.0

    for target in _capability_targets():
        measured = signals.get(target.capability_id)
        status = "not_achieved"
        normalized = 0.0
        gap = 0.0

        if target.kind == "bool":
            measured_bool = _b(measured, False)
            target_bool = _b(target.target, True)
            achieved = measured_bool == target_bool
            status = "achieved" if achieved else "not_achieved"
            normalized = 1.0 if achieved else 0.0
            gap = 0.0 if achieved else 1.0
            measured_value: Any = measured_bool
        else:
            measured_float = _clip01(_f(measured, 0.0))
            target_float = max(0.0, _f(target.target, 0.0))
            achieved = measured_float >= target_float
            gap = max(0.0, target_float - measured_float)
            if achieved:
                status = "achieved"
                normalized = 1.0
            else:
                ratio = (measured_float / target_float) if target_float > 1e-12 else 0.0
                normalized = _clip01(ratio)
                status = "partial" if ratio >= 0.75 else "not_achieved"
            measured_value = measured_float

        total_weight += max(0.0, target.weight)
        weighted_completion += max(0.0, target.weight) * normalized

        rows.append(
            {
                "capability_id": target.capability_id,
                "title": target.title,
                "kind": target.kind,
                "target": target.target,
                "measured": measured_value,
                "gap": gap,
                "status": status,
                "normalized_completion": normalized,
                "weight": target.weight,
                "source_hint": target.source_hint,
                "notes": target.notes,
            }
        )

    score = (weighted_completion / total_weight) if total_weight > 0 else 0.0
    achieved_count = sum(1 for r in rows if r["status"] == "achieved")
    partial_count = sum(1 for r in rows if r["status"] == "partial")
    not_achieved_count = sum(1 for r in rows if r["status"] == "not_achieved")

    if not_achieved_count == 0 and partial_count == 0 and score >= 0.90:
        readiness = "final_candidate"
    elif score >= 0.75:
        readiness = "release_candidate_with_gaps"
    else:
        readiness = "needs_iteration"

    unresolved = [r for r in rows if r["status"] != "achieved"]
    unresolved = sorted(unresolved, key=lambda x: (-float(x["weight"]), -float(x["gap"])))

    return {
        "weighted_completion": _clip01(score),
        "counts": {
            "achieved": achieved_count,
            "partial": partial_count,
            "not_achieved": not_achieved_count,
            "total": len(rows),
        },
        "readiness": readiness,
        "capabilities": rows,
        "unresolved_priority": unresolved,
    }


def _research_and_dev_map() -> List[Dict[str, Any]]:
    return [
        {
            "trend": "SWE-bench style issue-resolution evaluation (2024 mainstream)",
            "focus": "Real-world code issue closure and patch validity",
            "local_instances": [
                "tools/run_interactive_reasoning_benchmark.py",
                "tools/run_agi_integrated_validation.py",
            ],
            "mapped_capabilities": [
                "interactive_success_rate",
                "public_alignment_overall",
            ],
        },
        {
            "trend": "Self-rewarding and judge-model loops (2024)",
            "focus": "Self-evaluation driven policy improvement",
            "local_instances": [
                "tools/run_self_eval_distillation_pipeline.py",
                "tools/run_self_eval_distillation_pipeline_chunked.py",
                "tools/train_self_eval_distillation_adapter.py",
            ],
            "mapped_capabilities": [
                "distill_schema_uplift",
                "robustness",
            ],
        },
        {
            "trend": "Agent reliability under long-horizon orchestration (2024-2025)",
            "focus": "Multi-controller consistency and failure isolation",
            "local_instances": [
                "tools/run_systemic_platform_joint_capability_assessment.py",
                "tools/dynamic_blueprint_bootstrap.py",
            ],
            "mapped_capabilities": [
                "systemic_aggregate_score",
                "systemic_cv_min_score",
                "horizon",
            ],
        },
        {
            "trend": "Formalized trust and verification stacks (2024-2026)",
            "focus": "Evidence-backed claims with machine-checkable closure",
            "local_instances": [
                "tools/run_distill_evolution_public_formal_assessment.py",
                "tools/release_gate.py",
            ],
            "mapped_capabilities": [
                "formal_logic_closed",
                "release_gate_pass",
            ],
        },
        {
            "trend": "Cost-aware small/large model routing (2024-2026)",
            "focus": "Edge-core decomposition and route policy stability",
            "local_instances": [
                "tools/run_open_source_model_fusion_pathway.py",
                "reports/open_source_model_fusion_router_latest.json",
            ],
            "mapped_capabilities": [
                "fusion_tier_stability_ratio",
                "robustness",
            ],
        },
    ]


def _resolve_python_exe(requested: str) -> str:
    req = (requested or "").strip()
    if not req:
        return sys.executable
    p = Path(req)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = (ROOT / p).resolve()
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _instance_library(py_exe: str) -> Dict[str, Dict[str, Any]]:
    return {
        "release_gate_hardening": {
            "title": "Release gate hardening run",
            "command": [
                py_exe,
                "tools/release_gate.py",
                "--profile",
                "quick",
                "--assist-provider",
                "none",
                "--min-breadth",
                "0.60",
                "--min-horizon",
                "0.80",
                "--min-robustness",
                "0.60",
                "--output-prefix",
                "reality_finalize_release_gate",
            ],
            "expected_artifacts": [
                "reports/release_gate_latest.json",
                "reports/release_gate_latest.md",
            ],
        },
        "interactive_reasoning_regression": {
            "title": "Interactive reasoning regression benchmark",
            "command": [
                py_exe,
                "tools/run_interactive_reasoning_benchmark.py",
                "--output-prefix",
                "interactive_reasoning_benchmark",
            ],
            "expected_artifacts": [
                "reports/interactive_reasoning_benchmark_latest.json",
                "reports/interactive_reasoning_benchmark_latest.md",
            ],
        },
        "systemic_joint_ci_safe": {
            "title": "Systemic joint capability reassessment (ci-safe)",
            "command": [
                py_exe,
                "tools/run_systemic_platform_joint_capability_assessment.py",
                "--ci-safe",
                "--output-prefix",
                "systemic_platform_joint_capability",
            ],
            "expected_artifacts": [
                "reports/systemic_platform_joint_capability_latest.json",
                "reports/systemic_platform_joint_capability_latest.md",
            ],
        },
        "research_cv_refresh": {
            "title": "Research aggregation cross-validation refresh",
            "command": [
                py_exe,
                "tools/run_research_aggregation_cross_validation.py",
            ],
            "expected_artifacts": [
                "reports/research_aggregation_cross_validation_latest.json",
                "reports/research_aggregation_cross_validation_latest.md",
            ],
        },
        "formal_closure_refresh": {
            "title": "Formal closure refresh",
            "command": [
                py_exe,
                "tools/run_distill_evolution_public_formal_assessment.py",
            ],
            "expected_artifacts": [
                "reports/distill_evo_public_formal_assessment_latest.json",
                "reports/distill_evo_public_formal_assessment_latest.md",
            ],
        },
        "distillation_uplift_refresh": {
            "title": "Distillation uplift pipeline refresh",
            "command": [
                py_exe,
                "tools/run_self_eval_distillation_pipeline_chunked.py",
                "--output-prefix",
                "self_eval_distillation_pipeline_chunked",
            ],
            "expected_artifacts": [
                "reports/self_eval_distillation_pipeline_chunked_latest.json",
                "reports/self_eval_distillation_pipeline_latest.json",
            ],
        },
        "fusion_pathway_refresh": {
            "title": "Open-source fusion pathway refresh",
            "command": [
                py_exe,
                "tools/run_open_source_model_fusion_pathway.py",
                "--output-prefix",
                "open_source_model_fusion_pathway",
            ],
            "expected_artifacts": [
                "reports/open_source_model_fusion_pathway_latest.json",
                "reports/open_source_model_fusion_pathway_latest.md",
            ],
        },
    }


def _capability_to_instance_ids(ci_safe: bool) -> Dict[str, List[str]]:
    mapping = {
        "release_gate_pass": ["release_gate_hardening"],
        "breadth": ["release_gate_hardening", "interactive_reasoning_regression"],
        "horizon": ["systemic_joint_ci_safe"],
        "robustness": ["systemic_joint_ci_safe", "fusion_pathway_refresh"],
        "interactive_success_rate": ["interactive_reasoning_regression"],
        "public_alignment_overall": ["interactive_reasoning_regression", "release_gate_hardening"],
        "systemic_aggregate_score": ["systemic_joint_ci_safe"],
        "systemic_cv_min_score": ["systemic_joint_ci_safe"],
        "research_aggregate_score": ["research_cv_refresh"],
        "research_loo_min_score": ["research_cv_refresh"],
        "formal_logic_closed": ["formal_closure_refresh"],
        "distill_schema_uplift": ["distillation_uplift_refresh"],
        "fusion_tier_stability_ratio": ["fusion_pathway_refresh"],
    }
    if ci_safe:
        mapping["formal_logic_closed"] = ["systemic_joint_ci_safe"]
    return mapping


def _build_instance_plan(
    evaluation: Dict[str, Any],
    py_exe: str,
    ci_safe: bool,
) -> List[Dict[str, Any]]:
    library = _instance_library(py_exe)
    cap_to_instances = _capability_to_instance_ids(ci_safe=ci_safe)

    chosen: List[str] = []
    reasons: Dict[str, List[str]] = {}

    for row in evaluation.get("unresolved_priority", []):
        cap_id = str(row.get("capability_id", ""))
        for instance_id in cap_to_instances.get(cap_id, []):
            if instance_id not in chosen:
                chosen.append(instance_id)
            reasons.setdefault(instance_id, []).append(cap_id)

    items: List[Dict[str, Any]] = []
    for idx, instance_id in enumerate(chosen, start=1):
        item = library.get(instance_id)
        if not item:
            continue
        items.append(
            {
                "order": idx,
                "instance_id": instance_id,
                "title": item["title"],
                "command": item["command"],
                "expected_artifacts": item["expected_artifacts"],
                "addresses_capabilities": sorted(set(reasons.get(instance_id, []))),
            }
        )

    return items


def _execute_instance_plan(
    plan: List[Dict[str, Any]],
    execute: bool,
    max_instances: int,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not execute:
        for p in plan:
            results.append(
                {
                    "instance_id": p["instance_id"],
                    "executed": False,
                    "status": "planned_only",
                    "reason": "execute_instances=false",
                }
            )
        return results

    run_count = max(0, int(max_instances))
    for idx, p in enumerate(plan):
        if idx >= run_count:
            results.append(
                {
                    "instance_id": p["instance_id"],
                    "executed": False,
                    "status": "skipped",
                    "reason": "max_instances limit",
                }
            )
            continue

        run = _safe_run(p["command"], cwd=ROOT, timeout=max(30, int(timeout_sec)))
        results.append(
            {
                "instance_id": p["instance_id"],
                "executed": True,
                "status": "ok" if run["ok"] else "failed",
                "returncode": run["returncode"],
                "command": p["command"],
                "stdout_tail": _tail(str(run.get("stdout", "")), lines=50),
                "stderr_tail": _tail(str(run.get("stderr", "")), lines=50),
            }
        )

    return results


def _finalization_statement(evaluation: Dict[str, Any], git_sync: Dict[str, Any]) -> str:
    readiness = str(evaluation.get("readiness", ""))
    weighted = _f(evaluation.get("weighted_completion", 0.0), 0.0)
    counts = evaluation.get("counts", {})
    unresolved = int(counts.get("not_achieved", 0) or 0)
    partial = int(counts.get("partial", 0) or 0)
    unresolved_total = unresolved + partial
    base_ok = bool(git_sync.get("divergence_vs_base", {}).get("synced_or_ahead", False))

    if readiness == "final_candidate" and unresolved_total == 0 and base_ok:
        return "Project is a final candidate under current targets with no unresolved capability gaps."
    if readiness == "release_candidate_with_gaps" and base_ok:
        return (
            "Project reaches release-candidate level but still has unresolved gaps; "
            "final claim should remain conditional."
        )
    if weighted >= 0.60:
        return "Project is partially validated; further targeted iteration is required before final claim."
    return "Project is not ready for final claim; major capability gaps remain."


def _render_markdown(payload: Dict[str, Any]) -> str:
    before = payload.get("evaluation_before", {})
    after = payload.get("evaluation_after", {})
    git_sync = payload.get("git_sync", {})
    plan = payload.get("instance_plan", [])
    execution = payload.get("instance_execution", [])

    lines: List[str] = [
        "# Project Reality Finalization Report",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`",
        f"- weighted_completion_before: `{_f(before.get('weighted_completion', 0.0)):.6f}`",
        f"- weighted_completion_after: `{_f(after.get('weighted_completion', 0.0)):.6f}`",
        f"- readiness_after: `{after.get('readiness', '')}`",
        "",
        "## Git Sync Status",
        f"- branch: `{git_sync.get('branch', '')}`",
        f"- head: `{git_sync.get('head', '')}`",
        f"- upstream: `{git_sync.get('upstream', '')}`",
        f"- dirty_file_count: `{git_sync.get('dirty_file_count', 0)}`",
        f"- base_branch: `{git_sync.get('base_branch', '')}`",
        (
            "- divergence_vs_base: "
            f"behind=`{git_sync.get('divergence_vs_base', {}).get('behind', 0)}`, "
            f"ahead=`{git_sync.get('divergence_vs_base', {}).get('ahead', 0)}`"
        ),
        (
            "- divergence_vs_upstream: "
            f"behind=`{git_sync.get('divergence_vs_upstream', {}).get('behind', 0)}`, "
            f"ahead=`{git_sync.get('divergence_vs_upstream', {}).get('ahead', 0)}`"
        ),
        "",
        "## Unmet Capabilities (After)",
    ]

    unresolved_after = after.get("unresolved_priority", [])
    if unresolved_after:
        for row in unresolved_after:
            lines.append(
                "- "
                f"{row.get('title', row.get('capability_id', ''))}: "
                f"target={row.get('target')}, measured={row.get('measured')}, "
                f"status={row.get('status')}, gap={row.get('gap')}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Instance Development Plan"])
    if plan:
        for item in plan:
            lines.append(
                "- "
                f"{item.get('instance_id', '')}: {item.get('title', '')}; "
                f"addresses={','.join(item.get('addresses_capabilities', []))}"
            )
            lines.append(f"  command: `{' '.join(item.get('command', []))}`")
    else:
        lines.append("- no instance actions required")

    lines.extend(["", "## Instance Execution"])
    if execution:
        for item in execution:
            lines.append(
                "- "
                f"{item.get('instance_id', '')}: executed={item.get('executed')}, "
                f"status={item.get('status')}"
            )
    else:
        lines.append("- no execution records")

    lines.extend(
        [
            "",
            "## Research/Development Integration",
        ]
    )
    for row in payload.get("research_and_dev_map", []):
        lines.append(f"- {row.get('trend', '')}: {row.get('focus', '')}")
        lines.append(f"  instances: `{', '.join(row.get('local_instances', []))}`")
        lines.append(f"  mapped_capabilities: `{', '.join(row.get('mapped_capabilities', []))}`")

    lines.extend(
        [
            "",
            "## Final Statement",
            f"- {payload.get('final_statement', '')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reality-first project finalization")
    parser.add_argument("--output-prefix", default="project_reality_finalization")
    parser.add_argument("--base-branch", default="origin/main")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip git fetch --all --prune before the sync snapshot.",
    )
    parser.add_argument(
        "--execute-instances",
        action="store_true",
        help="Execute generated instance-development actions.",
    )
    parser.add_argument("--max-instances", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument(
        "--ci-safe",
        action="store_true",
        help="Use CI-safe mapping for expensive/formal steps.",
    )
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    git_sync = _collect_git_sync(fetch_first=(not args.no_fetch), base_branch=args.base_branch)

    artifacts_before = _collect_artifacts()
    signals_before = _extract_signals(artifacts_before)
    evaluation_before = _evaluate_capabilities(signals_before)

    py_exe = _resolve_python_exe(args.python)
    instance_plan = _build_instance_plan(evaluation_before, py_exe=py_exe, ci_safe=bool(args.ci_safe))
    instance_execution = _execute_instance_plan(
        instance_plan,
        execute=bool(args.execute_instances),
        max_instances=max(0, args.max_instances),
        timeout_sec=max(30, args.timeout_sec),
    )

    artifacts_after = _collect_artifacts()
    signals_after = _extract_signals(artifacts_after)
    evaluation_after = _evaluate_capabilities(signals_after)

    final_statement = _finalization_statement(evaluation_after, git_sync=git_sync)

    payload: Dict[str, Any] = {
        "generated_at_utc": _now_utc(),
        "meta": {
            "output_prefix": args.output_prefix,
            "base_branch": args.base_branch,
            "fetch_before_sync": not args.no_fetch,
            "execute_instances": bool(args.execute_instances),
            "max_instances": max(0, args.max_instances),
            "timeout_sec": max(30, args.timeout_sec),
            "python_executable": py_exe,
            "ci_safe": bool(args.ci_safe),
        },
        "git_sync": git_sync,
        "artifact_sources_before": {k: v.get("path", "") for k, v in artifacts_before.items()},
        "artifact_sources_after": {k: v.get("path", "") for k, v in artifacts_after.items()},
        "signals_before": signals_before,
        "signals_after": signals_after,
        "evaluation_before": evaluation_before,
        "evaluation_after": evaluation_after,
        "instance_plan": instance_plan,
        "instance_execution": instance_execution,
        "research_and_dev_map": _research_and_dev_map(),
        "final_statement": final_statement,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    out_md.write_text(_render_markdown(payload), encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Weighted completion (after): {evaluation_after.get('weighted_completion', 0.0):.6f}")
    print(f"Readiness (after): {evaluation_after.get('readiness', '')}")
    print(f"Final statement: {final_statement}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
