#!/usr/bin/env python3
"""Master controller for project finalization and release evidence.

This script orchestrates existing pipelines and adds:
1) explicit Git sync conclusion recording,
2) capability-gap scoring model,
3) research-mapping refresh hooks,
4) optional validation bundle execution,
5) consolidated finalization report generation.
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
from typing import Any, Dict, List, Sequence, Tuple


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


def _tail(text: str, lines: int = 50) -> str:
    return "\n".join((text or "").splitlines()[-max(1, int(lines)):])


def _safe_run(cmd: Sequence[str], cwd: Path, timeout: int) -> Dict[str, Any]:
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
            "command": list(cmd),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -2,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTIMEOUT",
            "command": list(cmd),
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "command": list(cmd),
        }


def _run_git(args: Sequence[str]) -> Tuple[int, str, str]:
    run = _safe_run(["git", *args], cwd=ROOT, timeout=120)
    return int(run["returncode"]), str(run.get("stdout", "")), str(run.get("stderr", ""))


def _parse_divergence(text: str) -> Tuple[int, int]:
    parts = text.strip().split()
    if len(parts) < 2:
        return (0, 0)
    try:
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_git_sync(base_branch: str, fetch_first: bool) -> Dict[str, Any]:
    fetch = {
        "attempted": bool(fetch_first),
        "ok": False,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if fetch_first:
        rc, out, err = _run_git(["fetch", "--all", "--prune"])
        fetch["ok"] = rc == 0
        fetch["stdout_tail"] = _tail(out)
        fetch["stderr_tail"] = _tail(err)

    rc_branch, out_branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    rc_head, out_head, _ = _run_git(["rev-parse", "HEAD"])
    rc_up, out_up, _ = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    rc_status, out_status, _ = _run_git(["status", "--porcelain"])

    branch = out_branch.strip() if rc_branch == 0 else ""
    head = out_head.strip() if rc_head == 0 else ""
    upstream = out_up.strip() if rc_up == 0 else ""
    dirty = [x for x in out_status.splitlines() if x.strip()] if rc_status == 0 else []

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

    synced_or_ahead_base = base_behind == 0
    fully_synced_upstream = (up_behind == 0 and up_ahead == 0) if upstream else False

    if not fetch["attempted"]:
        fetch_note = "fetch_skipped"
    elif fetch["ok"]:
        fetch_note = "fetch_ok"
    else:
        fetch_note = "fetch_failed"

    if not synced_or_ahead_base:
        sync_label = "behind_base_branch"
    elif len(dirty) > 0:
        sync_label = "synced_with_local_changes"
    elif fully_synced_upstream:
        sync_label = "fully_synced"
    else:
        sync_label = "synced_or_ahead"

    return {
        "checked_at_utc": _now_utc(),
        "fetch": fetch,
        "fetch_note": fetch_note,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "base_branch": base_branch,
        "dirty_file_count": len(dirty),
        "dirty_preview": dirty[:50],
        "divergence_vs_base": {
            "behind": base_behind,
            "ahead": base_ahead,
            "synced_or_ahead": synced_or_ahead_base,
        },
        "divergence_vs_upstream": {
            "behind": up_behind,
            "ahead": up_ahead,
            "fully_synced": fully_synced_upstream,
        },
        "sync_label": sync_label,
        "sync_ok_for_finalization": bool(synced_or_ahead_base),
    }


def _run_project_reality_finalization(
    py_exe: str,
    base_branch: str,
    fetch_first: bool,
    execute_instances: bool,
    max_instances: int,
    timeout_sec: int,
    ci_safe: bool,
    output_prefix: str,
) -> Dict[str, Any]:
    cmd: List[str] = [
        py_exe,
        "tools/run_project_reality_finalization.py",
        "--output-prefix",
        output_prefix,
        "--base-branch",
        base_branch,
        "--python",
        py_exe,
        "--max-instances",
        str(max(0, int(max_instances))),
        "--timeout-sec",
        str(max(30, int(timeout_sec))),
    ]
    if not fetch_first:
        cmd.append("--no-fetch")
    if execute_instances:
        cmd.append("--execute-instances")
    if ci_safe:
        cmd.append("--ci-safe")

    run = _safe_run(cmd, cwd=ROOT, timeout=max(60, int(timeout_sec) + 120))

    latest_json = REPORTS / f"{output_prefix}_latest.json"
    payload = _load_json(latest_json)
    return {
        "run": {
            "ok": run["ok"],
            "returncode": run["returncode"],
            "command": run["command"],
            "stdout_tail": _tail(str(run.get("stdout", "")), lines=80),
            "stderr_tail": _tail(str(run.get("stderr", "")), lines=80),
        },
        "latest_json": str(latest_json),
        "payload": payload,
    }


def _research_tags_for_capability(capability_id: str) -> List[str]:
    mapping = {
        "interactive_success_rate": ["SWE-bench style closure"],
        "public_alignment_overall": ["Public benchmark alignment"],
        "distill_schema_uplift": ["Self-reward and distillation loops"],
        "formal_logic_closed": ["Formal verification stack"],
        "systemic_aggregate_score": ["Long-horizon orchestration reliability"],
        "systemic_cv_min_score": ["Cross-evidence robustness"],
        "research_aggregate_score": ["Research integration effectiveness"],
        "research_loo_min_score": ["Research ablation robustness"],
        "fusion_tier_stability_ratio": ["Cost-aware model routing"],
        "release_gate_pass": ["Release governance and safety gate"],
        "breadth": ["Multi-domain capability breadth"],
        "horizon": ["Long-horizon consistency"],
        "robustness": ["Failure isolation and resilience"],
    }
    return mapping.get(capability_id, ["General capability hardening"])


def _priority_level(priority_score: float) -> str:
    if priority_score >= 0.08:
        return "critical"
    if priority_score >= 0.05:
        return "high"
    if priority_score >= 0.03:
        return "medium"
    return "low"


def _build_capability_gap_model(finalization_payload: Dict[str, Any]) -> Dict[str, Any]:
    evaluation_after = finalization_payload.get("evaluation_after", {})
    capabilities = evaluation_after.get("capabilities", [])
    instance_plan = finalization_payload.get("instance_plan", [])

    unresolved_rows: List[Dict[str, Any]] = []
    weighted_gap_total = 0.0

    for row in capabilities:
        status = str(row.get("status", "")).strip().lower()
        if status == "achieved":
            continue

        cap_id = str(row.get("capability_id", ""))
        kind = str(row.get("kind", "numeric"))
        weight = max(0.0, _f(row.get("weight", 0.0), 0.0))
        gap = max(0.0, _f(row.get("gap", 0.0), 0.0))

        if kind == "bool":
            normalized_gap = 1.0 if status != "achieved" else 0.0
        else:
            target = max(1e-8, _f(row.get("target", 0.0), 0.0))
            measured = _f(row.get("measured", 0.0), 0.0)
            normalized_gap = _clip01((target - measured) / target)

        status_factor = 1.0 if status == "not_achieved" else 0.6
        priority_score = weight * (0.65 * normalized_gap + 0.35 * status_factor)

        recommended_instances: List[str] = []
        for inst in instance_plan:
            addresses = inst.get("addresses_capabilities", [])
            if cap_id in addresses:
                recommended_instances.append(str(inst.get("instance_id", "")))

        weighted_gap_total += weight * max(normalized_gap, gap)

        unresolved_rows.append(
            {
                "capability_id": cap_id,
                "title": row.get("title", cap_id),
                "kind": kind,
                "status": status,
                "target": row.get("target"),
                "measured": row.get("measured"),
                "gap": gap,
                "normalized_gap": normalized_gap,
                "weight": weight,
                "priority_score": priority_score,
                "priority_level": _priority_level(priority_score),
                "recommended_instances": sorted(set(x for x in recommended_instances if x)),
                "research_tags": _research_tags_for_capability(cap_id),
                "notes": row.get("notes", ""),
            }
        )

    unresolved_rows = sorted(
        unresolved_rows,
        key=lambda x: (
            -_f(x.get("priority_score", 0.0), 0.0),
            -_f(x.get("weight", 0.0), 0.0),
        ),
    )

    level_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    for row in unresolved_rows:
        level = str(row.get("priority_level", "low"))
        if level not in level_counts:
            level = "low"
        level_counts[level] += 1

    total_caps = len(capabilities)
    unresolved_count = len(unresolved_rows)
    resolved_count = max(0, total_caps - unresolved_count)

    return {
        "model_version": "gap-model-v1",
        "generated_at_utc": _now_utc(),
        "summary": {
            "total_capabilities": total_caps,
            "resolved_capabilities": resolved_count,
            "unresolved_capabilities": unresolved_count,
            "priority_level_counts": level_counts,
            "weighted_gap_total": weighted_gap_total,
            "weighted_completion_after": _f(evaluation_after.get("weighted_completion", 0.0), 0.0),
            "readiness_after": evaluation_after.get("readiness", ""),
        },
        "unresolved_priority": unresolved_rows,
        "top_actions": [
            {
                "capability_id": row.get("capability_id"),
                "priority_level": row.get("priority_level"),
                "priority_score": row.get("priority_score"),
                "recommended_instances": row.get("recommended_instances", []),
            }
            for row in unresolved_rows[:8]
        ],
    }


def _run_research_refresh(py_exe: str, timeout_sec: int, ci_safe: bool, enabled: bool) -> List[Dict[str, Any]]:
    jobs: List[Tuple[str, List[str]]] = []
    if not enabled:
        return [
            {
                "name": "research_refresh",
                "executed": False,
                "status": "skipped",
                "reason": "run_research_refresh=false",
            }
        ]

    jobs.append(
        (
            "research_aggregation_cross_validation",
            [py_exe, "tools/run_research_aggregation_cross_validation.py"],
        )
    )
    jobs.append(
        (
            "systemic_platform_joint_capability",
            [
                py_exe,
                "tools/run_systemic_platform_joint_capability_assessment.py",
                "--output-prefix",
                "systemic_platform_joint_capability",
                "--timeout-sec",
                str(max(300, int(timeout_sec))),
            ]
            + (["--ci-safe"] if ci_safe else []),
        )
    )

    results: List[Dict[str, Any]] = []
    for name, cmd in jobs:
        run = _safe_run(cmd, cwd=ROOT, timeout=max(300, int(timeout_sec) + 120))
        results.append(
            {
                "name": name,
                "executed": True,
                "status": "ok" if run["ok"] else "failed",
                "returncode": run["returncode"],
                "command": run["command"],
                "stdout_tail": _tail(str(run.get("stdout", "")), lines=70),
                "stderr_tail": _tail(str(run.get("stderr", "")), lines=70),
            }
        )
    return results


def _run_validation_bundle(
    py_exe: str,
    timeout_sec: int,
    enabled: bool,
    with_longrun: bool,
    longrun_cycles: int,
    docker_policy: str,
    allow_missing_docker: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "executed": False,
            "status": "skipped",
            "reason": "run_validation=false",
            "steps": [],
            "summary": {},
        }

    steps: List[Dict[str, Any]] = []

    rg_cmd = [
        py_exe,
        "tools/release_gate.py",
        "--profile",
        "quick",
        "--docker-policy",
        str(docker_policy),
        "--assist-provider",
        "none",
        "--min-breadth",
        "0.60",
        "--min-horizon",
        "0.80",
        "--min-robustness",
        "0.60",
        "--output-prefix",
        "release_gate_master",
    ]
    if allow_missing_docker:
        rg_cmd.append("--allow-missing-docker")
    rg_run = _safe_run(rg_cmd, cwd=ROOT, timeout=max(180, int(timeout_sec)))
    steps.append(
        {
            "name": "release_gate_master",
            "status": "ok" if rg_run["ok"] else "failed",
            "returncode": rg_run["returncode"],
            "command": rg_run["command"],
            "stdout_tail": _tail(str(rg_run.get("stdout", "")), lines=80),
            "stderr_tail": _tail(str(rg_run.get("stderr", "")), lines=80),
        }
    )

    docker_diag_cmd = [
        py_exe,
        "tools/run_docker_consistency_diagnostics.py",
        "--release-gate-json",
        "reports/release_gate_master_latest.json",
        "--output-prefix",
        "docker_consistency_diagnostics_master",
    ]
    docker_diag_run = _safe_run(docker_diag_cmd, cwd=ROOT, timeout=max(60, int(timeout_sec) // 3))
    steps.append(
        {
            "name": "docker_consistency_diagnostics_master",
            "status": "ok" if docker_diag_run["ok"] else "failed",
            "returncode": docker_diag_run["returncode"],
            "command": docker_diag_run["command"],
            "stdout_tail": _tail(str(docker_diag_run.get("stdout", "")), lines=80),
            "stderr_tail": _tail(str(docker_diag_run.get("stderr", "")), lines=80),
        }
    )

    iv_cmd = [
        py_exe,
        "tools/run_agi_integrated_validation.py",
        "--python",
        py_exe,
        "--release-gate-docker-policy",
        str(docker_policy),
        "--output-prefix",
        "agi_integrated_validation_master",
    ]
    if allow_missing_docker:
        iv_cmd.append("--release-gate-allow-missing-docker")
    if with_longrun:
        iv_cmd.extend(["--with-longrun", "--longrun-cycles", str(max(1, int(longrun_cycles)))])

    iv_run = _safe_run(iv_cmd, cwd=ROOT, timeout=max(600, int(timeout_sec) * (2 if with_longrun else 1)))
    steps.append(
        {
            "name": "agi_integrated_validation_master",
            "status": "ok" if iv_run["ok"] else "failed",
            "returncode": iv_run["returncode"],
            "command": iv_run["command"],
            "stdout_tail": _tail(str(iv_run.get("stdout", "")), lines=80),
            "stderr_tail": _tail(str(iv_run.get("stderr", "")), lines=80),
        }
    )

    release_gate_latest = _load_json(REPORTS / "release_gate_master_latest.json")
    iv_latest = _load_json(REPORTS / "agi_integrated_validation_master_latest.json")

    rg_ok = _b(release_gate_latest.get("gate_ok", False), False)
    rg_meta = release_gate_latest.get("meta", {}) if isinstance(release_gate_latest, dict) else {}
    rg_signals = release_gate_latest.get("signals", {}) if isinstance(release_gate_latest, dict) else {}

    baseline = iv_latest.get("baseline_metrics", {}) if isinstance(iv_latest, dict) else {}
    longrun = iv_latest.get("longrun_metrics", {}) if isinstance(iv_latest, dict) else {}
    iv_baseline_gate = _b(baseline.get("gate_ok", False), False)
    iv_longrun_gate = _b(longrun.get("gate_ok", iv_baseline_gate), iv_baseline_gate)

    all_steps_ok = all(str(s.get("status", "")) == "ok" for s in steps)
    validation_ok = bool(all_steps_ok and rg_ok and iv_baseline_gate and iv_longrun_gate)

    return {
        "executed": True,
        "status": "ok" if validation_ok else "failed",
        "steps": steps,
        "summary": {
            "release_gate_ok": rg_ok,
            "docker_ok": _b(rg_signals.get("docker_ok", False), False),
            "docker_reason": str(rg_signals.get("docker_reason", "")),
            "docker_policy": str(rg_meta.get("docker_policy", docker_policy)),
            "docker_check_enabled": _b(rg_meta.get("docker_check_enabled", False), False),
            "docker_daemon_available": _b(rg_signals.get("docker_daemon_available", False), False),
            "allow_missing_docker": _b(rg_meta.get("allow_missing_docker", allow_missing_docker), allow_missing_docker),
            "integrated_validation_baseline_gate_ok": iv_baseline_gate,
            "integrated_validation_longrun_gate_ok": iv_longrun_gate,
            "all_steps_ok": all_steps_ok,
            "validation_ok": validation_ok,
            "artifacts": {
                "release_gate_latest": "reports/release_gate_master_latest.json",
                "integrated_validation_latest": "reports/agi_integrated_validation_master_latest.json",
                "docker_diagnostics_latest": "reports/docker_consistency_diagnostics_master_latest.json",
            },
        },
    }


def _run_axiomatic_bootstrap_report(py_exe: str, timeout_sec: int) -> Dict[str, Any]:
    cmd = [
        py_exe,
        "tools/run_directional_axiom_bootstrap_report.py",
        "--state-file",
        "autonomous_evolution_state.json",
        "--output-prefix",
        "directional_axiom_bootstrap_master",
    ]
    run = _safe_run(cmd, cwd=ROOT, timeout=max(30, int(timeout_sec) // 4))
    return {
        "executed": True,
        "status": "ok" if run["ok"] else "failed",
        "returncode": run["returncode"],
        "command": run["command"],
        "stdout_tail": _tail(str(run.get("stdout", "")), lines=80),
        "stderr_tail": _tail(str(run.get("stderr", "")), lines=80),
        "artifacts": {
            "directional_axiom_bootstrap_latest": "reports/directional_axiom_bootstrap_master_latest.json",
            "directional_axiom_bootstrap_latest_md": "reports/directional_axiom_bootstrap_master_latest.md",
        },
    }


def _consolidated_statement(
    git_sync: Dict[str, Any],
    finalization_payload: Dict[str, Any],
    gap_model: Dict[str, Any],
    validation: Dict[str, Any],
) -> str:
    readiness = str(finalization_payload.get("evaluation_after", {}).get("readiness", ""))
    unresolved = int(gap_model.get("summary", {}).get("unresolved_capabilities", 0) or 0)
    sync_ok = bool(git_sync.get("sync_ok_for_finalization", False))
    validation_ok = bool(validation.get("summary", {}).get("validation_ok", False)) if validation.get("executed") else True

    if sync_ok and unresolved == 0 and readiness == "final_candidate" and validation_ok:
        return "Finalization is ready for definitive publication under current target constraints."
    if sync_ok and readiness in {"final_candidate", "release_candidate_with_gaps"} and validation_ok:
        return "Finalization reached release-candidate state with residual capability gaps pending closure."
    if not sync_ok:
        return "Finalization is blocked by Git sync divergence against the base branch."
    if not validation_ok:
        return "Finalization remains conditional because validation bundle did not fully pass."
    return "Finalization is partially complete and requires further iterative closure of high-priority gaps."


def _render_markdown(payload: Dict[str, Any]) -> str:
    git_sync = payload.get("git_sync", {})
    finalization = payload.get("finalization", {})
    gap_model = payload.get("capability_gap_model", {})
    research_runs = payload.get("research_refresh_runs", [])
    validation = payload.get("validation", {})
    axiomatic_bootstrap = payload.get("axiomatic_bootstrap_report", {})

    lines: List[str] = [
        "# Master Project Finalization Report",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- sync_label: {git_sync.get('sync_label', '')}",
        f"- sync_ok_for_finalization: {git_sync.get('sync_ok_for_finalization', False)}",
        f"- final_readiness: {finalization.get('evaluation_after', {}).get('readiness', '')}",
        f"- weighted_completion_after: {_f(finalization.get('evaluation_after', {}).get('weighted_completion', 0.0)):.6f}",
        "",
        "## Git Sync Conclusion",
        f"- branch: {git_sync.get('branch', '')}",
        f"- head: {git_sync.get('head', '')}",
        f"- upstream: {git_sync.get('upstream', '')}",
        f"- base_branch: {git_sync.get('base_branch', '')}",
        f"- divergence_vs_base: behind={git_sync.get('divergence_vs_base', {}).get('behind', 0)}, ahead={git_sync.get('divergence_vs_base', {}).get('ahead', 0)}",
        f"- divergence_vs_upstream: behind={git_sync.get('divergence_vs_upstream', {}).get('behind', 0)}, ahead={git_sync.get('divergence_vs_upstream', {}).get('ahead', 0)}",
        f"- dirty_file_count: {git_sync.get('dirty_file_count', 0)}",
        "",
        "## Capability Gap Model",
        f"- model_version: {gap_model.get('model_version', '')}",
        f"- unresolved_capabilities: {gap_model.get('summary', {}).get('unresolved_capabilities', 0)}",
        f"- weighted_gap_total: {_f(gap_model.get('summary', {}).get('weighted_gap_total', 0.0)):.6f}",
    ]

    top_actions = gap_model.get("top_actions", [])
    if top_actions:
        lines.append("- top_actions:")
        for item in top_actions:
            lines.append(
                f"  - {item.get('capability_id', '')}: level={item.get('priority_level', '')}, score={_f(item.get('priority_score', 0.0)):.6f}, instances={','.join(item.get('recommended_instances', []))}"
            )
    else:
        lines.append("- top_actions: none")

    lines.extend(["", "## Research Mapping Refresh Runs"])
    if research_runs:
        for row in research_runs:
            if not row.get("executed", False):
                lines.append(f"- {row.get('name', '')}: status={row.get('status', '')}, reason={row.get('reason', '')}")
            else:
                lines.append(f"- {row.get('name', '')}: status={row.get('status', '')}, returncode={row.get('returncode')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Validation Summary"])
    if validation.get("executed"):
        summary = validation.get("summary", {})
        lines.append(f"- validation_ok: {summary.get('validation_ok', False)}")
        lines.append(f"- release_gate_ok: {summary.get('release_gate_ok', False)}")
        lines.append(
            f"- docker: policy={summary.get('docker_policy', '')}, daemon_available={summary.get('docker_daemon_available', False)}, check_enabled={summary.get('docker_check_enabled', False)}, allow_missing={summary.get('allow_missing_docker', False)}, ok={summary.get('docker_ok', False)}, reason={summary.get('docker_reason', '')}"
        )
        lines.append(
            f"- integrated_validation_baseline_gate_ok: {summary.get('integrated_validation_baseline_gate_ok', False)}"
        )
        lines.append(
            f"- integrated_validation_longrun_gate_ok: {summary.get('integrated_validation_longrun_gate_ok', False)}"
        )
    else:
        lines.append(f"- status: {validation.get('status', 'skipped')}")
        if validation.get("reason"):
            lines.append(f"- reason: {validation.get('reason', '')}")

    lines.extend(["", "## Axiomatic Bootstrap"])
    lines.append(f"- status: {axiomatic_bootstrap.get('status', 'unknown')}")
    lines.append(f"- returncode: {axiomatic_bootstrap.get('returncode', '')}")
    artifacts = axiomatic_bootstrap.get("artifacts", {}) if isinstance(axiomatic_bootstrap, dict) else {}
    if artifacts:
        lines.append(f"- directional_axiom_bootstrap_latest: {artifacts.get('directional_axiom_bootstrap_latest', '')}")
        lines.append(f"- directional_axiom_bootstrap_latest_md: {artifacts.get('directional_axiom_bootstrap_latest_md', '')}")

    lines.extend(["", "## Final Statement", f"- {payload.get('final_statement', '')}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Master project finalization controller")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--base-branch", default="origin/main")
    parser.add_argument("--output-prefix", default="master_project_finalization")
    parser.add_argument("--finalization-prefix", default="project_reality_finalization")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--max-instances", type=int, default=2)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--execute-instances", action="store_true")
    parser.add_argument("--ci-safe", action="store_true")
    parser.add_argument("--run-research-refresh", action="store_true")
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--validation-with-longrun", action="store_true")
    parser.add_argument("--validation-longrun-cycles", type=int, default=2)
    parser.add_argument(
        "--validation-docker-policy",
        choices=["auto", "strict", "allow-missing"],
        default="auto",
    )
    parser.add_argument("--validation-allow-missing-docker", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    py_exe = _resolve_python_exe(args.python)
    fetch_first = not bool(args.no_fetch)

    git_sync = _collect_git_sync(base_branch=args.base_branch, fetch_first=fetch_first)

    finalization = _run_project_reality_finalization(
        py_exe=py_exe,
        base_branch=args.base_branch,
        fetch_first=fetch_first,
        execute_instances=bool(args.execute_instances),
        max_instances=max(0, args.max_instances),
        timeout_sec=max(60, args.timeout_sec),
        ci_safe=bool(args.ci_safe),
        output_prefix=args.finalization_prefix,
    )

    finalization_payload = finalization.get("payload", {})
    if not isinstance(finalization_payload, dict) or not finalization_payload:
        payload = {
            "generated_at_utc": _now_utc(),
            "meta": {
                "output_prefix": args.output_prefix,
                "base_branch": args.base_branch,
                "python_executable": py_exe,
            },
            "git_sync": git_sync,
            "finalization_run": finalization.get("run", {}),
            "final_statement": "Master controller failed because base finalization output was not generated.",
        }
        ts = int(time.time())
        out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
        latest_json = REPORTS / f"{args.output_prefix}_latest.json"
        out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
        latest_md = REPORTS / f"{args.output_prefix}_latest.md"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(out_json, latest_json)
        out_md.write_text("# Master Project Finalization Report\n\nBase finalization output missing.\n", encoding="utf-8")
        shutil.copy2(out_md, latest_md)
        print(f"JSON: {out_json}")
        print(f"Latest JSON: {latest_json}")
        print(f"MD: {out_md}")
        print(f"Latest MD: {latest_md}")
        return 1

    gap_model = _build_capability_gap_model(finalization_payload)

    research_runs = _run_research_refresh(
        py_exe=py_exe,
        timeout_sec=max(300, args.timeout_sec),
        ci_safe=bool(args.ci_safe),
        enabled=bool(args.run_research_refresh),
    )

    validation = _run_validation_bundle(
        py_exe=py_exe,
        timeout_sec=max(300, args.timeout_sec),
        enabled=bool(args.run_validation),
        with_longrun=bool(args.validation_with_longrun),
        longrun_cycles=max(1, args.validation_longrun_cycles),
        docker_policy=str(args.validation_docker_policy),
        allow_missing_docker=bool(args.validation_allow_missing_docker),
    )

    axiomatic_bootstrap = _run_axiomatic_bootstrap_report(
        py_exe=py_exe,
        timeout_sec=max(120, args.timeout_sec),
    )

    statement = _consolidated_statement(
        git_sync=git_sync,
        finalization_payload=finalization_payload,
        gap_model=gap_model,
        validation=validation,
    )

    payload = {
        "generated_at_utc": _now_utc(),
        "meta": {
            "output_prefix": args.output_prefix,
            "base_branch": args.base_branch,
            "python_executable": py_exe,
            "fetch_before_sync": fetch_first,
            "execute_instances": bool(args.execute_instances),
            "run_research_refresh": bool(args.run_research_refresh),
            "run_validation": bool(args.run_validation),
            "validation_with_longrun": bool(args.validation_with_longrun),
            "validation_longrun_cycles": max(1, args.validation_longrun_cycles),
            "validation_docker_policy": str(args.validation_docker_policy),
            "validation_allow_missing_docker": bool(args.validation_allow_missing_docker),
            "ci_safe": bool(args.ci_safe),
        },
        "git_sync": git_sync,
        "finalization_run": finalization.get("run", {}),
        "finalization_latest_json": finalization.get("latest_json", ""),
        "finalization": {
            "evaluation_before": finalization_payload.get("evaluation_before", {}),
            "evaluation_after": finalization_payload.get("evaluation_after", {}),
            "instance_plan": finalization_payload.get("instance_plan", []),
            "instance_execution": finalization_payload.get("instance_execution", []),
            "research_and_dev_map": finalization_payload.get("research_and_dev_map", []),
            "final_statement": finalization_payload.get("final_statement", ""),
        },
        "capability_gap_model": gap_model,
        "research_refresh_runs": research_runs,
        "validation": validation,
        "axiomatic_bootstrap_report": axiomatic_bootstrap,
        "final_statement": statement,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    out_md.write_text(_render_markdown(payload), encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Sync label: {git_sync.get('sync_label', '')}")
    print(f"Final statement: {statement}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
