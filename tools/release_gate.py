#!/usr/bin/env python3
"""Release gate runner for unified AGI system evidence.

Pipeline:
1) Trusted center quick validation.
2) One self-evolution round with docker consistency enabled.
3) Realtime monitor single snapshot (lookback window).
4) Unified framework aggregation.

Produces machine-readable gate result and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _latest(glob_pat: str) -> Optional[Path]:
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_assist_key_file(key_file: str) -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return key_file
    p = Path(key_file)
    return key_file if p.exists() and p.is_file() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified release gate")
    parser.add_argument("--lookback-rounds", type=int, default=48)
    parser.add_argument("--docker-image", default="h2q-sandbox")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--allow-missing-docker", action="store_true")
    parser.add_argument("--assist-provider", choices=["none", "deepseek"], default="none")
    parser.add_argument("--assist-model", default="deepseek-chat")
    parser.add_argument("--assist-base-url", default="https://api.deepseek.com")
    parser.add_argument("--assist-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--assist-min-success-rate", type=float, default=0.20)
    parser.add_argument("--assist-min-calls", type=int, default=1)
    parser.add_argument("--assist-max-calls-per-round", type=int, default=6)
    parser.add_argument("--assist-max-est-tokens-per-round", type=int, default=16000)
    parser.add_argument("--assist-max-est-tokens-total", type=int, default=120000)
    parser.add_argument("--assist-max-tokens", type=int, default=640)
    parser.add_argument("--assist-retries", type=int, default=2)
    parser.add_argument("--assist-retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--assist-retry-backoff-max-seconds", type=float, default=8.0)
    parser.add_argument("--min-breadth", type=float, default=0.60)
    parser.add_argument("--min-horizon", type=float, default=0.80)
    parser.add_argument("--min-robustness", type=float, default=0.60)
    parser.add_argument("--output-prefix", default="release_gate")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    steps: Dict[str, Dict[str, Any]] = {}

    steps["trusted_center"] = _run(
        [
            sys.executable,
            "tools/trusted_joint_agi_quantum_center.py",
            "--profile",
            args.profile,
            "--skip-rsa",
        ]
    )

    daemon_cmd = [
        sys.executable,
        "tools/agi_self_evolution_daemon.py",
        "--rounds",
        "1",
        "--interval-minutes",
        "0.01",
        "--profile",
        args.profile,
        "--goal-mode",
        "basic",
        "--basic-lock-rounds",
        str(max(0, args.lookback_rounds)),
        "--external-goal-limit",
        "2",
        "--assist-provider",
        args.assist_provider,
        "--enable-docker-consistency-check",
        "--docker-check-interval-rounds",
        "1",
        "--docker-image",
        args.docker_image,
        "--docker-min-overlap",
        "0.05",
        "--min-overall-success-ratio",
        "0.75",
        "--min-core-success-ratio",
        "1.0",
        "--fail-on-empty",
    ]

    if args.assist_provider == "deepseek":
        key_file = _resolve_assist_key_file(args.assist_key_file)
        if not key_file:
            payload = {
                "meta": {
                    "created_at_utc": _now_utc(),
                    "lookback_rounds": max(0, args.lookback_rounds),
                    "docker_image": args.docker_image,
                    "allow_missing_docker": bool(args.allow_missing_docker),
                    "assist_provider": args.assist_provider,
                    "assist_key_file": args.assist_key_file,
                },
                "steps": {},
                "evidence": {},
                "signals": {
                    "assist_gate_ok": False,
                    "assist_reason": "deepseek-key-missing",
                },
                "gate_ok": False,
            }
            ts = int(time.time())
            out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
            out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
            out_latest_json = REPORTS / f"{args.output_prefix}_latest.json"
            out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

            lines = [
                "# Unified Release Gate",
                "",
                f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
                "- gate_ok: `False`",
                "- assist_provider: `deepseek`",
                f"- assist_reason: `{payload['signals']['assist_reason']}`",
                f"- assist_key_file: `{args.assist_key_file}`",
                "",
            ]

            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            out_md.write_text("\n".join(lines), encoding="utf-8")
            out_latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            out_latest_md.write_text("\n".join(lines), encoding="utf-8")
            print("Release gate completed")
            print("Gate OK: False")
            print("Reason: deepseek-key-missing")
            print(f"JSON: {out_json}")
            print(f"MD: {out_md}")
            raise SystemExit(1)

        daemon_cmd.extend(
            [
                "--assist-model",
                args.assist_model,
                "--assist-base-url",
                args.assist_base_url,
                "--assist-key-file",
                key_file,
                "--assist-max-calls-per-round",
                str(max(1, args.assist_max_calls_per_round)),
                "--assist-max-est-tokens-per-round",
                str(max(0, args.assist_max_est_tokens_per_round)),
                "--assist-max-est-tokens-total",
                str(max(0, args.assist_max_est_tokens_total)),
                "--assist-max-tokens",
                str(max(16, args.assist_max_tokens)),
                "--assist-retries",
                str(max(0, args.assist_retries)),
                "--assist-retry-backoff-seconds",
                str(max(0.0, args.assist_retry_backoff_seconds)),
                "--assist-retry-backoff-max-seconds",
                str(max(0.0, args.assist_retry_backoff_max_seconds)),
            ]
        )

    steps["daemon_round"] = _run(daemon_cmd)

    steps["monitor_snapshot"] = _run(
        [
            sys.executable,
            "tools/agi_realtime_monitor.py",
            "--interval-seconds",
            "5",
            "--cycles",
            "1",
            "--lookback-rounds",
            str(max(0, args.lookback_rounds)),
        ]
    )

    steps["unified_framework"] = _run([sys.executable, "tools/unified_system_framework.py"])
    steps["capability_registry"] = _run([sys.executable, "tools/capability_registry.py"])

    trust_path = _latest("trusted_joint_agi_quantum_center_*.json")
    round_path = _latest("agi_self_evolution_round_*.json")
    monitor_path = _latest("agi_realtime_monitor_latest.json")
    framework_path = _latest("unified_system_framework_latest.json")
    capability_path = _latest("capability_registry_latest.json")
    interactive_path = _latest("interactive_reasoning_benchmark_latest.json")

    trust_obj = _load_json(trust_path)
    round_obj = _load_json(round_path)
    monitor_obj = _load_json(monitor_path)
    framework_obj = _load_json(framework_path)
    capability_obj = _load_json(capability_path)
    interactive_obj = _load_json(interactive_path)

    trust_ok = bool(trust_obj.get("aggregate", {}).get("trusted_ready", False))
    acceptance_ok = bool(round_obj.get("round", {}).get("acceptance", {}).get("success", False))
    docker_section = round_obj.get("round", {}).get("docker_consistency", {})
    docker_ok = bool(docker_section.get("ok", False))
    docker_reason = str(docker_section.get("reason", ""))
    monitor_ok = bool(monitor_obj.get("metrics", {}).get("round_count", 0) >= 1)
    framework_score = float(framework_obj.get("robustness", {}).get("overall_score", 0.0) or 0.0)
    round_section = round_obj.get("round", {})
    assist_summary = round_section.get("assist_summary", {})
    assist_calls = int(assist_summary.get("enabled_calls", 0) or 0)
    assist_success_calls = int(assist_summary.get("success_calls", 0) or 0)
    assist_success_rate = float(assist_summary.get("success_rate", 0.0) or 0.0)
    if assist_calls <= 0:
        entries = round_section.get("entries", []) if isinstance(round_section, dict) else []
        for entry in entries:
            runtime = entry.get("runtime", {}) if isinstance(entry, dict) else {}
            assist = runtime.get("assist", {}) if isinstance(runtime, dict) else {}
            if bool(assist.get("enabled", False)):
                assist_calls += 1
                if bool(assist.get("ok", False)):
                    assist_success_calls += 1
        assist_success_rate = (
            float(assist_success_calls) / float(assist_calls) if assist_calls > 0 else 0.0
        )
    assist_enabled = args.assist_provider == "deepseek"
    assist_gate_ok = (
        (not assist_enabled)
        or (assist_calls >= max(0, args.assist_min_calls) and assist_success_rate >= max(0.0, args.assist_min_success_rate))
    )

    capability_caps = capability_obj.get("capabilities", {}) if isinstance(capability_obj, dict) else {}
    breadth_score = float(capability_caps.get("breadth", 0.0) or 0.0)
    horizon_score = float(capability_caps.get("horizon", 0.0) or 0.0)
    robustness_score = float(capability_caps.get("robustness", 0.0) or 0.0)
    interactive_metrics = interactive_obj.get("metrics", {}) if isinstance(interactive_obj, dict) else {}
    interactive_success_rate = float(interactive_metrics.get("success_rate", 0.0) or 0.0)
    interactive_task_count = int(interactive_obj.get("meta", {}).get("task_count", 0) or 0)
    interactive_passed = bool(interactive_metrics.get("passed", False))
    interactive_avg_steps = float(interactive_metrics.get("avg_steps", 0.0) or 0.0)
    breadth_ok = breadth_score >= max(0.0, args.min_breadth)
    horizon_ok = horizon_score >= max(0.0, args.min_horizon)
    robustness_ok = robustness_score >= max(0.0, args.min_robustness)

    docker_gate_ok = docker_ok or (
        args.allow_missing_docker and docker_reason in {"docker-not-found", "docker-run-failed"}
    )
    steps_ok = all(v.get("returncode", 1) == 0 for v in steps.values())
    gate_ok = bool(
        steps_ok
        and trust_ok
        and acceptance_ok
        and docker_gate_ok
        and monitor_ok
        and assist_gate_ok
        and breadth_ok
        and horizon_ok
        and robustness_ok
    )

    payload = {
        "meta": {
            "created_at_utc": _now_utc(),
            "lookback_rounds": max(0, args.lookback_rounds),
            "docker_image": args.docker_image,
            "allow_missing_docker": bool(args.allow_missing_docker),
            "assist_provider": args.assist_provider,
            "assist_model": args.assist_model,
            "assist_key_file": args.assist_key_file,
            "min_breadth": max(0.0, args.min_breadth),
            "min_horizon": max(0.0, args.min_horizon),
            "min_robustness": max(0.0, args.min_robustness),
        },
        "steps": steps,
        "evidence": {
            "trusted_center": str(trust_path) if trust_path else "",
            "round": str(round_path) if round_path else "",
            "monitor": str(monitor_path) if monitor_path else "",
            "framework": str(framework_path) if framework_path else "",
            "capability_registry": str(capability_path) if capability_path else "",
            "interactive_benchmark": str(interactive_path) if interactive_path else "",
        },
        "signals": {
            "trust_ok": trust_ok,
            "acceptance_ok": acceptance_ok,
            "docker_ok": docker_ok,
            "docker_reason": docker_reason,
            "monitor_ok": monitor_ok,
            "framework_score": framework_score,
            "assist_enabled": assist_enabled,
            "assist_calls": assist_calls,
            "assist_success_calls": assist_success_calls,
            "assist_success_rate": assist_success_rate,
            "assist_min_calls": max(0, args.assist_min_calls),
            "assist_min_success_rate": max(0.0, args.assist_min_success_rate),
            "assist_gate_ok": assist_gate_ok,
            "breadth": breadth_score,
            "horizon": horizon_score,
            "robustness": robustness_score,
            "breadth_ok": breadth_ok,
            "horizon_ok": horizon_ok,
            "robustness_ok": robustness_ok,
            "interactive_success_rate": interactive_success_rate,
            "interactive_task_count": interactive_task_count,
            "interactive_passed": interactive_passed,
            "interactive_avg_steps": interactive_avg_steps,
        },
        "gate_ok": gate_ok,
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    lines = [
        "# Unified Release Gate",
        "",
        f"- created_at_utc: `{payload['meta']['created_at_utc']}`",
        f"- gate_ok: `{gate_ok}`",
        f"- trust_ok: `{trust_ok}`",
        f"- acceptance_ok: `{acceptance_ok}`",
        f"- docker_ok: `{docker_ok}` (reason: `{docker_reason}`)",
        f"- monitor_ok: `{monitor_ok}`",
        f"- framework_score: `{framework_score:.3f}`",
        f"- assist_enabled: `{assist_enabled}`",
        f"- assist_calls: `{assist_calls}`",
        f"- assist_success_calls: `{assist_success_calls}`",
        f"- assist_success_rate: `{assist_success_rate:.2%}`",
        f"- assist_gate_ok: `{assist_gate_ok}`",
        f"- breadth: `{breadth_score:.3f}` (ok: `{breadth_ok}` / min: `{max(0.0, args.min_breadth):.3f}`)",
        f"- horizon: `{horizon_score:.3f}` (ok: `{horizon_ok}` / min: `{max(0.0, args.min_horizon):.3f}`)",
        f"- robustness: `{robustness_score:.3f}` (ok: `{robustness_ok}` / min: `{max(0.0, args.min_robustness):.3f}`)",
        f"- interactive_success_rate: `{interactive_success_rate:.2%}` (tasks: `{interactive_task_count}`, passed: `{interactive_passed}`)",
        f"- interactive_avg_steps: `{interactive_avg_steps:.2f}`",
        "",
        "## Evidence",
        "",
        f"- trusted_center: `{payload['evidence']['trusted_center']}`",
        f"- round: `{payload['evidence']['round']}`",
        f"- monitor: `{payload['evidence']['monitor']}`",
        f"- framework: `{payload['evidence']['framework']}`",
        f"- capability_registry: `{payload['evidence']['capability_registry']}`",
        f"- interactive_benchmark: `{payload['evidence']['interactive_benchmark']}`",
        "",
    ]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest_md.write_text("\n".join(lines), encoding="utf-8")

    print("Release gate completed")
    print(f"Gate OK: {gate_ok}")
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")

    if not gate_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
