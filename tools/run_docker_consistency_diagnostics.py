#!/usr/bin/env python3
"""Generate a concrete Docker consistency diagnostics report from release gate artifacts."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _recommendations(payload: Dict[str, Any]) -> List[str]:
    signals = payload.get("signals", {}) if isinstance(payload, dict) else {}
    reason = str(signals.get("docker_reason", ""))
    ok = bool(signals.get("docker_ok", False))
    policy = str(signals.get("docker_policy", ""))
    check_enabled = bool(signals.get("docker_check_enabled", False))

    recs: List[str] = []
    if ok:
        recs.append("Keep docker consistency check enabled for strict/CI paths.")
        recs.append("Track overlap stability over time to detect drift between local and container outputs.")
        return recs

    if reason in {"docker-not-found"}:
        recs.append("Install Docker CLI/desktop on the execution host.")
    elif reason in {"docker-daemon-unreachable", "docker-daemon-timeout", "docker-daemon-probe-error"}:
        recs.append("Start Docker daemon first, then rerun strict validation.")
        recs.append("Use auto policy for local quick runs to avoid blocking non-docker evidence.")
    elif reason in {"docker-run-failed", "docker-timeout"}:
        recs.append("Check image availability and runtime permissions for docker run.")
        recs.append("Validate mounted workdir and container entrypoint path.")
    elif reason in {"docker-check-skipped-by-policy", "docker-check-missing-result"}:
        recs.append("Run one strict validation cycle with docker daemon available before release.")

    if policy == "strict" and not check_enabled:
        recs.append("Strict policy without daemon availability will fail gate by design.")

    if not recs:
        recs.append("Investigate release gate round artifact for full docker_consistency payload.")
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description="Docker consistency diagnostics report")
    parser.add_argument("--release-gate-json", default="reports/release_gate_master_latest.json")
    parser.add_argument("--round-json", default="")
    parser.add_argument("--output-prefix", default="docker_consistency_diagnostics")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    rg_path = Path(args.release_gate_json)
    if not rg_path.is_absolute():
        rg_path = ROOT / rg_path

    rg_obj = _load_json(rg_path)
    evidence = rg_obj.get("evidence", {}) if isinstance(rg_obj, dict) else {}
    inferred_round = str(evidence.get("round", "")).strip()

    round_path = Path(args.round_json) if args.round_json else Path(inferred_round)
    if str(round_path):
        if not round_path.is_absolute():
            round_path = ROOT / round_path
        round_obj = _load_json(round_path)
    else:
        round_obj = {}

    meta = rg_obj.get("meta", {}) if isinstance(rg_obj, dict) else {}
    signals = rg_obj.get("signals", {}) if isinstance(rg_obj, dict) else {}
    round_docker = ((round_obj.get("round", {}) if isinstance(round_obj, dict) else {}).get("docker_consistency", {}))

    payload = {
        "generated_at_utc": _now_utc(),
        "sources": {
            "release_gate_json": str(rg_path),
            "round_json": str(round_path) if str(round_path) else "",
        },
        "diagnostics": {
            "gate_ok": bool(rg_obj.get("gate_ok", False)),
            "docker_policy": str(meta.get("docker_policy", "")),
            "docker_check_enabled": bool(meta.get("docker_check_enabled", False)),
            "allow_missing_docker": bool(meta.get("allow_missing_docker", False)),
            "docker_probe": meta.get("docker_probe", {}),
            "docker_ok": bool(signals.get("docker_ok", False)),
            "docker_reason": str(signals.get("docker_reason", "")),
            "docker_gate_ok": bool(signals.get("docker_gate_ok", False)),
            "docker_daemon_available": bool(signals.get("docker_daemon_available", False)),
            "docker_overlap": round_docker.get("overlap"),
            "docker_min_overlap": round_docker.get("min_overlap"),
            "docker_local_route": round_docker.get("local_route", ""),
            "docker_workdir": round_docker.get("docker_workdir", ""),
            "docker_script": round_docker.get("docker_script", ""),
            "docker_stderr_tail": round_docker.get("stderr", ""),
        },
    }
    payload["recommendations"] = _recommendations(payload)

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    d = payload.get("diagnostics", {})
    lines = [
        "# Docker Consistency Diagnostics",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate_ok: `{d.get('gate_ok', False)}`",
        f"- docker_policy: `{d.get('docker_policy', '')}`",
        f"- docker_check_enabled: `{d.get('docker_check_enabled', False)}`",
        f"- allow_missing_docker: `{d.get('allow_missing_docker', False)}`",
        f"- docker_daemon_available: `{d.get('docker_daemon_available', False)}`",
        f"- docker_ok: `{d.get('docker_ok', False)}`",
        f"- docker_gate_ok: `{d.get('docker_gate_ok', False)}`",
        f"- docker_reason: `{d.get('docker_reason', '')}`",
        f"- overlap/min_overlap: `{d.get('docker_overlap')}` / `{d.get('docker_min_overlap')}`",
        "",
        "## Sources",
        "",
        f"- release_gate_json: `{payload['sources']['release_gate_json']}`",
        f"- round_json: `{payload['sources']['round_json']}`",
        "",
        "## Recommendations",
        "",
    ]
    for rec in payload.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
