#!/usr/bin/env python3
"""Run automatic blueprint + cross-public validation as a single pipeline.

Outputs:
- reports/auto_blueprint_cross_public_latest.json
- reports/agi_cross_public_validation_latest.json
- reports/auto_blueprint_cross_public_validation_latest.md
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
REPORTS = ROOT / "reports"


def run_cmd(cmd: list[str], log_path: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSee log: {log_path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    bp_log = REPORTS / "auto_blueprint_cross_public_pipeline_blueprint.log"
    cv_log = REPORTS / "auto_blueprint_cross_public_pipeline_validation.log"

    run_cmd(
        [
            str(PY),
            "tools/dynamic_blueprint_bootstrap.py",
            "--cycles",
            "3",
            "--max-actions-per-cycle",
            "2",
            "--enable-release-gate-cycle",
            "--strong-release-gate-cycle",
            "--release-gate-retries",
            "2",
            "--release-gate-profile",
            "quick",
            "--release-gate-relax-step",
            "0.05",
            "--min-breadth",
            "0.60",
            "--min-horizon",
            "0.80",
            "--min-robustness",
            "0.60",
            "--output-prefix",
            "auto_blueprint_cross_public",
        ],
        bp_log,
    )

    run_cmd(
        [
            str(PY),
            "tools/run_agi_integrated_validation.py",
            "--with-longrun",
            "--longrun-cycles",
            "3",
            "--output-prefix",
            "agi_cross_public_validation",
        ],
        cv_log,
    )

    bp = load_json(REPORTS / "auto_blueprint_cross_public_latest.json")
    cv = load_json(REPORTS / "agi_cross_public_validation_latest.json")

    baseline = cv.get("baseline_metrics", {})
    longrun = cv.get("longrun_metrics", {})
    strategy = (bp.get("summary") or {}).get("strategy", {})
    gate_stats = ((bp.get("summary") or {}).get("module_stats") or {}).get("release_gate", {})

    runs = int(gate_stats.get("runs", 0))
    succ = int(gate_stats.get("success", 0))
    ratio = (succ / runs) if runs else 0.0

    now = datetime.now(timezone.utc).isoformat()
    out = REPORTS / "auto_blueprint_cross_public_validation_latest.md"
    evidence_dir = REPORTS / "auto_blueprint_cross_public_evidence_latest"

    lines = [
        "# Auto Blueprint Cross-Public Validation",
        "",
        f"- generated_at_utc: `{now}`",
        "- objective: `按指标自动蓝图化 -> 综合实施 -> 公开交叉验证`",
        f"- final_status: `{'PASS' if longrun.get('gate_ok') else 'CHECK'}`",
        "",
        "## 1) Auto Blueprint",
        "",
        f"- cycle_count: `{(bp.get('summary') or {}).get('cycle_count', 0)}`",
        f"- overall_ok: `{(bp.get('summary') or {}).get('overall_ok', False)}`",
        f"- strategy_ok_rate: `{float(strategy.get('ok_rate', 0.0)):.6f}`",
        f"- release_gate_success_ratio: `{succ}/{runs} = {ratio:.6f}`",
        "",
        "## 2) Cross-Public Metrics",
        "",
        f"- gate_ok: `{baseline.get('gate_ok')} -> {longrun.get('gate_ok')}`",
        f"- robustness: `{float(baseline.get('robustness', 0.0)):.6f} -> {float(longrun.get('robustness', 0.0)):.6f}`",
        f"- alignment_overall: `{float(baseline.get('alignment_overall', 0.0)):.6f} -> {float(longrun.get('alignment_overall', 0.0)):.6f}`",
        f"- blueprint_ok_rate: `{float(baseline.get('blueprint_ok_rate', 0.0)):.6f} -> {float(longrun.get('blueprint_ok_rate', 0.0)):.6f}`",
        f"- blueprint_gate_success_ratio: `{float(baseline.get('blueprint_gate_success_ratio', 0.0)):.6f} -> {float(longrun.get('blueprint_gate_success_ratio', 0.0)):.6f}`",
        "",
        "## 3) Evidence",
        "",
        "- blueprint_latest: `reports/auto_blueprint_cross_public_latest.json`",
        "- cross_validation_latest: `reports/agi_cross_public_validation_latest.json`",
        "- release_gate_post_longrun: `reports/release_gate_post_longrun_latest.json`",
        "- public_alignment_post_longrun: `reports/public_alignment_post_longrun_latest.json`",
        "",
        "## 4) Logs",
        "",
        "- `reports/auto_blueprint_cross_public_pipeline_blueprint.log`",
        "- `reports/auto_blueprint_cross_public_pipeline_validation.log`",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    candidates = [
        REPORTS / "auto_blueprint_cross_public_latest.json",
        REPORTS / "agi_cross_public_validation_latest.json",
        REPORTS / "release_gate_post_longrun_latest.json",
        REPORTS / "public_alignment_post_longrun_latest.json",
        REPORTS / "dynamic_blueprint_bootstrap_longrun_latest.json",
        REPORTS / "auto_blueprint_cross_public_pipeline_blueprint.log",
        REPORTS / "auto_blueprint_cross_public_pipeline_validation.log",
        out,
    ]
    for src in candidates:
        if _copy_if_exists(src, evidence_dir / src.name):
            copied.append(src.name)

    manifest = {
        "generated_at_utc": now,
        "objective": "auto_blueprint_cross_public_reproducibility",
        "source_reports_dir": str(REPORTS),
        "evidence_dir": str(evidence_dir),
        "files": copied,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    evidence_latest = REPORTS / "auto_blueprint_cross_public_evidence_latest.json"
    evidence_latest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report: {out}")
    print(f"Evidence dir: {evidence_dir}")
    print(f"Evidence manifest: {evidence_latest}")
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
