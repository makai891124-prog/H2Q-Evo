#!/usr/bin/env python3
"""One-click integrated AGI validation pipeline.

This script runs a reproducible validation flow and writes:
- reports/agi_system_instantiation_public_validation_<ts>.md
- reports/agi_system_instantiation_public_validation_latest.md
- reports/agi_system_instantiation_stability_<ts>.md
- reports/agi_system_instantiation_stability_latest.md
- reports/agi_integrated_validation_<ts>.json
- reports/agi_integrated_validation_latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    name: str
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


def run_cmd(name: str, cmd: List[str], cwd: Path) -> StepResult:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return StepResult(name=name, cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def latest_by_prefix(reports_dir: Path, prefix: str, suffix: str) -> Path | None:
    candidates = sorted(reports_dir.glob(f"{prefix}*{suffix}"))
    return candidates[-1] if candidates else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(release_gate_json: Path, public_json: Path, monitor_json: Path, blueprint_json: Path | None) -> Dict[str, float]:
    gate = load_json(release_gate_json)
    public = load_json(public_json)
    monitor = load_json(monitor_json)

    signals = gate.get("signals", {})
    align = public.get("alignment", {})

    out = {
        "gate_ok": bool(gate.get("gate_ok", False)),
        "breadth": float(signals.get("breadth", 0.0)),
        "horizon": float(signals.get("horizon", 0.0)),
        "robustness": float(signals.get("robustness", 0.0)),
        "interactive_success_rate": float(signals.get("interactive_success_rate", 0.0)),
        "arc": float(align.get("arc_agi", {}).get("score", 0.0)),
        "swe": float(align.get("swe_bench", {}).get("score", 0.0)),
        "metr": float(align.get("metr_horizon", {}).get("score", 0.0)),
        "alignment_overall": float(align.get("overall", 0.0)),
    }

    monitor_sr = 0.0
    rounds = monitor.get("rounds", []) if isinstance(monitor, dict) else []
    if rounds:
        success_values = [float(r.get("overall_success_ratio", 0.0)) for r in rounds]
        monitor_sr = sum(success_values) / max(1, len(success_values))
    out["monitor_success_ratio"] = monitor_sr

    if blueprint_json and blueprint_json.exists():
        bp = load_json(blueprint_json)
        summary = bp.get("summary", {})
        strategy = summary.get("strategy", {})
        ms = summary.get("module_stats", {}).get("release_gate", {})
        out["blueprint_ok_rate"] = float(strategy.get("ok_rate", 0.0))
        runs = int(ms.get("runs", 0))
        succ = int(ms.get("success", 0))
        out["blueprint_gate_success_ratio"] = (succ / runs) if runs else 0.0
        cycles = bp.get("cycles", [])
        if cycles:
            out["blueprint_robustness_end"] = float(cycles[-1].get("context_after", {}).get("scores", {}).get("robustness", 0.0))
        else:
            out["blueprint_robustness_end"] = 0.0
    else:
        out["blueprint_ok_rate"] = 0.0
        out["blueprint_gate_success_ratio"] = 0.0
        out["blueprint_robustness_end"] = 0.0

    return out


def write_markdown_report(path: Path, latest_path: Path, title: str, lines: List[str]) -> None:
    text = "\n".join([f"# {title}", ""] + lines) + "\n"
    path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run integrated AGI validation and generate reports")
    parser.add_argument("--python", default=".venv/bin/python", help="Python executable path")
    parser.add_argument("--with-longrun", action="store_true", help="Run dynamic blueprint longrun (8 cycles) and re-validate")
    parser.add_argument("--longrun-cycles", type=int, default=8)
    parser.add_argument(
        "--release-gate-docker-policy",
        choices=["auto", "strict", "allow-missing"],
        default="auto",
    )
    parser.add_argument("--release-gate-allow-missing-docker", action="store_true")
    parser.add_argument("--output-prefix", default="agi_integrated_validation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    py = str((repo_root / args.python).resolve()) if not Path(args.python).is_absolute() else args.python

    ts = int(time.time())
    now = datetime.now(timezone.utc).isoformat()

    steps: List[StepResult] = []

    def must(step_name: str, cmd: List[str]) -> StepResult:
        res = run_cmd(step_name, cmd, cwd=repo_root)
        steps.append(res)
        if res.returncode != 0:
            raise RuntimeError(f"Step failed: {step_name}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        return res

    def with_release_gate_docker_flags(cmd: List[str]) -> List[str]:
        cmd.extend(["--docker-policy", args.release_gate_docker_policy])
        if args.release_gate_allow_missing_docker:
            cmd.append("--allow-missing-docker")
        return cmd

    # Baseline integrated instantiation and validation.
    trust_res = must("trust_center", [py, "tools/trusted_joint_agi_quantum_center.py", "--profile", "quick", "--skip-rsa"])
    trust_json = first_match(r"Data:\s*(.+\.json)", trust_res.stdout)

    must(
        "unified_framework",
        [
            py,
            "tools/unified_system_framework.py",
            "--output-prefix",
            "unified_system_framework_instantiation",
            "--refresh-trust-center",
            "--trust-profile",
            "quick",
            "--trust-skip-rsa",
        ],
    )

    if trust_json:
        must("showcase", [py, "tools/agi_showcase_generator.py", "--center-json", trust_json])
    else:
        must("showcase", [py, "tools/agi_showcase_generator.py"])

    public_res_1 = must(
        "public_alignment_baseline",
        [
            py,
            "tools/public_alignment_report.py",
            "--output-prefix",
            "public_alignment_instantiation",
            "--arc-target",
            "0.75",
            "--swe-target",
            "0.50",
            "--metr-target",
            "0.55",
        ],
    )
    public_json_1 = first_match(r"JSON:\s*(.+\.json)", public_res_1.stdout)

    gate_res_1 = must(
        "release_gate_baseline",
        with_release_gate_docker_flags(
            [
                py,
                "tools/release_gate.py",
                "--profile",
                "quick",
                "--lookback-rounds",
                "48",
                "--assist-provider",
                "none",
                "--min-breadth",
                "0.60",
                "--min-horizon",
                "0.80",
                "--min-robustness",
                "0.60",
                "--output-prefix",
                "release_gate_instantiation",
            ]
        ),
    )
    gate_json_1 = first_match(r"JSON:\s*(.+\.json)", gate_res_1.stdout)

    must(
        "realtime_monitor_baseline",
        [
            py,
            "tools/agi_realtime_monitor.py",
            "--interval-seconds",
            "5",
            "--cycles",
            "1",
            "--lookback-rounds",
            "48",
            "--output-prefix",
            "agi_realtime_monitor_instantiation",
        ],
    )

    monitor_json_1 = reports_dir / "agi_realtime_monitor_latest.json"

    blueprint_baseline = reports_dir / "dynamic_blueprint_bootstrap_autorun_latest.json"
    baseline_metrics = extract_metrics(
        Path(gate_json_1) if gate_json_1 else reports_dir / "release_gate_instantiation_latest.json",
        Path(public_json_1) if public_json_1 else reports_dir / "public_alignment_instantiation_latest.json",
        monitor_json_1,
        blueprint_baseline if blueprint_baseline.exists() else None,
    )

    # Optional longrun + re-validation.
    long_metrics: Dict[str, float] | None = None
    long_blueprint_json = ""
    gate_json_2 = ""
    public_json_2 = ""
    if args.with_longrun:
        longrun_res = must(
            "dynamic_blueprint_longrun",
            [
                py,
                "tools/dynamic_blueprint_bootstrap.py",
                "--cycles",
                str(args.longrun_cycles),
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
                "dynamic_blueprint_bootstrap_longrun",
            ],
        )
        long_blueprint_json = first_match(r"Latest JSON:\s*(.+\.json)", longrun_res.stdout)

        public_res_2 = must(
            "public_alignment_post_longrun",
            [
                py,
                "tools/public_alignment_report.py",
                "--output-prefix",
                "public_alignment_post_longrun",
                "--arc-target",
                "0.75",
                "--swe-target",
                "0.50",
                "--metr-target",
                "0.55",
            ],
        )
        public_json_2 = first_match(r"JSON:\s*(.+\.json)", public_res_2.stdout)

        gate_res_2 = must(
            "release_gate_post_longrun",
            with_release_gate_docker_flags(
                [
                    py,
                    "tools/release_gate.py",
                    "--profile",
                    "quick",
                    "--lookback-rounds",
                    "48",
                    "--assist-provider",
                    "none",
                    "--min-breadth",
                    "0.60",
                    "--min-horizon",
                    "0.80",
                    "--min-robustness",
                    "0.60",
                    "--output-prefix",
                    "release_gate_post_longrun",
                ]
            ),
        )
        gate_json_2 = first_match(r"JSON:\s*(.+\.json)", gate_res_2.stdout)

        must(
            "realtime_monitor_post_longrun",
            [
                py,
                "tools/agi_realtime_monitor.py",
                "--interval-seconds",
                "5",
                "--cycles",
                "1",
                "--lookback-rounds",
                "48",
                "--output-prefix",
                "agi_realtime_monitor_post_longrun",
            ],
        )

        long_metrics = extract_metrics(
            Path(gate_json_2) if gate_json_2 else reports_dir / "release_gate_post_longrun_latest.json",
            Path(public_json_2) if public_json_2 else reports_dir / "public_alignment_post_longrun_latest.json",
            reports_dir / "agi_realtime_monitor_latest.json",
            Path(long_blueprint_json) if long_blueprint_json else reports_dir / "dynamic_blueprint_bootstrap_longrun_latest.json",
        )

    # Write machine-readable summary.
    summary = {
        "generated_at_utc": now,
        "with_longrun": args.with_longrun,
        "release_gate_docker_policy": args.release_gate_docker_policy,
        "release_gate_allow_missing_docker": bool(args.release_gate_allow_missing_docker),
        "steps": [
            {
                "name": s.name,
                "cmd": s.cmd,
                "returncode": s.returncode,
                "stdout_tail": s.stdout[-1200:],
                "stderr_tail": s.stderr[-1200:],
            }
            for s in steps
        ],
        "baseline_metrics": baseline_metrics,
        "longrun_metrics": long_metrics,
        "artifacts": {
            "trust_center": trust_json,
            "public_alignment_baseline": public_json_1,
            "release_gate_baseline": gate_json_1,
            "dynamic_blueprint_longrun": long_blueprint_json,
            "public_alignment_post_longrun": public_json_2,
            "release_gate_post_longrun": gate_json_2,
        },
    }

    summary_json = reports_dir / f"{args.output_prefix}_{ts}.json"
    summary_json_latest = reports_dir / f"{args.output_prefix}_latest.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json_latest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Main public validation report.
    main_md = reports_dir / f"agi_system_instantiation_public_validation_{ts}.md"
    main_md_latest = reports_dir / "agi_system_instantiation_public_validation_latest.md"
    main_lines = [
        f"- Generated (UTC): `{now}`",
        "- Status: `PASS`",
        "",
        "## Key Conclusions",
        "",
        f"- Release gate pass: `{baseline_metrics['gate_ok']}`",
        f"- Core scores: breadth=`{baseline_metrics['breadth']:.6f}`, horizon=`{baseline_metrics['horizon']:.6f}`, robustness=`{baseline_metrics['robustness']:.6f}`",
        f"- Public alignment: ARC=`{baseline_metrics['arc']:.6f}`, SWE=`{baseline_metrics['swe']:.6f}`, METR=`{baseline_metrics['metr']:.6f}`, overall=`{baseline_metrics['alignment_overall']:.6f}`",
        f"- Interactive success rate: `{baseline_metrics['interactive_success_rate']:.6f}`",
        f"- Realtime monitor success ratio: `{baseline_metrics['monitor_success_ratio']:.6f}`",
        "",
        "## Evidence Links",
        "",
        f"- Trust center data: `{trust_json}`",
        f"- Unified framework latest: `{reports_dir / 'unified_system_framework_instantiation_latest.json'}`",
        f"- Showcase report (latest by prefix): `{latest_by_prefix(reports_dir, 'AGI真实功能展示最终报告_', '.md')}`",
        f"- Public alignment baseline: `{public_json_1}`",
        f"- Release gate baseline: `{gate_json_1}`",
        f"- Realtime monitor latest: `{reports_dir / 'agi_realtime_monitor_latest.json'}`",
        f"- Validation summary JSON: `{summary_json}`",
    ]
    write_markdown_report(main_md, main_md_latest, "AGI System Instantiation Public Validation", main_lines)

    # Long-stability comparison report.
    stability_md = reports_dir / f"agi_system_instantiation_stability_{ts}.md"
    stability_md_latest = reports_dir / "agi_system_instantiation_stability_latest.md"

    stability_lines = [
        f"- Generated (UTC): `{now}`",
        f"- Longrun enabled: `{args.with_longrun}`",
        "",
        "## Baseline Snapshot",
        "",
        f"- gate_ok=`{baseline_metrics['gate_ok']}`",
        f"- robustness=`{baseline_metrics['robustness']:.6f}`",
        f"- alignment_overall=`{baseline_metrics['alignment_overall']:.6f}`",
        f"- blueprint_ok_rate=`{baseline_metrics['blueprint_ok_rate']:.6f}`",
        f"- blueprint_gate_success_ratio=`{baseline_metrics['blueprint_gate_success_ratio']:.6f}`",
    ]

    if long_metrics is not None:
        stability_lines += [
            "",
            "## Longrun Snapshot",
            "",
            f"- gate_ok=`{long_metrics['gate_ok']}`",
            f"- robustness=`{long_metrics['robustness']:.6f}`",
            f"- alignment_overall=`{long_metrics['alignment_overall']:.6f}`",
            f"- blueprint_ok_rate=`{long_metrics['blueprint_ok_rate']:.6f}`",
            f"- blueprint_gate_success_ratio=`{long_metrics['blueprint_gate_success_ratio']:.6f}`",
            "",
            "## Delta (Longrun - Baseline)",
            "",
            f"- robustness_delta=`{long_metrics['robustness'] - baseline_metrics['robustness']:.6f}`",
            f"- alignment_overall_delta=`{long_metrics['alignment_overall'] - baseline_metrics['alignment_overall']:.6f}`",
            f"- blueprint_ok_rate_delta=`{long_metrics['blueprint_ok_rate'] - baseline_metrics['blueprint_ok_rate']:.6f}`",
            f"- blueprint_gate_success_ratio_delta=`{long_metrics['blueprint_gate_success_ratio'] - baseline_metrics['blueprint_gate_success_ratio']:.6f}`",
            "",
            "## Longrun Evidence",
            "",
            f"- Dynamic blueprint longrun latest: `{long_blueprint_json or (reports_dir / 'dynamic_blueprint_bootstrap_longrun_latest.json')}`",
            f"- Public alignment post longrun: `{public_json_2}`",
            f"- Release gate post longrun: `{gate_json_2}`",
        ]

    stability_lines += [
        "",
        "## Validation Summary JSON",
        "",
        f"- `{summary_json}`",
    ]

    write_markdown_report(stability_md, stability_md_latest, "AGI System Instantiation Stability Comparison", stability_lines)

    print(f"Summary JSON: {summary_json}")
    print(f"Latest Summary JSON: {summary_json_latest}")
    print(f"Public Validation MD: {main_md}")
    print(f"Latest Public Validation MD: {main_md_latest}")
    print(f"Stability MD: {stability_md}")
    print(f"Latest Stability MD: {stability_md_latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
