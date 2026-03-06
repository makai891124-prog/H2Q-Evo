#!/usr/bin/env python3
"""Generate key trend chart from dynamic blueprint bootstrap reports.

Produces a 3-panel chart in reports/:
- robustness trend by cycle
- strategy ok_rate comparison
- release_gate success ratio comparison
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract(report: dict) -> dict:
    cycles = report.get("cycles", [])
    robustness = [c["context_after"]["scores"].get("robustness", 0.0) for c in cycles]

    summary = report.get("summary", {})
    strategy = summary.get("strategy", {})
    module_stats = summary.get("module_stats", {})
    gate = module_stats.get("release_gate", {})

    gate_runs = gate.get("runs", 0)
    gate_success = gate.get("success", 0)
    gate_success_ratio = (gate_success / gate_runs) if gate_runs else 0.0

    return {
        "robustness": robustness,
        "robustness_start": float(robustness[0]) if robustness else 0.0,
        "robustness_end": float(robustness[-1]) if robustness else 0.0,
        "ok_rate": float(strategy.get("ok_rate", 0.0)),
        "gate_success_ratio": float(gate_success_ratio),
        "gate_runs": int(gate_runs),
        "gate_success": int(gate_success),
        "cycle_count": int(summary.get("cycle_count", len(cycles))),
    }


def write_markdown_summary(
    short_data: dict,
    long_data: dict,
    out_png: Path,
    out_latest_png: Path,
    short_json: Path,
    long_json: Path,
    out_md: Path,
    out_latest_md: Path,
) -> None:
    delta_robustness_end = long_data["robustness_end"] - short_data["robustness_end"]
    delta_ok_rate = long_data["ok_rate"] - short_data["ok_rate"]
    delta_gate_success_ratio = long_data["gate_success_ratio"] - short_data["gate_success_ratio"]
    delta_gate_runs = long_data["gate_runs"] - short_data["gate_runs"]
    delta_gate_success = long_data["gate_success"] - short_data["gate_success"]

    lines = [
        "# Dynamic Blueprint Key Metrics Trend Summary",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Short JSON: `{short_json}`",
        f"- Long JSON: `{long_json}`",
        f"- Chart (timestamped): `{out_png}`",
        f"- Chart (latest): `{out_latest_png}`",
        "",
        "## Key Delta (Long - Short)",
        "",
        f"- cycle_count delta: `{long_data['cycle_count']} - {short_data['cycle_count']} = {long_data['cycle_count'] - short_data['cycle_count']}`",
        f"- robustness_end delta: `{long_data['robustness_end']:.6f} - {short_data['robustness_end']:.6f} = {delta_robustness_end:.6f}`",
        f"- ok_rate delta: `{long_data['ok_rate']:.6f} - {short_data['ok_rate']:.6f} = {delta_ok_rate:.6f}`",
        f"- gate_success_ratio delta: `{long_data['gate_success_ratio']:.6f} - {short_data['gate_success_ratio']:.6f} = {delta_gate_success_ratio:.6f}`",
        f"- gate_runs delta: `{long_data['gate_runs']} - {short_data['gate_runs']} = {delta_gate_runs}`",
        f"- gate_success delta: `{long_data['gate_success']} - {short_data['gate_success']} = {delta_gate_success}`",
        "",
        "## Snapshot",
        "",
        f"- short: cycles=`{short_data['cycle_count']}`, robustness_end=`{short_data['robustness_end']:.6f}`, ok_rate=`{short_data['ok_rate']:.6f}`, gate_success_ratio=`{short_data['gate_success_ratio']:.6f}`",
        f"- long: cycles=`{long_data['cycle_count']}`, robustness_end=`{long_data['robustness_end']:.6f}`, ok_rate=`{long_data['ok_rate']:.6f}`, gate_success_ratio=`{long_data['gate_success_ratio']:.6f}`",
    ]

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    out_latest_md.write_text(text, encoding="utf-8")


def plot(short_data: dict, long_data: dict, out_png: Path, out_latest_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    x_short = list(range(1, len(short_data["robustness"]) + 1))
    x_long = list(range(1, len(long_data["robustness"]) + 1))

    ax0 = axes[0]
    if x_short:
        ax0.plot(x_short, short_data["robustness"], marker="o", linewidth=2.0, label="short(2-cycle)")
    if x_long:
        ax0.plot(x_long, long_data["robustness"], marker="s", linewidth=2.0, label="long(8-cycle)")
    ax0.set_title("Robustness Trend")
    ax0.set_xlabel("Cycle")
    ax0.set_ylabel("Robustness")
    ax0.set_ylim(0.0, 1.05)
    ax0.grid(alpha=0.25)
    ax0.legend()

    ax1 = axes[1]
    ax1.bar(["short", "long"], [short_data["ok_rate"], long_data["ok_rate"]], color=["#4c78a8", "#72b7b2"])
    ax1.set_title("Strategy OK Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = axes[2]
    ax2.bar(
        ["short", "long"],
        [short_data["gate_success_ratio"], long_data["gate_success_ratio"]],
        color=["#f58518", "#54a24b"],
    )
    ax2.set_title("Release Gate Success Ratio")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Dynamic Blueprint Key Metrics Comparison", fontsize=13)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_latest_png, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dynamic blueprint trend chart from two JSON reports")
    parser.add_argument(
        "--short-json",
        default="reports/dynamic_blueprint_bootstrap_autorun_latest.json",
        help="Short-run JSON path",
    )
    parser.add_argument(
        "--long-json",
        default="reports/dynamic_blueprint_bootstrap_longrun_latest.json",
        help="Long-run JSON path",
    )
    parser.add_argument(
        "--output-prefix",
        default="dynamic_blueprint_key_metrics_trend",
        help="Output file prefix under reports/",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    short_json = (repo_root / args.short_json).resolve()
    long_json = (repo_root / args.long_json).resolve()

    short_report = load_json(short_json)
    long_report = load_json(long_json)

    short_data = extract(short_report)
    long_data = extract(long_report)

    reports_dir = repo_root / "reports"
    ts = int(time.time())
    out_png = reports_dir / f"{args.output_prefix}_{ts}.png"
    out_latest_png = reports_dir / f"{args.output_prefix}_latest.png"
    out_md = reports_dir / f"{args.output_prefix}_{ts}.md"
    out_latest_md = reports_dir / f"{args.output_prefix}_latest.md"

    plot(short_data, long_data, out_png, out_latest_png)
    write_markdown_summary(
        short_data,
        long_data,
        out_png,
        out_latest_png,
        short_json,
        long_json,
        out_md,
        out_latest_md,
    )

    print(f"Trend PNG: {out_png}")
    print(f"Latest Trend PNG: {out_latest_png}")
    print(f"Trend MD: {out_md}")
    print(f"Latest Trend MD: {out_latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
