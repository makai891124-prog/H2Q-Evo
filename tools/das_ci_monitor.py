import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_reports(workspace: Path):
    reports = []
    for path in sorted(workspace.glob("das_validation_report_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = int(data.get("meta", {}).get("timestamp", 0))
            if ts <= 0:
                continue
            reports.append((ts, path, data))
        except Exception:
            continue
    reports.sort(key=lambda x: x[0])
    return reports


def _extract_metrics(entry):
    ts, path, data = entry
    conf = data.get("confidence", {}).get("isomorphic_confidence_score", np.nan)
    verdict = data.get("verdict", {})
    mc = data.get("qgem_scan_and_mc", {}).get("monte_carlo", {})
    return {
        "timestamp": ts,
        "file": str(path),
        "confidence": float(conf),
        "decision_grade_ready": bool(verdict.get("decision_grade_ready", False)),
        "physics_ready": bool(verdict.get("physics_ready", False)),
        "isomorphism_ready": bool(verdict.get("isomorphism_ready", False)),
        "cohen_d": float(mc.get("cohen_d", np.nan)),
        "negative_probability_margin": float(mc.get("negative_probability_margin", np.nan)),
        "das_neg_prob": float(mc.get("das_negative_probability", np.nan)),
        "base_neg_prob": float(mc.get("baseline_negative_probability", np.nan)),
    }


def _plot_trend(metrics, output_path: Path):
    x = np.arange(len(metrics))
    confidence = [m["confidence"] for m in metrics]
    cohen_d = [m["cohen_d"] for m in metrics]
    margin = [m["negative_probability_margin"] for m in metrics]
    labels = [str(m["timestamp"])[-6:] for m in metrics]

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(x, confidence, marker="o", linewidth=2)
    ax[0].set_ylabel("Confidence")
    ax[0].set_title("DAS Validation Drift Trend")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(x, cohen_d, marker="o", color="#D81B60", linewidth=2)
    ax[1].axhline(-1.2, linestyle="--", color="#777777", linewidth=1)
    ax[1].axhline(1.2, linestyle="--", color="#777777", linewidth=1)
    ax[1].set_ylabel("Cohen d")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(x, margin, marker="o", color="#1E88E5", linewidth=2)
    ax[2].axhline(0.30, linestyle="--", color="#777777", linewidth=1)
    ax[2].set_ylabel("Neg. Prob Margin")
    ax[2].set_xlabel("Report Sequence")
    ax[2].grid(True, alpha=0.3)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_relationship(report, output_path: Path):
    mc = report.get("qgem_scan_and_mc", {}).get("monte_carlo", {})
    noise = report.get("qgem_noise_robustness", {}).get("sweep", [])
    cross = report.get("qgem_cross_validation", {})
    ext = cross.get("external_predictions", [])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sigmas = [r["noise_sigma"] for r in noise]
    probs = [r["negative_probability"] for r in noise]
    ax[0].plot(sigmas, probs, marker="o", linewidth=2)
    ax[0].set_xscale("log")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("Noise Sigma")
    ax[0].set_ylabel("P(W<0)")
    ax[0].set_title("Noise Robustness Curve")
    ax[0].grid(True, alpha=0.3)

    obs = [r["observed_delta_x_m"] for r in ext]
    pred = [r["predicted_delta_x_m"] for r in ext]
    if obs and pred:
        ax[1].scatter(obs, pred, color="#FB8C00", s=70)
        min_v = min(min(obs), min(pred))
        max_v = max(max(obs), max(pred))
        ax[1].plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#666666")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Observed delta_x (m)")
    ax[1].set_ylabel("Predicted delta_x (m)")
    ax[1].set_title("Cross-Validation: Model vs External Data")
    ax[1].grid(True, which="both", alpha=0.3)

    # Text block with latest headline metrics.
    text = (
        f"Confidence: {report.get('confidence', {}).get('isomorphic_confidence_score', float('nan')):.4f}\n"
        f"Cohen d: {mc.get('cohen_d', float('nan')):.4f}\n"
        f"Neg margin: {mc.get('negative_probability_margin', float('nan')):.4f}\n"
        f"Physics ready: {report.get('verdict', {}).get('physics_ready', False)}\n"
        f"Decision ready: {report.get('verdict', {}).get('decision_grade_ready', False)}"
    )
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_summary(metrics, drift, output_path: Path):
    latest = metrics[-1]
    lines = [
        "# DAS CI Drift Summary",
        "",
        f"- Latest report: `{latest['file']}`",
        f"- Latest confidence: `{latest['confidence']:.4f}`",
        f"- Latest cohen_d: `{latest['cohen_d']:.4f}`",
        f"- Latest negative probability margin: `{latest['negative_probability_margin']:.4f}`",
        f"- Decision ready: `{latest['decision_grade_ready']}`",
        f"- Physics ready: `{latest['physics_ready']}`",
        f"- Isomorphism ready: `{latest['isomorphism_ready']}`",
        "",
        "## Drift vs Previous",
        "",
        f"- Delta confidence: `{drift['delta_confidence']:.4f}`",
        f"- Delta cohen_d: `{drift['delta_cohen_d']:.4f}`",
        f"- Delta negative probability margin: `{drift['delta_margin']:.4f}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="DAS CI monitor: drift check + trend plots")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--output-dir", default="ci_artifacts", help="Artifact output directory")
    parser.add_argument("--max-confidence-drop", type=float, default=0.03)
    parser.add_argument("--max-margin-drop", type=float, default=0.05)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    output_dir = workspace / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = _load_reports(workspace)
    if len(reports) < 2:
        if len(reports) == 0:
            raise SystemExit("No das_validation_report_*.json files found")
        only = _extract_metrics(reports[0])
        trend_png = output_dir / "das_confidence_trend.png"
        rel_png = output_dir / "das_experiment_relationship.png"
        summary_md = output_dir / "das_ci_summary.md"
        drift_json = output_dir / "das_drift_metrics.json"

        _plot_trend([only], trend_png)
        _plot_relationship(reports[0][2], rel_png)
        summary_md.write_text(
            "# DAS CI Drift Summary\n\n"
            f"- Only one report available: `{only['file']}`\n"
            "- Drift comparison will start from the next run.\n",
            encoding="utf-8",
        )
        drift_json.write_text(json.dumps({"delta_confidence": 0.0, "delta_cohen_d": 0.0, "delta_margin": 0.0}, indent=2), encoding="utf-8")
        return

    metrics = [_extract_metrics(r) for r in reports]
    latest = metrics[-1]
    prev = metrics[-2]

    drift = {
        "delta_confidence": latest["confidence"] - prev["confidence"],
        "delta_cohen_d": latest["cohen_d"] - prev["cohen_d"],
        "delta_margin": latest["negative_probability_margin"] - prev["negative_probability_margin"],
    }

    trend_png = output_dir / "das_confidence_trend.png"
    rel_png = output_dir / "das_experiment_relationship.png"
    summary_md = output_dir / "das_ci_summary.md"
    drift_json = output_dir / "das_drift_metrics.json"

    _plot_trend(metrics, trend_png)
    _plot_relationship(reports[-1][2], rel_png)
    _write_summary(metrics, drift, summary_md)
    drift_json.write_text(json.dumps(drift, indent=2), encoding="utf-8")

    if args.fail_on_regression:
        confidence_drop = drift["delta_confidence"] < -abs(args.max_confidence_drop)
        margin_drop = drift["delta_margin"] < -abs(args.max_margin_drop)
        latest_ready = latest["decision_grade_ready"]
        if confidence_drop or margin_drop or (not latest_ready):
            raise SystemExit(
                "Regression detected: "
                f"confidence_drop={confidence_drop}, margin_drop={margin_drop}, decision_ready={latest_ready}"
            )


if __name__ == "__main__":
    main()
