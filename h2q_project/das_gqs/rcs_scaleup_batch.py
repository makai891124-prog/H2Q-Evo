from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from h2q_project.das_gqs.rcs_subset_stat_benchmark import RCSStatsReport, run_rcs_stats


@dataclass
class IterationGapRow:
    iteration: int
    n_qubits: int
    qubit_gap_to_public: int
    state_space_ratio_local_over_public: float
    mean_abs_error: float
    rmse: float
    max_abs_error: float
    tost_pass: bool
    baseline_time_mean_sec: float
    das_time_mean_sec: float


@dataclass
class ScaleupReport:
    public_reference_qubits: int
    baseline_iteration_n: int
    scaleup_n_list: list[int]
    seeds: list[int]
    depth_factor: int
    equivalence_margin: float
    alpha: float
    rows: list[IterationGapRow]
    rcs_scaleup_stats: RCSStatsReport


def _read_existing_n14(repo_root: Path) -> dict | None:
    path = repo_root / "reports" / "das_gqs_rcs_subset_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_n_stat(report_json: dict, n: int) -> dict | None:
    for r in report_json.get("per_n_stats", []):
        if int(r.get("n_qubits", -1)) == n:
            return r
    return None


def _row_from_stat(iteration: int, n: int, stat: dict, public_qubits: int) -> IterationGapRow:
    return IterationGapRow(
        iteration=iteration,
        n_qubits=n,
        qubit_gap_to_public=int(public_qubits - n),
        state_space_ratio_local_over_public=float((2**n) / (2**public_qubits)),
        mean_abs_error=float(stat["mean_abs_error"]),
        rmse=float(stat["rmse"]),
        max_abs_error=float(stat["max_abs_error"]),
        tost_pass=bool(stat["equivalence_pass"]),
        baseline_time_mean_sec=float(stat["baseline_time_mean_sec"]),
        das_time_mean_sec=float(stat["das_time_mean_sec"]),
    )


def _plot_gap(path: Path, rows: list[IterationGapRow]) -> None:
    rows_s = sorted(rows, key=lambda x: x.iteration)
    xs = [r.iteration for r in rows_s]
    ns = [r.n_qubits for r in rows_s]
    qgap = [r.qubit_gap_to_public for r in rows_s]
    ratio = [r.state_space_ratio_local_over_public for r in rows_s]
    mae = [r.mean_abs_error for r in rows_s]

    plt.figure(figsize=(9.2, 6.2), dpi=140)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(xs, ns, marker="o", linewidth=2.0, label="local n")
    ax1.plot(xs, qgap, marker="s", linewidth=1.8, label="qubit gap to public")
    ax1.set_xlabel("optimization iteration")
    ax1.set_ylabel("qubit scale")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(xs, ratio, marker="o", linewidth=2.0, label="state-space ratio local/public")
    ax2.plot(xs, mae, marker="^", linewidth=1.8, label="MAE")
    ax2.set_yscale("log")
    ax2.set_xlabel("optimization iteration")
    ax2.set_ylabel("log scale")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="best")

    for x, n in zip(xs, ns):
        ax1.annotate(f"n={n}", (x, n), textcoords="offset points", xytext=(4, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_csv(path: Path, rows: list[IterationGapRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "n_qubits",
                "qubit_gap_to_public",
                "state_space_ratio_local_over_public",
                "mean_abs_error",
                "rmse",
                "max_abs_error",
                "tost_pass",
                "baseline_time_mean_sec",
                "das_time_mean_sec",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.iteration,
                    r.n_qubits,
                    r.qubit_gap_to_public,
                    r.state_space_ratio_local_over_public,
                    r.mean_abs_error,
                    r.rmse,
                    r.max_abs_error,
                    r.tost_pass,
                    r.baseline_time_mean_sec,
                    r.das_time_mean_sec,
                ]
            )


def _render_md(report: ScaleupReport) -> str:
    lines: list[str] = []
    lines.append("# DAS-GQS RCS Scaleup Batch (n=16/18)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- public reference qubits: {report.public_reference_qubits}")
    lines.append(f"- baseline iteration n: {report.baseline_iteration_n}")
    lines.append(f"- scaleup n list: {report.scaleup_n_list}")
    lines.append(f"- seeds: {report.seeds}")
    lines.append(f"- depth factor: {report.depth_factor}")
    lines.append(f"- equivalence margin: {report.equivalence_margin}")
    lines.append(f"- alpha: {report.alpha}")
    lines.append("")

    lines.append("## Iteration Gap Table")
    lines.append("| iter | n | qubit_gap | state_ratio(local/public) | MAE | RMSE | max | TOST pass | baseline_t(s) | das_t(s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in sorted(report.rows, key=lambda x: x.iteration):
        lines.append(
            f"| {r.iteration} | {r.n_qubits} | {r.qubit_gap_to_public} | {r.state_space_ratio_local_over_public:.3e} | "
            f"{r.mean_abs_error:.3e} | {r.rmse:.3e} | {r.max_abs_error:.3e} | {r.tost_pass} | "
            f"{r.baseline_time_mean_sec:.6f} | {r.das_time_mean_sec:.6f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Iteration baseline uses existing n=14 report if available.")
    lines.append("- New scaleup iterations run n=16 and n=18 using DAS core and same statistical protocol.")
    lines.append("- Gap-to-public tracks distance to 53-qubit RCS challenge reference.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run n=16/18 RCS subset batch and generate scale-gap convergence figure")
    parser.add_argument("--n-list", type=str, default="16,18")
    parser.add_argument("--seed-list", type=str, default="11,17,23,31")
    parser.add_argument("--depth-factor", type=int, default=3)
    parser.add_argument("--equiv-margin", type=float, default=1e-9)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--public-qubits", type=int, default=53)
    parser.add_argument("--baseline-iteration-n", type=int, default=14)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]

    scaleup_stats = run_rcs_stats(
        n_list=n_list,
        seeds=seeds,
        depth_factor=args.depth_factor,
        equivalence_margin=args.equiv_margin,
        alpha=args.alpha,
    )

    rows: list[IterationGapRow] = []
    existing = _read_existing_n14(repo_root)
    if existing is not None:
        base_stat = _extract_n_stat(existing, args.baseline_iteration_n)
        if base_stat is not None:
            rows.append(_row_from_stat(0, args.baseline_iteration_n, base_stat, args.public_qubits))

    for i, r in enumerate(scaleup_stats.per_n_stats, start=1):
        rows.append(
            IterationGapRow(
                iteration=i,
                n_qubits=r.n_qubits,
                qubit_gap_to_public=int(args.public_qubits - r.n_qubits),
                state_space_ratio_local_over_public=float((2**r.n_qubits) / (2**args.public_qubits)),
                mean_abs_error=r.mean_abs_error,
                rmse=r.rmse,
                max_abs_error=r.max_abs_error,
                tost_pass=r.equivalence_pass,
                baseline_time_mean_sec=r.baseline_time_mean_sec,
                das_time_mean_sec=r.das_time_mean_sec,
            )
        )

    report = ScaleupReport(
        public_reference_qubits=args.public_qubits,
        baseline_iteration_n=args.baseline_iteration_n,
        scaleup_n_list=n_list,
        seeds=seeds,
        depth_factor=args.depth_factor,
        equivalence_margin=args.equiv_margin,
        alpha=args.alpha,
        rows=rows,
        rcs_scaleup_stats=scaleup_stats,
    )

    json_path = out_dir / "das_gqs_rcs_scaleup_16_18.json"
    md_path = out_dir / "das_gqs_rcs_scaleup_16_18.md"
    csv_path = out_dir / "das_gqs_rcs_scaleup_16_18.csv"
    fig_path = out_dir / "das_gqs_scale_gap_convergence.png"

    json_path.write_text(json.dumps(asdict(report), ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    _write_csv(csv_path, rows)
    _plot_gap(fig_path, rows)

    print("=== RCS scaleup n=16/18 done ===")
    print(f"json: {json_path}")
    print(f"md:   {md_path}")
    print(f"csv:  {csv_path}")
    print(f"fig:  {fig_path}")


if __name__ == "__main__":
    main()
