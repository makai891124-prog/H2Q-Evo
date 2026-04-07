from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .autocompute import (
        autotune_threads_with_matmul,
        cache_lookup,
        cache_store,
        default_thread_candidates,
        detect_hardware_profile,
        load_compute_plan_cache,
        make_cache_key,
        parse_thread_candidates,
        profile_as_dict,
        save_compute_plan_cache,
    )
except ImportError:
    from autocompute import (  # type: ignore
        autotune_threads_with_matmul,
        cache_lookup,
        cache_store,
        default_thread_candidates,
        detect_hardware_profile,
        load_compute_plan_cache,
        make_cache_key,
        parse_thread_candidates,
        profile_as_dict,
        save_compute_plan_cache,
    )


def _corr_theory(delta_deg: float, jitter_deg: float, flip_prob: float) -> float:
    # E_noisy = -cos(delta) * exp(-sigma^2) * (1-2p)^2
    # sigma is per-axis angular std in radians.
    delta = math.radians(delta_deg)
    sigma = math.radians(jitter_deg)
    visibility = math.exp(-(sigma**2))
    flip_scale = (1.0 - 2.0 * flip_prob) ** 2
    return float(-math.cos(delta) * visibility * flip_scale)


def _draw_products(E: float, n: int, rng: np.random.Generator) -> np.ndarray:
    p_same = min(1.0, max(0.0, (1.0 + E) * 0.5))
    a = rng.choice(np.array([-1, 1], dtype=int), size=n)
    same = rng.random(size=n) < p_same
    b = np.where(same, a, -a)
    return a * b


def _z_to_p_two_sided(z: float) -> float:
    # p = erfc(|z|/sqrt(2))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


@dataclass
class ConvergenceRow:
    n_pairs: int
    trials: int
    s_theory: float
    s_mean: float
    s_bias: float
    s_abs_error_mean: float
    s_rmse: float
    s_ci_coverage: float
    s_se_mean: float
    z_vs_classical_2: float
    p_two_sided_vs_classical_2: float


@dataclass
class ConvergenceSummary:
    noise_axis_jitter_deg: float
    noise_outcome_flip_prob: float
    n_list: list[int]
    trials: int
    hardware_profile: dict[str, object]
    compute_plan: dict[str, object]
    slope_log_rmse_vs_log_n: float
    intercept_log_rmse_vs_log_n: float
    r2_log_fit: float
    rows: list[ConvergenceRow]


def _resolve_compute_plan(
    compute_plan_cache: Path,
    refresh_compute_plan_cache: bool,
    autotune_threads: bool,
    thread_candidates: str | None,
    n_list: list[int],
    trials: int,
) -> tuple[dict[str, object], dict[str, object]]:
    hw = detect_hardware_profile()
    cands = (
        parse_thread_candidates(thread_candidates, cpu_count=max(1, hw.cpu_count))
        if autotune_threads
        else default_thread_candidates(max(1, hw.cpu_count))
    )
    if not cands:
        cands = [max(1, torch.get_num_threads())]

    cache_key = make_cache_key(
        namespace="convergence_experiment",
        hardware=hw,
        params={
            "n_list": n_list,
            "trials": int(trials),
            "autotune_threads": bool(autotune_threads),
            "thread_candidates": cands,
            "probe": "matmul",
        },
    )

    cache = load_compute_plan_cache(compute_plan_cache)
    cached = None if refresh_compute_plan_cache else cache_lookup(cache, cache_key)

    cache_hit = False
    probe_timings: dict[str, float] | None = None
    if cached is not None:
        cache_hit = True
        selected_threads = int(cached.get("selected_torch_threads", max(1, torch.get_num_threads())))
    else:
        if autotune_threads:
            selected_threads, probe_timings = autotune_threads_with_matmul(
                thread_candidates=cands,
                matmul_size=768,
                warmup=2,
                repeat=5,
            )
        else:
            selected_threads = cands[0]
            probe_timings = None
        cache_store(
            cache,
            cache_key,
            {
                "selected_torch_threads": int(selected_threads),
                "probe_timings_ms": probe_timings,
            },
        )
        save_compute_plan_cache(compute_plan_cache, cache)

    torch.set_num_threads(max(1, int(selected_threads)))
    plan = {
        "selected_torch_threads": int(torch.get_num_threads()),
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "autotune_timings_ms": probe_timings,
    }
    return profile_as_dict(hw), plan


def run_convergence_experiment(
    n_list: list[int],
    trials: int,
    seed: int,
    axis_jitter_deg: float,
    outcome_flip_prob: float,
    hardware_profile: dict[str, object] | None = None,
    compute_plan: dict[str, object] | None = None,
) -> ConvergenceSummary:
    rng = np.random.default_rng(seed)

    # CHSH angle set.
    deltas = {
        "ab": 45.0,
        "abp": 135.0,
        "apb": 45.0,
        "apbp": 45.0,
    }

    e_ab = _corr_theory(deltas["ab"], axis_jitter_deg, outcome_flip_prob)
    e_abp = _corr_theory(deltas["abp"], axis_jitter_deg, outcome_flip_prob)
    e_apb = _corr_theory(deltas["apb"], axis_jitter_deg, outcome_flip_prob)
    e_apbp = _corr_theory(deltas["apbp"], axis_jitter_deg, outcome_flip_prob)
    s_theory = e_ab - e_abp + e_apb + e_apbp

    rows: list[ConvergenceRow] = []

    for n in n_list:
        s_hats: list[float] = []
        covers = 0
        s_ses: list[float] = []

        for _ in range(trials):
            prod_ab = _draw_products(e_ab, n=n, rng=rng)
            prod_abp = _draw_products(e_abp, n=n, rng=rng)
            prod_apb = _draw_products(e_apb, n=n, rng=rng)
            prod_apbp = _draw_products(e_apbp, n=n, rng=rng)

            est_ab = float(np.mean(prod_ab))
            est_abp = float(np.mean(prod_abp))
            est_apb = float(np.mean(prod_apb))
            est_apbp = float(np.mean(prod_apbp))
            s_hat = est_ab - est_abp + est_apb + est_apbp
            s_hats.append(s_hat)

            se_ab = float(np.std(prod_ab, ddof=1) / math.sqrt(n))
            se_abp = float(np.std(prod_abp, ddof=1) / math.sqrt(n))
            se_apb = float(np.std(prod_apb, ddof=1) / math.sqrt(n))
            se_apbp = float(np.std(prod_apbp, ddof=1) / math.sqrt(n))
            se_s = math.sqrt(se_ab**2 + se_abp**2 + se_apb**2 + se_apbp**2)
            s_ses.append(se_s)

            lo, hi = s_hat - 1.96 * se_s, s_hat + 1.96 * se_s
            if lo <= s_theory <= hi:
                covers += 1

        s_arr = np.asarray(s_hats, dtype=float)
        errors = s_arr - s_theory
        s_mean = float(np.mean(s_arr))
        s_bias = float(s_mean - s_theory)
        s_abs_error_mean = float(np.mean(np.abs(errors)))
        s_rmse = float(np.sqrt(np.mean(errors**2)))
        s_ci_coverage = float(covers / trials)
        s_se_mean = float(np.mean(np.asarray(s_ses, dtype=float)))

        # Trial-level uncertainty of mean Shat.
        se_of_mean = float(np.std(s_arr, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0
        if se_of_mean > 0:
            z = (abs(s_mean) - 2.0) / se_of_mean
            p = _z_to_p_two_sided(z)
        else:
            z, p = 0.0, 1.0

        rows.append(
            ConvergenceRow(
                n_pairs=n,
                trials=trials,
                s_theory=float(s_theory),
                s_mean=s_mean,
                s_bias=s_bias,
                s_abs_error_mean=s_abs_error_mean,
                s_rmse=s_rmse,
                s_ci_coverage=s_ci_coverage,
                s_se_mean=s_se_mean,
                z_vs_classical_2=float(z),
                p_two_sided_vs_classical_2=float(p),
            )
        )

    # log-log fit: RMSE ~= c * N^slope
    x = np.log(np.asarray([r.n_pairs for r in rows], dtype=float))
    y = np.log(np.asarray([r.s_rmse for r in rows], dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ConvergenceSummary(
        noise_axis_jitter_deg=axis_jitter_deg,
        noise_outcome_flip_prob=outcome_flip_prob,
        n_list=n_list,
        trials=trials,
        hardware_profile=hardware_profile or {},
        compute_plan=compute_plan or {},
        slope_log_rmse_vs_log_n=float(slope),
        intercept_log_rmse_vs_log_n=float(intercept),
        r2_log_fit=float(r2),
        rows=rows,
    )


def _render_markdown(summary: ConvergenceSummary) -> str:
    lines: list[str] = []
    lines.append("# DAS-GQS Convergence Contrast Experiment")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- axis jitter (deg): {summary.noise_axis_jitter_deg}")
    lines.append(f"- outcome flip probability: {summary.noise_outcome_flip_prob}")
    lines.append(f"- N list: {summary.n_list}")
    lines.append(f"- trials per N: {summary.trials}")
    if summary.compute_plan:
        lines.append(f"- selected torch threads: {summary.compute_plan.get('selected_torch_threads')}")
        lines.append(f"- compute-plan cache hit: {summary.compute_plan.get('cache_hit')}")
    lines.append("")
    lines.append("## Convergence Fit")
    lines.append(f"- slope (log RMSE vs log N): {summary.slope_log_rmse_vs_log_n:.6f}")
    lines.append(f"- intercept: {summary.intercept_log_rmse_vs_log_n:.6f}")
    lines.append(f"- R^2: {summary.r2_log_fit:.6f}")
    lines.append("- ideal Monte Carlo convergence reference: slope = -0.5")
    lines.append("")
    lines.append("## Results")
    lines.append("| N | S_theory | S_mean | Bias | MAE | RMSE | CI coverage | z(|S|-2) | p(two-sided) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary.rows:
        lines.append(
            f"| {r.n_pairs} | {r.s_theory:.6f} | {r.s_mean:.6f} | {r.s_bias:.6f} | "
            f"{r.s_abs_error_mean:.6f} | {r.s_rmse:.6f} | {r.s_ci_coverage:.3f} | "
            f"{r.z_vs_classical_2:.3f} | {r.p_two_sided_vs_classical_2:.3e} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, summary: ConvergenceSummary) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_pairs",
                "trials",
                "s_theory",
                "s_mean",
                "s_bias",
                "s_abs_error_mean",
                "s_rmse",
                "s_ci_coverage",
                "s_se_mean",
                "z_vs_classical_2",
                "p_two_sided_vs_classical_2",
            ]
        )
        for r in summary.rows:
            writer.writerow(
                [
                    r.n_pairs,
                    r.trials,
                    r.s_theory,
                    r.s_mean,
                    r.s_bias,
                    r.s_abs_error_mean,
                    r.s_rmse,
                    r.s_ci_coverage,
                    r.s_se_mean,
                    r.z_vs_classical_2,
                    r.p_two_sided_vs_classical_2,
                ]
            )


def _plot(path: Path, summary: ConvergenceSummary) -> None:
    ns = np.asarray([r.n_pairs for r in summary.rows], dtype=float)
    rmse = np.asarray([r.s_rmse for r in summary.rows], dtype=float)
    mae = np.asarray([r.s_abs_error_mean for r in summary.rows], dtype=float)

    plt.figure(figsize=(8, 5), dpi=130)
    plt.loglog(ns, rmse, marker="o", linewidth=2.0, label="RMSE(|S_hat - S_theory|)")
    plt.loglog(ns, mae, marker="s", linewidth=1.8, label="MAE(|S_hat - S_theory|)")

    # N^-1/2 reference through first RMSE point.
    ref = rmse[0] * (ns / ns[0]) ** (-0.5)
    plt.loglog(ns, ref, linestyle="--", linewidth=1.5, label="Reference N^-1/2")

    plt.xlabel("Sample size N")
    plt.ylabel("Error")
    plt.title("DAS-GQS Convergence Under Fixed Noise")
    plt.grid(True, which="both", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DAS-GQS convergence contrast experiment")
    parser.add_argument("--n-list", type=str, default="200,500,1000,2000,5000,10000,20000")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--axis-jitter-deg", type=float, default=2.0)
    parser.add_argument("--outcome-flip-prob", type=float, default=0.03)
    parser.add_argument(
        "--autotune-threads",
        action="store_true",
        help="Autotune CPU thread count and cache selected plan.",
    )
    parser.add_argument(
        "--thread-candidates",
        type=str,
        default="",
        help="Comma-separated thread candidates, e.g. 1,2,4,6,8,10",
    )
    parser.add_argument(
        "--compute-plan-cache",
        type=Path,
        default=Path("reports/compute_plan_cache.json"),
        help="Offline compute plan cache file shared across DAS scripts.",
    )
    parser.add_argument(
        "--refresh-compute-plan-cache",
        action="store_true",
        help="Ignore cache and re-probe thread plan.",
    )
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    hardware_profile, compute_plan = _resolve_compute_plan(
        compute_plan_cache=args.compute_plan_cache,
        refresh_compute_plan_cache=args.refresh_compute_plan_cache,
        autotune_threads=args.autotune_threads,
        thread_candidates=args.thread_candidates,
        n_list=n_list,
        trials=args.trials,
    )
    summary = run_convergence_experiment(
        n_list=n_list,
        trials=args.trials,
        seed=args.seed,
        axis_jitter_deg=args.axis_jitter_deg,
        outcome_flip_prob=args.outcome_flip_prob,
        hardware_profile=hardware_profile,
        compute_plan=compute_plan,
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "das_gqs_convergence_curve.json"
    csv_path = out_dir / "das_gqs_convergence_curve.csv"
    md_path = out_dir / "das_gqs_convergence_curve.md"
    png_path = out_dir / "das_gqs_convergence_curve.png"

    json_path.write_text(json.dumps(asdict(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    _write_csv(csv_path, summary)
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    _plot(png_path, summary)

    print("=== DAS-GQS convergence experiment generated ===")
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print(f"md:   {md_path}")
    print(f"png:  {png_path}")
    print(f"slope(log RMSE vs log N): {summary.slope_log_rmse_vs_log_n:.6f}")
    print(
        "compute plan: "
        f"threads={summary.compute_plan.get('selected_torch_threads')}, "
        f"cache_hit={summary.compute_plan.get('cache_hit')}"
    )


if __name__ == "__main__":
    main()
