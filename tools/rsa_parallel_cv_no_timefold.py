import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fault_tolerant_rsa_throughput_report import FTParams, evaluate_rsa_case


def public_rsa_dataset_large_parallel() -> List[Dict[str, int]]:
    return [
        {"label": "RSA-512", "digits": 512, "tasks": 3600},
        {"label": "RSA-768", "digits": 768, "tasks": 3000},
        {"label": "RSA-1024", "digits": 1024, "tasks": 2400},
        {"label": "RSA-1536", "digits": 1536, "tasks": 1600},
        {"label": "RSA-2048", "digits": 2048, "tasks": 1000},
        {"label": "RSA-3072", "digits": 3072, "tasks": 600},
    ]


def deterministic_odd_modulus(bits: int) -> int:
    rng = np.random.default_rng(bits * 10007 + 97)
    chunks = int(math.ceil(bits / 64.0))
    n = 0
    for _ in range(chunks):
        n = (n << 64) | int(rng.integers(0, 1 << 63, dtype=np.uint64))
    n &= (1 << bits) - 1
    n |= (1 << (bits - 1))
    n |= 1
    return n


def make_messages(n: int, tasks: int, fold_seed: int) -> List[int]:
    # Limit sampled values to 63-bit to keep generation cheap while preserving modular arithmetic complexity in pow.
    rng = np.random.default_rng((n ^ fold_seed) & 0xFFFFFFFF)
    upper = min(n - 1, (1 << 63) - 1)
    vals = rng.integers(2, max(3, upper), size=tasks, dtype=np.uint64)
    return [int(v) for v in vals]


def batch_pow_checksum(messages: List[int], e: int, n: int) -> int:
    cs = 0
    for m in messages:
        cs ^= pow(m, e, n)
    return cs


def run_sequential(messages: List[int], e: int, n: int) -> Tuple[float, int]:
    t0 = time.perf_counter()
    cs = batch_pow_checksum(messages, e, n)
    return time.perf_counter() - t0, cs


def _worker_batch(args: Tuple[List[int], int, int]) -> int:
    chunk, e, n = args
    return batch_pow_checksum(chunk, e, n)


def split_chunks(arr: List[int], k: int) -> List[List[int]]:
    k = max(1, k)
    size = int(math.ceil(len(arr) / k))
    return [arr[i : i + size] for i in range(0, len(arr), size)]


def run_parallel(messages: List[int], e: int, n: int, workers: int) -> Tuple[float, int]:
    chunks = split_chunks(messages, workers * 2)
    args = [(c, e, n) for c in chunks]
    t0 = time.perf_counter()
    cs = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_worker_batch, args):
            cs ^= int(out)
    return time.perf_counter() - t0, cs


def mean_ci95(xs: List[float]) -> Dict[str, float]:
    arr = np.array(xs, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return {"mean": mean, "ci95": 0.0, "std": 0.0}
    std = float(np.std(arr, ddof=1))
    ci = 1.96 * std / math.sqrt(len(arr))
    return {"mean": mean, "ci95": float(ci), "std": std}


def cross_validate_parallel_advantage(k_folds: int = 4) -> Dict[str, object]:
    dataset = public_rsa_dataset_large_parallel()
    workers = min(12, max(2, (os.cpu_count() or 4)))
    e = 65537
    ft = FTParams()

    rows = []
    for item in dataset:
        digits = int(item["digits"])
        bits = int(math.ceil(digits * math.log2(10)))
        tasks = int(item["tasks"])

        n = deterministic_odd_modulus(bits)

        seq_times = []
        par_times = []
        speedups = []
        qratios = []
        checks = []

        for fold in range(k_folds):
            messages = make_messages(n, tasks, fold_seed=2026 + 17 * fold)
            seq_t, seq_cs = run_sequential(messages, e, n)
            par_t, par_cs = run_parallel(messages, e, n, workers)

            # Integrity check: parallel and sequential should produce same checksum.
            if seq_cs != par_cs:
                raise RuntimeError(f"Checksum mismatch for {item['label']} fold={fold}")

            seq_times.append(seq_t)
            par_times.append(par_t)
            speedups.append(seq_t / max(par_t, 1e-12))
            checks.append(int(seq_cs))

            q_pred = evaluate_rsa_case(digits=digits, p_phys=1e-4, factory_count=1000, ft=ft)
            classical_h = par_t / 3600.0
            q_h = q_pred["total_runtime_hours"]
            qratios.append(classical_h / max(q_h, 1e-12))

        seq_stat = mean_ci95(seq_times)
        par_stat = mean_ci95(par_times)
        spd_stat = mean_ci95(speedups)
        q_stat = mean_ci95(qratios)

        rows.append(
            {
                "label": item["label"],
                "digits": digits,
                "bits": bits,
                "tasks": tasks,
                "workers": workers,
                "sequential_s": seq_stat,
                "parallel_s": par_stat,
                "parallel_speedup": spd_stat,
                "quantum_vs_classical_ratio": q_stat,
                "checksums": checks,
            }
        )

    rows_sorted = sorted(rows, key=lambda x: x["digits"])
    x = np.array([r["digits"] for r in rows_sorted], dtype=np.float64)
    y = np.array([r["parallel_speedup"]["mean"] for r in rows_sorted], dtype=np.float64)
    slope = float(np.polyfit(x, np.log(np.maximum(y, 1e-12)), deg=1)[0])

    usable = bool(np.mean(y) > 1.2)
    # ratio > 1 means projected quantum faster than measured classical pipeline.
    q_means = np.array([r["quantum_vs_classical_ratio"]["mean"] for r in rows_sorted], dtype=np.float64)
    quantum_adv_cases = int(np.sum(q_means > 1.0))

    verdict = {
        "k_folds": k_folds,
        "workers": int(rows_sorted[0]["workers"] if rows_sorted else workers),
        "mean_parallel_speedup": float(np.mean(y)) if len(y) else 0.0,
        "parallel_scaling_log_slope": slope,
        "has_final_usability": usable,
        "quantum_advantage_case_count": quantum_adv_cases,
        "total_case_count": int(len(rows_sorted)),
        "has_real_quantum_advantage": bool(quantum_adv_cases >= max(1, len(rows_sorted) // 2)),
    }

    return {"rows": rows_sorted, "verdict": verdict}


def render_plots(rows: List[Dict[str, object]], p_speed: Path, p_q: Path) -> None:
    xs = [r["digits"] for r in rows]
    sp = [r["parallel_speedup"]["mean"] for r in rows]
    sp_err = [r["parallel_speedup"]["ci95"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.errorbar(xs, sp, yerr=sp_err, marker="o", linewidth=2, capsize=4)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(alpha=0.25)
    plt.xlabel("RSA digits")
    plt.ylabel("Parallel speedup over sequential")
    plt.title("Large-scale parallel cross-validation speedup (no time-fold overhead)")
    plt.tight_layout()
    plt.savefig(p_speed, dpi=180)
    plt.close()

    q = [r["quantum_vs_classical_ratio"]["mean"] for r in rows]
    q_err = [r["quantum_vs_classical_ratio"]["ci95"] for r in rows]
    plt.figure(figsize=(10, 6))
    plt.errorbar(xs, q, yerr=q_err, marker="s", linewidth=2, capsize=4, color="#aa3377")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.xlabel("RSA digits")
    plt.ylabel("Classical parallel runtime / projected FT quantum runtime")
    plt.title("Cross-validated projected quantum advantage ratio")
    plt.tight_layout()
    plt.savefig(p_q, dpi=180)
    plt.close()


def build_report(payload: Dict[str, object], data_path: Path, p_speed: Path, p_q: Path) -> str:
    rows = payload["rows"]
    verdict = payload["verdict"]

    lines = [
        "# 大规模并行去时间折叠开销交叉验证报告（公开RSA非攻击计算）",
        "",
        "## 1. 实验目标",
        "",
        "在去除时间折叠开销后，进行大规模并行与真实交叉验证，判断系统是否具备真实量子优越性。",
        "",
        "## 2. 实验设计",
        "",
        "1. 数据集：公开RSA规模类（RSA-512 到 RSA-3072）。",
        "2. 真实运算：大整数模幂 `pow(m, 65537, n)` 批处理（非攻击、仅公开参数计算）。",
        "3. 并行策略：大任务分块 + 多进程池常驻，去除时间折叠与快照开销影响。",
        "4. 交叉验证：K-fold 重复（不同消息样本），输出均值与95%置信区间。",
        "5. 量子对照：容错吞吐模型（`p_phys=1e-4`, `F=1000`）预测耗时。",
        "",
        "## 3. 交叉验证结果",
        "",
        "| RSA类 | 任务数 | 顺序时间(s) | 并行时间(s) | 并行加速(均值±CI95) | 量子/经典比(均值±CI95) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['label']} | {r['tasks']} | {r['sequential_s']['mean']:.4f} | {r['parallel_s']['mean']:.4f} | "
            f"{r['parallel_speedup']['mean']:.3f} +/- {r['parallel_speedup']['ci95']:.3f} | "
            f"{r['quantum_vs_classical_ratio']['mean']:.3e} +/- {r['quantum_vs_classical_ratio']['ci95']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 4. 判定",
            "",
            f"- 交叉验证折数：`{verdict['k_folds']}`",
            f"- 并行worker数：`{verdict['workers']}`",
            f"- 平均并行加速：`{verdict['mean_parallel_speedup']:.3f}x`",
            f"- 并行规模对数斜率：`{verdict['parallel_scaling_log_slope']:.6f}`",
            f"- 最终可用性：`{verdict['has_final_usability']}`",
            f"- 真实量子优越案例：`{verdict['quantum_advantage_case_count']}/{verdict['total_case_count']}`",
            f"- 是否具备真实量子优越性：`{verdict['has_real_quantum_advantage']}`",
            "",
            "## 5. 结论",
            "",
            (
                "1. 去除时间折叠开销后，并行流水线已达到可用门槛。"
                if verdict["has_final_usability"]
                else "1. 去除时间折叠开销后，并行流水线仍未达到可用门槛。"
            ),
            (
                "2. 交叉验证显示在多数公开RSA规模类上具备真实量子优越性。"
                if verdict["has_real_quantum_advantage"]
                else "2. 交叉验证显示在公开RSA规模类上尚未具备真实量子优越性。"
            ),
            "3. 该报告提供了可复现实验数据、统计置信区间与清晰判据，可作为后续优化基线。",
            "",
            "## 6. 附件",
            "",
            f"- 数据：`{data_path}`",
            f"- 并行加速图：`{p_speed}`",
            f"- 量子对比图：`{p_q}`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    payload = cross_validate_parallel_advantage(k_folds=4)

    ts = int(time.time())
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"rsa_parallel_cv_no_timefold_{ts}.json"
    p_speed = out_dir / f"rsa_parallel_cv_speedup_{ts}.png"
    p_q = out_dir / f"rsa_parallel_cv_quantum_ratio_{ts}.png"
    report_path = out_dir / f"公开RSA去时间折叠并行交叉验证报告_{ts}.md"

    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plots(payload["rows"], p_speed, p_q)
    report = build_report(payload, data_path, p_speed, p_q)
    report_path.write_text(report, encoding="utf-8")

    print("RSA parallel no-timefold cross-validation completed")
    print(f"Verdict: {payload['verdict']}")
    print(f"Data: {data_path}")
    print(f"Speedup plot: {p_speed}")
    print(f"Quantum ratio plot: {p_q}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
