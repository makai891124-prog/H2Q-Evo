import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fault_tolerant_rsa_throughput_report import FTParams, evaluate_rsa_case


@dataclass
class TimeFoldConfig:
    max_workers: int = 8
    p_fail_per_hour: float = 0.02
    snapshot_overhead_s: float = 0.002
    snapshot_every_tasks: int = 20


def public_rsa_size_dataset() -> List[Dict[str, int]]:
    # Public RSA benchmark size classes (non-attack experiment).
    return [
        {"label": "RSA-100", "digits": 100, "tasks": 400},
        {"label": "RSA-129", "digits": 129, "tasks": 350},
        {"label": "RSA-250", "digits": 250, "tasks": 260},
        {"label": "RSA-512", "digits": 512, "tasks": 200},
        {"label": "RSA-768", "digits": 768, "tasks": 140},
        {"label": "RSA-1024", "digits": 1024, "tasks": 100},
        {"label": "RSA-2048", "digits": 2048, "tasks": 60},
    ]


def deterministic_modulus(bits: int) -> int:
    seed = hashlib.sha256(f"public-rsa-size-{bits}".encode("utf-8")).digest()
    x = int.from_bytes(seed, "big")
    rng = np.random.default_rng(x)

    # Construct arbitrary-size integer using 64-bit chunks.
    chunks = int(math.ceil(bits / 64.0))
    n = 0
    for _ in range(chunks):
        n = (n << 64) | int(rng.integers(0, (1 << 63), dtype=np.uint64))

    # Trim and force exact bit-width pattern.
    n &= (1 << bits) - 1
    n |= 1
    n |= (1 << (bits - 1))
    return n


def _pow_task(args: Tuple[int, int, int]) -> int:
    m, e, n = args
    return pow(m, e, n)


def build_messages(n: int, count: int) -> List[int]:
    rng = np.random.default_rng(n + count)
    msgs = []
    for _ in range(count):
        m = int(rng.integers(2, min(n - 1, (1 << 63) - 1), dtype=np.uint64))
        msgs.append(m)
    return msgs


def run_sequential_modexp(n: int, e: int, messages: List[int]) -> float:
    t0 = time.perf_counter()
    checksum = 0
    for m in messages:
        checksum ^= pow(m, e, n)
    dt = time.perf_counter() - t0
    return dt + (checksum & 0) * 0.0


def run_parallel_modexp_with_snapshots(
    n: int,
    e: int,
    messages: List[int],
    workers: int,
    cfg: TimeFoldConfig,
    snapshot_dir: Path,
    label: str,
) -> Dict[str, float]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    chunks = [messages[i : i + cfg.snapshot_every_tasks] for i in range(0, len(messages), cfg.snapshot_every_tasks)]
    completed = 0
    checksum = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk_id, chunk in enumerate(chunks):
            args = [(m, e, n) for m in chunk]
            for out in ex.map(_pow_task, args):
                checksum ^= int(out)

        completed += len(chunk)
        snap_path = snapshot_dir / f"{label}_snapshot_{chunk_id}.json"
        snap_payload = {"completed": completed, "checksum": int(checksum), "chunk_id": chunk_id}
        st = time.perf_counter()
        snap_path.write_text(json.dumps(snap_payload), encoding="utf-8")
        io_dt = time.perf_counter() - st
        # Normalize snapshot overhead with configured floor.
        _ = max(io_dt, cfg.snapshot_overhead_s)

    wall = time.perf_counter() - t0

    # Time-fold model: expected recomputation loss reduced by snapshot interval.
    fail_prob = cfg.p_fail_per_hour * (wall / 3600.0)
    no_snapshot_loss = fail_prob * (wall / 2.0)
    with_snapshot_loss = fail_prob * ((wall / max(len(chunks), 1)) / 2.0)
    folded_effective = wall + with_snapshot_loss
    folded_gain = (wall + no_snapshot_loss) / max(folded_effective, 1e-12)

    return {
        "wall_time_s": wall,
        "effective_time_folded_s": folded_effective,
        "expected_loss_no_snapshot_s": no_snapshot_loss,
        "expected_loss_with_snapshot_s": with_snapshot_loss,
        "time_fold_gain": folded_gain,
        "checksum": int(checksum),
    }


def analyze_dataset() -> Dict[str, object]:
    cfg = TimeFoldConfig(max_workers=min(8, os.cpu_count() or 4))
    ft = FTParams()

    dataset = public_rsa_size_dataset()
    ts = int(time.time())
    snapshot_root = Path("reports") / f"rsa_timefold_snapshots_{ts}"

    rows = []
    for item in dataset:
        digits = int(item["digits"])
        tasks = int(item["tasks"])
        bits = int(math.ceil(digits * math.log2(10)))

        n = deterministic_modulus(bits)
        e = 65537
        messages = build_messages(n, tasks)

        seq_s = run_sequential_modexp(n, e, messages)
        par = run_parallel_modexp_with_snapshots(
            n=n,
            e=e,
            messages=messages,
            workers=cfg.max_workers,
            cfg=cfg,
            snapshot_dir=snapshot_root,
            label=item["label"],
        )

        classical_parallel_speedup = seq_s / max(par["wall_time_s"], 1e-12)
        folded_speedup = seq_s / max(par["effective_time_folded_s"], 1e-12)

        # Compare against projected FT quantum runtime using prior model (non-attack, feasibility only).
        q_proj = evaluate_rsa_case(digits=digits, p_phys=1e-4, factory_count=1000, ft=ft)
        quantum_hours = q_proj["total_runtime_hours"]
        classical_hours = par["effective_time_folded_s"] / 3600.0
        quantum_vs_classical = classical_hours / max(quantum_hours, 1e-12)

        rows.append(
            {
                "label": item["label"],
                "digits": digits,
                "bits": bits,
                "tasks": tasks,
                "sequential_time_s": seq_s,
                "parallel_wall_time_s": par["wall_time_s"],
                "parallel_effective_folded_time_s": par["effective_time_folded_s"],
                "classical_parallel_speedup": classical_parallel_speedup,
                "classical_folded_speedup": folded_speedup,
                "time_fold_gain": par["time_fold_gain"],
                "ft_quantum_runtime_h_p1e4_F1000": quantum_hours,
                "classical_effective_runtime_h": classical_hours,
                "quantum_vs_classical_runtime_ratio": quantum_vs_classical,
                "snapshot_expected_loss_saved_s": par["expected_loss_no_snapshot_s"] - par["expected_loss_with_snapshot_s"],
                "checksum": par["checksum"],
            }
        )

    # Final usability and quantum-advantage judgment under this benchmark definition.
    advantage_cases = [r for r in rows if r["quantum_vs_classical_runtime_ratio"] > 5.0]
    scalable_signal = np.polyfit(
        np.array([r["digits"] for r in rows], dtype=np.float64),
        np.log(np.maximum(np.array([r["classical_folded_speedup"] for r in rows], dtype=np.float64), 1e-12)),
        1,
    )[0]

    verdict = {
        "has_final_usability": bool(np.mean([r["classical_folded_speedup"] for r in rows]) > 1.5),
        "has_real_quantum_advantage": bool(len(advantage_cases) >= 3),
        "advantage_case_count": int(len(advantage_cases)),
        "total_cases": int(len(rows)),
        "classical_timefold_scaling_slope": float(scalable_signal),
    }

    return {
        "config": cfg.__dict__,
        "rows": rows,
        "verdict": verdict,
        "snapshot_root": str(snapshot_root),
    }


def render_plots(rows: List[Dict[str, float]], p1: Path, p2: Path) -> None:
    rows = sorted(rows, key=lambda x: x["digits"])
    x = [r["digits"] for r in rows]
    y1 = [r["classical_parallel_speedup"] for r in rows]
    y2 = [r["classical_folded_speedup"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker="o", linewidth=2, label="Parallel speedup")
    plt.plot(x, y2, marker="s", linewidth=2, label="Snapshot time-fold speedup")
    plt.grid(alpha=0.25)
    plt.xlabel("RSA digits")
    plt.ylabel("Speedup over sequential baseline")
    plt.title("Public RSA dataset: parallel and time-fold speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p1, dpi=180)
    plt.close()

    yq = [r["quantum_vs_classical_runtime_ratio"] for r in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(x, yq, marker="D", linewidth=2, color="#aa3377")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.xlabel("RSA digits")
    plt.ylabel("Classical effective runtime / projected FT quantum runtime")
    plt.title("Projected FT quantum advantage ratio on public RSA classes")
    plt.tight_layout()
    plt.savefig(p2, dpi=180)
    plt.close()


def build_report(payload: Dict[str, object], data_path: Path, p1: Path, p2: Path) -> str:
    rows = sorted(payload["rows"], key=lambda x: x["digits"])
    verdict = payload["verdict"]
    usable_text = (
        "该系统在公开RSA非攻击运算任务上可稳定获得并行与时间折叠收益，具备工程可用性。"
        if verdict["has_final_usability"]
        else "该系统在当前任务粒度下并行与时间折叠收益不足，尚未达到工程可用门槛。"
    )
    advantage_text = (
        "在当前公开规模类与对照口径下，已观察到可验证的真实量子优越性信号。"
        if verdict["has_real_quantum_advantage"]
        else "在当前公开规模类与对照口径下，尚不能据此宣称最终达到真实量子优越性。"
    )

    lines = [
        "# 公开RSA数据集并行-快照时间折叠实验报告（非攻击）",
        "",
        "## 1. 实验目标",
        "",
        "在公开RSA规模数据集上执行非攻击性质的真实运算（模幂批处理验证），",
        "通过并行化与中间快照叠加时间折叠评估系统可用性，并对比容错量子吞吐预测，",
        "判断是否已具备真实量子优越性。",
        "",
        "## 2. 方法",
        "",
        "1. 数据集：RSA-100 到 RSA-2048 公开规模类。",
        "2. 真实运算：`pow(m, 65537, n)` 大整数模幂批处理（非破解，仅公开参数计算）。",
        "3. 加速策略：多进程并行 + 每固定任务数保存快照（用于故障恢复时间折叠）。",
        "4. 对比基线：顺序执行时间 vs 并行时间 vs 快照折叠有效时间。",
        "5. 量子对照：调用容错吞吐模型 `p_phys=1e-4, F=1000` 的预测总耗时。",
        "",
        "## 3. 结果汇总",
        "",
        "| RSA类 | 顺序时间(s) | 并行时间(s) | 折叠有效时间(s) | 并行加速 | 折叠加速 | 量子/经典时间比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['label']} | {r['sequential_time_s']:.4f} | {r['parallel_wall_time_s']:.4f} | "
            f"{r['parallel_effective_folded_time_s']:.4f} | {r['classical_parallel_speedup']:.2f}x | "
            f"{r['classical_folded_speedup']:.2f}x | {r['quantum_vs_classical_runtime_ratio']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 4. 判定",
            "",
            f"- 最终可用性（并行+折叠平均收益门槛）：`{verdict['has_final_usability']}`",
            f"- 真实量子优越性（本实验定义）：`{verdict['has_real_quantum_advantage']}`",
            f"- 优越案例数：`{verdict['advantage_case_count']}/{verdict['total_cases']}`",
            f"- 时间折叠规模斜率：`{verdict['classical_timefold_scaling_slope']:.6f}`",
            "",
            "## 5. 结论",
            "",
            f"1. {usable_text}",
            f"2. {advantage_text}",
            "3. 结果可作为后续扩展到更高并行工厂、更低物理门错率时的基线验证数据。",
            "",
            "## 6. 附件",
            "",
            f"- 数据：`{data_path}`",
            f"- 加速图：`{p1}`",
            f"- 量子对比图：`{p2}`",
            f"- 快照目录：`{payload['snapshot_root']}`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    payload = analyze_dataset()

    ts = int(time.time())
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"rsa_public_timefold_analysis_{ts}.json"
    p1 = out_dir / f"rsa_public_parallel_timefold_speedup_{ts}.png"
    p2 = out_dir / f"rsa_public_quantum_comparison_{ts}.png"
    report_path = out_dir / f"公开RSA并行快照时间折叠分析报告_{ts}.md"

    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plots(payload["rows"], p1, p2)
    report = build_report(payload, data_path, p1, p2)
    report_path.write_text(report, encoding="utf-8")

    print("Public RSA time-fold analysis completed")
    print(f"Verdict: {payload['verdict']}")
    print(f"Data: {data_path}")
    print(f"Speedup plot: {p1}")
    print(f"Quantum compare plot: {p2}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
