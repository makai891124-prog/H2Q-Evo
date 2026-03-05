import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FTParams:
    code_cycle_us: float = 1.0
    factory_logical_qubits: int = 150
    factory_yield: float = 0.8
    factory_latency_cycles_per_d: float = 8.0
    physical_qubits_per_logical_per_d2: float = 3.0


def shor_resource_estimate(decimal_digits: int) -> Dict[str, float]:
    n_bits = int(math.ceil(decimal_digits * math.log2(10)))
    logical_qubits = int(2 * n_bits + 3)
    t_count = float(40.0 * (n_bits**3))
    return {
        "decimal_digits": decimal_digits,
        "n_bits": n_bits,
        "logical_qubits": logical_qubits,
        "t_count": t_count,
    }


def logical_error_per_t_gate(p_phys: float, d: int) -> float:
    # Surface-code style scaling approximation.
    return 0.1 * (100.0 * p_phys) ** ((d + 1) / 2.0)


def select_code_distance(p_phys: float, total_t_count: float) -> Dict[str, float]:
    target_logical_error = 1.0 / (100.0 * max(total_t_count, 1.0))
    for d in range(5, 128, 2):
        p_l = logical_error_per_t_gate(p_phys, d)
        if p_l <= target_logical_error:
            return {
                "code_distance": d,
                "logical_error_per_t": p_l,
                "target_logical_error": target_logical_error,
            }

    d = 127
    p_l = logical_error_per_t_gate(p_phys, d)
    return {
        "code_distance": d,
        "logical_error_per_t": p_l,
        "target_logical_error": target_logical_error,
    }


def throughput_t_per_sec(factory_count: int, code_distance: int, ft: FTParams) -> float:
    latency_us = ft.factory_latency_cycles_per_d * code_distance * ft.code_cycle_us
    single_factory_rate = (1e6 / max(latency_us, 1e-9)) * ft.factory_yield
    return factory_count * single_factory_rate


def physical_qubit_overhead(
    logical_algo_qubits: int,
    factory_count: int,
    code_distance: int,
    ft: FTParams,
) -> float:
    total_logical = logical_algo_qubits + factory_count * ft.factory_logical_qubits
    return total_logical * ft.physical_qubits_per_logical_per_d2 * (code_distance**2)


def evaluate_rsa_case(
    digits: int,
    p_phys: float,
    factory_count: int,
    ft: FTParams,
) -> Dict[str, float]:
    res = shor_resource_estimate(digits)
    d_sel = select_code_distance(p_phys, res["t_count"])

    tps = throughput_t_per_sec(factory_count, d_sel["code_distance"], ft)
    total_seconds = res["t_count"] / max(tps, 1e-12)
    phys_qubits = physical_qubit_overhead(
        logical_algo_qubits=res["logical_qubits"],
        factory_count=factory_count,
        code_distance=d_sel["code_distance"],
        ft=ft,
    )

    return {
        "decimal_digits": digits,
        "n_bits": res["n_bits"],
        "t_count": res["t_count"],
        "logical_qubits": res["logical_qubits"],
        "p_phys": p_phys,
        "factory_count": factory_count,
        "code_distance": d_sel["code_distance"],
        "logical_error_per_t": d_sel["logical_error_per_t"],
        "target_logical_error": d_sel["target_logical_error"],
        "throughput_t_per_sec": tps,
        "total_runtime_hours": total_seconds / 3600.0,
        "physical_qubits_total": phys_qubits,
    }


def required_hardware_for_target_runtime(
    digits: int,
    p_phys: float,
    target_hours: float,
    ft: FTParams,
) -> Dict[str, float]:
    res = shor_resource_estimate(digits)
    d_sel = select_code_distance(p_phys, res["t_count"])

    single_factory_tps = throughput_t_per_sec(1, d_sel["code_distance"], ft)
    required_tps = res["t_count"] / max(target_hours * 3600.0, 1e-12)
    factories_needed = int(math.ceil(required_tps / max(single_factory_tps, 1e-12)))

    phys_qubits = physical_qubit_overhead(
        logical_algo_qubits=res["logical_qubits"],
        factory_count=factories_needed,
        code_distance=d_sel["code_distance"],
        ft=ft,
    )

    return {
        "decimal_digits": digits,
        "p_phys": p_phys,
        "target_hours": target_hours,
        "code_distance": d_sel["code_distance"],
        "factories_needed": factories_needed,
        "required_physical_qubits": phys_qubits,
        "required_tps": required_tps,
    }


def render_runtime_plot(results: List[Dict[str, float]], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    combos = sorted(set((r["p_phys"], r["factory_count"]) for r in results))
    for p_phys, factories in combos:
        rows = [r for r in results if r["p_phys"] == p_phys and r["factory_count"] == factories]
        rows.sort(key=lambda x: x["decimal_digits"])
        x = [r["decimal_digits"] for r in rows]
        y = [r["total_runtime_hours"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=1.8, label=f"p={p_phys:.0e}, F={factories}")

    plt.yscale("log")
    plt.xlabel("RSA decimal digits")
    plt.ylabel("Total runtime (hours, log)")
    plt.title("Fault-tolerant RSA throughput simulation")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def render_required_hardware_curve(required_rows: List[Dict[str, float]], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    p_list = sorted(set(r["p_phys"] for r in required_rows))
    for p in p_list:
        rows = [r for r in required_rows if r["p_phys"] == p]
        rows.sort(key=lambda x: x["decimal_digits"])
        x = [r["decimal_digits"] for r in rows]
        y = [r["required_physical_qubits"] for r in rows]
        plt.plot(x, y, marker="s", linewidth=2.0, label=f"p_phys={p:.0e}")

    plt.yscale("log")
    plt.xlabel("RSA decimal digits")
    plt.ylabel("Required physical qubits for target runtime (log)")
    plt.title("Required hardware scale for feasible RSA decoding")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_report(
    results: List[Dict[str, float]],
    required_rows: List[Dict[str, float]],
    data_path: Path,
    runtime_plot: Path,
    hardware_plot: Path,
) -> str:
    # Benchmark assumption for "how far" comparison.
    current_physical_qubits_assumed = 1e6
    current_factory_assumed = 100

    focus = [
        r
        for r in required_rows
        if r["decimal_digits"] in [1024, 2048]
        and abs(r["p_phys"] - 1e-4) < 1e-20
        and abs(r["target_hours"] - 24.0) < 1e-9
    ]

    gap_lines = []
    for row in focus:
        q_gap_orders = math.log10(max(row["required_physical_qubits"] / current_physical_qubits_assumed, 1.0))
        f_gap_orders = math.log10(max(row["factories_needed"] / current_factory_assumed, 1.0))
        gap_lines.append(
            f"- RSA-{row['decimal_digits']}（24小时目标, p=1e-4）: 物理比特缺口约 `10^{q_gap_orders:.2f}` 倍，"
            f"工厂并行缺口约 `10^{f_gap_orders:.2f}` 倍"
        )

    lines = [
        "# 容错并行吞吐仿真报告（不执行破解）",
        "",
        "## 1. 目标",
        "",
        "按并行蒸馏工厂数量估算 RSA-100 到 RSA-2048 的 T 门吞吐与总耗时，",
        "并加入物理门错误率、码距与物理比特开销，评估达到“可行破解”所需硬件规模。",
        "",
        "## 2. 模型设定",
        "",
        "- 算法复杂度：Shor 资源估计（逻辑比特与 T 门数量级）",
        "- 容错模型：表面码近似 `p_L ≈ 0.1*(100*p_phys)^((d+1)/2)`",
        "- 蒸馏并行：工厂数 `F` 线性提升 T 态吞吐",
        "- 目标可行性：给定总耗时门槛（本报告使用 24 小时）反推所需硬件规模",
        "",
        "## 3. 吞吐与耗时样例（节选）",
        "",
        "| RSA位数 | p_phys | 工厂数F | 码距d | 吞吐(T/s) | 总耗时(h) | 物理比特总量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    sample_rows = [
        r for r in results if r["decimal_digits"] in [100, 512, 1024, 2048] and r["factory_count"] in [100, 1000]
    ]
    sample_rows.sort(key=lambda x: (x["decimal_digits"], x["p_phys"], x["factory_count"]))

    for r in sample_rows:
        lines.append(
            f"| {r['decimal_digits']} | {r['p_phys']:.0e} | {int(r['factory_count'])} | {int(r['code_distance'])} | "
            f"{r['throughput_t_per_sec']:.3e} | {r['total_runtime_hours']:.3e} | {r['physical_qubits_total']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 4. 可行破解硬件规模曲线（24小时目标）",
            "",
            "| RSA位数 | p_phys | 码距d | 所需工厂数 | 所需物理比特 |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    req_show = [r for r in required_rows if r["decimal_digits"] in [100, 512, 1024, 2048] and abs(r["target_hours"] - 24.0) < 1e-9]
    req_show.sort(key=lambda x: (x["decimal_digits"], x["p_phys"]))
    for r in req_show:
        lines.append(
            f"| {r['decimal_digits']} | {r['p_phys']:.0e} | {int(r['code_distance'])} | {int(r['factories_needed'])} | {r['required_physical_qubits']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 5. 与当前能力的数量级差距",
            "",
            "假设当前可用能力基线：`1e6` 物理比特、`100` 蒸馏工厂并行。",
            *gap_lines,
            "",
            "## 6. 结论",
            "",
            "1. 并行蒸馏工厂可显著提升 T 门吞吐并缩短总耗时，但 RSA-1024/2048 仍需极大容错硬件规模。",
            "2. 物理门错误率每下降一个量级，可显著降低码距与总物理比特开销。",
            "3. 本仿真给出的是“可执行实破门槛”的工程距离评估，不等于实际执行破解。",
            "",
            "## 7. 附件",
            "",
            f"- 原始数据：`{data_path}`",
            f"- 吞吐耗时图：`{runtime_plot}`",
            f"- 硬件规模曲线：`{hardware_plot}`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ft = FTParams()

    rsa_digits = [100, 129, 250, 512, 768, 1024, 2048]
    p_phys_list = [1e-3, 1e-4, 1e-5]
    factory_counts = [10, 100, 1000, 10000]

    results: List[Dict[str, float]] = []
    for digits in rsa_digits:
        for p_phys in p_phys_list:
            for f in factory_counts:
                results.append(evaluate_rsa_case(digits, p_phys, f, ft))

    required_rows: List[Dict[str, float]] = []
    for digits in rsa_digits:
        for p_phys in p_phys_list:
            required_rows.append(required_hardware_for_target_runtime(digits, p_phys, 24.0, ft))
            required_rows.append(required_hardware_for_target_runtime(digits, p_phys, 24.0 * 30.0, ft))

    ts = int(time.time())
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"ft_rsa_throughput_simulation_{ts}.json"
    runtime_plot = out_dir / f"ft_rsa_runtime_vs_factories_{ts}.png"
    hardware_plot = out_dir / f"ft_rsa_required_hardware_curve_{ts}.png"
    report_path = out_dir / f"容错并行吞吐仿真报告_{ts}.md"

    payload = {
        "ft_params": ft.__dict__,
        "results": results,
        "required_hardware_for_target": required_rows,
    }
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    render_runtime_plot(results, runtime_plot)
    render_required_hardware_curve(required_rows, hardware_plot)
    report = build_report(results, required_rows, data_path, runtime_plot, hardware_plot)
    report_path.write_text(report, encoding="utf-8")

    print("Fault-tolerant RSA throughput simulation completed")
    print(f"Data: {data_path}")
    print(f"Runtime plot: {runtime_plot}")
    print(f"Hardware curve: {hardware_plot}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
