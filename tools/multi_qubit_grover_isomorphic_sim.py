import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


@dataclass
class HardwareParams:
    name: str
    t1_us: float
    t2_us: float
    single_gate_ns: float
    two_qubit_gate_ns: float
    readout_ns: float
    reset_ns: float
    f1q: float
    f2q: float


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_full_single_qubit_gate(n_qubits: int, qubit: int, gate: np.ndarray) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[qubit] = gate
    return kron_n(ops)


def build_phase_flip_on_basis_state(n_qubits: int, basis_index: int) -> np.ndarray:
    dim = 2**n_qubits
    diag = np.ones(dim, dtype=np.complex128)
    diag[basis_index] = -1.0
    return np.diag(diag)


def build_uniform_reflection(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    ket_s = np.ones((dim, 1), dtype=np.complex128) / np.sqrt(dim)
    return 2.0 * (ket_s @ ket_s.conj().T) - np.eye(dim, dtype=np.complex128)


def apply_unitary(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    return u @ rho @ u.conj().T


def apply_channel_single_qubit(
    rho: np.ndarray,
    n_qubits: int,
    qubit: int,
    kraus_ops: List[np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(rho, dtype=np.complex128)
    for k in kraus_ops:
        k_full = build_full_single_qubit_gate(n_qubits, qubit, k)
        out += k_full @ rho @ k_full.conj().T
    return out


def apply_amplitude_damping_all_qubits(
    rho: np.ndarray,
    n_qubits: int,
    gamma: float,
) -> np.ndarray:
    if gamma <= 0.0:
        return rho

    gamma = min(max(gamma, 0.0), 1.0)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    out = rho
    for q in range(n_qubits):
        out = apply_channel_single_qubit(out, n_qubits, q, [k0, k1])
    return out


def apply_phase_damping_all_qubits(
    rho: np.ndarray,
    n_qubits: int,
    lam: float,
) -> np.ndarray:
    if lam <= 0.0:
        return rho

    lam = min(max(lam, 0.0), 1.0)
    k0 = np.sqrt(1.0 - lam) * np.eye(2, dtype=np.complex128)
    proj0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    proj1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    k1 = np.sqrt(lam) * proj0
    k2 = np.sqrt(lam) * proj1

    out = rho
    for q in range(n_qubits):
        out = apply_channel_single_qubit(out, n_qubits, q, [k0, k1, k2])
    return out


def apply_depolarizing_all_qubits(
    rho: np.ndarray,
    n_qubits: int,
    p: float,
) -> np.ndarray:
    if p <= 0.0:
        return rho

    p = min(max(p, 0.0), 1.0)
    py = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    pz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    k0 = np.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    kx = np.sqrt(p / 3.0) * X
    ky = np.sqrt(p / 3.0) * py
    kz = np.sqrt(p / 3.0) * pz

    out = rho
    for q in range(n_qubits):
        out = apply_channel_single_qubit(out, n_qubits, q, [k0, kx, ky, kz])
    return out


def noise_after_block(
    rho: np.ndarray,
    n_qubits: int,
    dt_ns: float,
    per_qubit_depol: float,
    hw: HardwareParams,
) -> np.ndarray:
    t1_ns = hw.t1_us * 1000.0
    t2_ns = hw.t2_us * 1000.0

    gamma = 1.0 - np.exp(-dt_ns / max(t1_ns, 1e-12))

    inv_tphi = (1.0 / max(t2_ns, 1e-12)) - (1.0 / max(2.0 * t1_ns, 1e-12))
    tphi_ns = np.inf if inv_tphi <= 0 else 1.0 / inv_tphi
    lam = 0.0 if not np.isfinite(tphi_ns) else 1.0 - np.exp(-dt_ns / max(tphi_ns, 1e-12))

    out = apply_amplitude_damping_all_qubits(rho, n_qubits, gamma)
    out = apply_phase_damping_all_qubits(out, n_qubits, lam)
    out = apply_depolarizing_all_qubits(out, n_qubits, per_qubit_depol)
    return out


def mcz_two_qubit_gate_count(n_qubits: int) -> int:
    # Practical compiled estimate with ancilla support on modern control stacks.
    return max(1, 2 * n_qubits - 2)


def block_timing_and_depol(n_qubits: int, hw: HardwareParams) -> Dict[str, float]:
    mcz_2q = mcz_two_qubit_gate_count(n_qubits)

    prep_1q = n_qubits
    oracle_2q = mcz_2q
    diffuser_1q = 4 * n_qubits
    diffuser_2q = mcz_2q

    prep_ns = prep_1q * hw.single_gate_ns
    oracle_ns = oracle_2q * hw.two_qubit_gate_ns
    diffuser_ns = diffuser_1q * hw.single_gate_ns + diffuser_2q * hw.two_qubit_gate_ns

    # Convert aggregate block fidelity to an equivalent per-qubit depolarizing strength.
    prep_fidelity = hw.f1q ** prep_1q
    oracle_fidelity = hw.f2q ** oracle_2q
    diffuser_fidelity = (hw.f1q ** diffuser_1q) * (hw.f2q ** diffuser_2q)

    prep_depol = 1.0 - (prep_fidelity ** (1.0 / n_qubits))
    oracle_depol = 1.0 - (oracle_fidelity ** (1.0 / n_qubits))
    diffuser_depol = 1.0 - (diffuser_fidelity ** (1.0 / n_qubits))

    return {
        "prep_ns": float(prep_ns),
        "oracle_ns": float(oracle_ns),
        "diffuser_ns": float(diffuser_ns),
        "prep_depol": float(prep_depol),
        "oracle_depol": float(oracle_depol),
        "diffuser_depol": float(diffuser_depol),
    }


def prepare_initial_density(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    psi0 = np.zeros((dim, 1), dtype=np.complex128)
    psi0[0, 0] = 1.0
    return psi0 @ psi0.conj().T


def build_h_all(n_qubits: int) -> np.ndarray:
    ops = [H for _ in range(n_qubits)]
    return kron_n(ops)


def simulate_noisy_grover(
    n_qubits: int,
    target_index: int,
    hw: HardwareParams,
) -> Dict[str, float]:
    dim = 2**n_qubits
    k_opt = max(1, int(round((np.pi / 4.0) * np.sqrt(dim))))

    h_all = build_h_all(n_qubits)
    oracle_u = build_phase_flip_on_basis_state(n_qubits, target_index)
    # Exact diffuser unitary; runtime/noise still uses gate-level decomposition estimates.
    diffuser_u = build_uniform_reflection(n_qubits)

    timings = block_timing_and_depol(n_qubits, hw)

    rho = prepare_initial_density(n_qubits)

    # State preparation: H on all qubits.
    rho = apply_unitary(rho, h_all)
    rho = noise_after_block(rho, n_qubits, timings["prep_ns"], timings["prep_depol"], hw)

    for _ in range(k_opt):
        # Oracle: flip target phase.
        rho = apply_unitary(rho, oracle_u)
        rho = noise_after_block(rho, n_qubits, timings["oracle_ns"], timings["oracle_depol"], hw)

        # Diffuser (inversion-about-mean) as exact unitary.
        rho = apply_unitary(rho, diffuser_u)
        rho = noise_after_block(rho, n_qubits, timings["diffuser_ns"], timings["diffuser_depol"], hw)

    p_success = float(np.real(rho[target_index, target_index]))
    p_success = min(max(p_success, 1e-9), 1.0)

    classical_queries = (dim + 1.0) / 2.0
    quantum_expected_queries = k_opt / p_success

    quantum_single_run_time_ns = timings["prep_ns"] + k_opt * (timings["oracle_ns"] + timings["diffuser_ns"]) + hw.readout_ns + hw.reset_ns
    quantum_expected_time_ns = quantum_single_run_time_ns / p_success

    classical_oracle_ns = timings["oracle_ns"]
    classical_expected_time_ns = classical_queries * classical_oracle_ns

    query_speedup = classical_queries / quantum_expected_queries
    time_speedup = classical_expected_time_ns / quantum_expected_time_ns

    return {
        "n_qubits": n_qubits,
        "search_space": dim,
        "target_index": target_index,
        "grover_iterations": int(k_opt),
        "success_probability": p_success,
        "classical_expected_queries": float(classical_queries),
        "quantum_expected_queries": float(quantum_expected_queries),
        "query_speedup_over_classical": float(query_speedup),
        "quantum_single_run_time_us": float(quantum_single_run_time_ns / 1000.0),
        "quantum_expected_time_us": float(quantum_expected_time_ns / 1000.0),
        "classical_expected_time_us": float(classical_expected_time_ns / 1000.0),
        "time_speedup_over_classical": float(time_speedup),
        "timing_model": timings,
    }


def aggregate_conclusion(experiments: List[Dict[str, float]]) -> Dict[str, float]:
    query_speedups = np.array([e["query_speedup_over_classical"] for e in experiments], dtype=np.float64)
    time_speedups = np.array([e["time_speedup_over_classical"] for e in experiments], dtype=np.float64)
    success_probs = np.array([e["success_probability"] for e in experiments], dtype=np.float64)

    credible_cases = [
        e
        for e in experiments
        if e["success_probability"] >= 0.20
        and e["query_speedup_over_classical"] >= 1.5
        and e["time_speedup_over_classical"] >= 1.2
    ]

    return {
        "mean_query_speedup": float(np.mean(query_speedups)),
        "mean_time_speedup": float(np.mean(time_speedups)),
        "min_success_probability": float(np.min(success_probs)),
        "credible_advantage_case_count": int(len(credible_cases)),
        "total_case_count": int(len(experiments)),
        "has_credible_quantum_advantage": bool(len(credible_cases) >= max(1, len(experiments) // 2)),
    }


def build_markdown_report(
    scenarios: List[Dict[str, object]],
    json_path: Path,
) -> str:
    lines = [
        "# 多比特量子搜索同构模拟报告",
        "",
        "## 1. 实验目标",
        "",
        "将现有可信量子纠缠计算元件（H 门、相位翻转、扩散算子、退相干噪声）扩展到多比特，",
        "并将 Grover 搜索算法拆解后映射到该模拟量子计算机实例，评估在真实实验参数下是否存在可信量子优越性。",
        "",
        "## 2. 算法拆解到同构实例",
        "",
        "1. 初始化：`|0...0> --H^n--> |s>`",
        "2. Oracle：对目标态执行相位翻转 `|w> -> -|w>`",
        "3. Diffuser：`H^n X^n phase(|0...0>) X^n H^n`",
        "4. 每个逻辑块后施加噪声：幅度阻尼 + 相位阻尼 + 门误差等效去极化",
        "5. 用密度矩阵演化得到 `P_success`，再计算期望查询数与期望时间",
        "",
    ]

    for scenario in scenarios:
        hw = scenario["hardware"]
        experiments = scenario["experiments"]
        conclusion = scenario["conclusion"]

        lines.extend(
            [
                f"## 3. 场景：{hw['name']}",
                "",
                f"- `T1 = {hw['t1_us']:.1f} us`, `T2 = {hw['t2_us']:.1f} us`",
                f"- 单比特门时长：`{hw['single_gate_ns']:.1f} ns`，双比特门时长：`{hw['two_qubit_gate_ns']:.1f} ns`",
                f"- 单比特门保真度：`{hw['f1q']:.5f}`，双比特门保真度：`{hw['f2q']:.5f}`",
                "",
                "| n | N=2^n | k_opt | P_success | Q_classical | Q_quantum(exp) | Query Speedup | Time Speedup |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for e in experiments:
            lines.append(
                f"| {e['n_qubits']} | {e['search_space']} | {e['grover_iterations']} | "
                f"{e['success_probability']:.4f} | {e['classical_expected_queries']:.2f} | "
                f"{e['quantum_expected_queries']:.2f} | {e['query_speedup_over_classical']:.2f}x | "
                f"{e['time_speedup_over_classical']:.2f}x |"
            )

        lines.extend(
            [
                "",
                f"- 平均查询加速：`{conclusion['mean_query_speedup']:.2f}x`",
                f"- 平均时间加速：`{conclusion['mean_time_speedup']:.2f}x`",
                f"- 可信量子优越案例数：`{conclusion['credible_advantage_case_count']}/{conclusion['total_case_count']}`",
                f"- 场景判定：`{conclusion['has_credible_quantum_advantage']}`",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. 总结",
            "",
            "当 n 增大时，Grover 理论查询复杂度从经典 `O(N)` 降为 `O(sqrt(N))`，",
            "本实验展示了当前 NISQ 条件与高保真条件下量子优势是否可维持的边界。",
            "这说明该同构模拟实例不仅复现了量子门结构，还可用于量化真实硬件参数下的可达优越性区间。",
            "",
            "## 5. 附件",
            "",
            f"- 数据文件：`{json_path}`",
        ]
    )

    return "\n".join(lines) + "\n"


def run_experiments() -> List[Dict[str, object]]:
    profiles = [
        HardwareParams(
            name="Current-NISQ",
            t1_us=120.0,
            t2_us=90.0,
            single_gate_ns=35.0,
            two_qubit_gate_ns=280.0,
            readout_ns=450.0,
            reset_ns=600.0,
            f1q=0.9992,
            f2q=0.9910,
        ),
        HardwareParams(
            name="High-Fidelity-SC",
            t1_us=500.0,
            t2_us=350.0,
            single_gate_ns=20.0,
            two_qubit_gate_ns=100.0,
            readout_ns=350.0,
            reset_ns=400.0,
            f1q=0.9998,
            f2q=0.9985,
        ),
        HardwareParams(
            name="Best-Calibrated-SC",
            t1_us=800.0,
            t2_us=600.0,
            single_gate_ns=18.0,
            two_qubit_gate_ns=80.0,
            readout_ns=300.0,
            reset_ns=350.0,
            f1q=0.9999,
            f2q=0.9992,
        ),
    ]

    scenarios = []
    for hw in profiles:
        experiments = []
        for n in [4, 5, 6, 7]:
            dim = 2**n
            # Choose a non-trivial marked state index.
            target = int((3 * dim) // 5)
            result = simulate_noisy_grover(n_qubits=n, target_index=target, hw=hw)
            experiments.append(result)

        scenarios.append(
            {
                "hardware": hw.__dict__,
                "experiments": experiments,
                "conclusion": aggregate_conclusion(experiments),
            }
        )

    return scenarios


def main() -> None:
    scenarios = run_experiments()

    ts = int(time.time())
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"multi_qubit_grover_isomorphic_results_{ts}.json"
    md_path = report_dir / f"多比特量子搜索同构模拟报告_{ts}.md"

    payload = {"scenarios": scenarios}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_text = build_markdown_report(scenarios, json_path)
    md_path.write_text(report_text, encoding="utf-8")

    summary = {s["hardware"]["name"]: s["conclusion"] for s in scenarios}

    print("Multi-qubit Grover isomorphic simulation completed")
    print(f"Conclusion: {summary}")
    print(f"Data: {json_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
