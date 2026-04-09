from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


GateName = Literal["H", "S", "SDG", "CNOT"]


@dataclass(frozen=True)
class GateOp:
    name: GateName
    q: int
    q2: int | None = None


@dataclass(frozen=True)
class ObservableSpec:
    kind: Literal["single", "pair"]
    pauli: str
    q: int
    q2: int | None = None


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _complexity_state_bytes(n: int) -> int:
    return int((2**n) * 16)


def generate_rcs_subset(n: int, depth: int, seed: int, ent_prob: float = 0.65) -> list[GateOp]:
    """
    Random Clifford-circuit subset (RCS subset):
    - local: H, S, Sdg
    - entangler: nearest-neighbor CNOTs with random stagger pattern
    """
    rng = np.random.default_rng(seed)
    ops: list[GateOp] = []

    local_set: list[GateName] = ["H", "S", "SDG"]
    for layer in range(depth):
        for q in range(n):
            if rng.random() < 0.85:
                g = local_set[int(rng.integers(0, len(local_set)))]
                ops.append(GateOp(name=g, q=q))

        if n >= 2 and rng.random() < ent_prob:
            start = int(layer % 2)
            for c in range(start, n - 1, 2):
                ops.append(GateOp(name="CNOT", q=c, q2=c + 1))
    return ops


def _apply_single_qubit_gate(state: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    bit = n - 1 - q
    dim = 1 << n
    mask = 1 << bit
    out = state.copy()
    idx = np.arange(dim)
    low = idx[(idx & mask) == 0]
    high = low | mask
    a0 = state[low]
    a1 = state[high]
    out[low] = U[0, 0] * a0 + U[0, 1] * a1
    out[high] = U[1, 0] * a0 + U[1, 1] * a1
    return out


def _apply_cnot(state: np.ndarray, n: int, c: int, t: int) -> np.ndarray:
    bit_c = 1 << (n - 1 - c)
    bit_t = 1 << (n - 1 - t)
    idx = np.arange(state.size)
    cond = ((idx & bit_c) != 0).astype(np.int64)
    j = idx ^ (cond * bit_t)
    out = np.empty_like(state)
    out[j] = state[idx]
    return out


def simulate_baseline_state(n: int, circuit: list[GateOp]) -> np.ndarray:
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0 + 0.0j

    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    for op in circuit:
        if op.name == "H":
            state = _apply_single_qubit_gate(state, n, op.q, H)
        elif op.name == "S":
            state = _apply_single_qubit_gate(state, n, op.q, S)
        elif op.name == "SDG":
            state = _apply_single_qubit_gate(state, n, op.q, SDG)
        elif op.name == "CNOT":
            assert op.q2 is not None
            state = _apply_cnot(state, n, op.q, op.q2)
        else:
            raise ValueError(f"Unsupported gate: {op.name}")
    return state


def _pauli_matrix(p: str) -> np.ndarray:
    if p == "I":
        return np.eye(2, dtype=np.complex128)
    if p == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if p == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if p == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(p)


def _apply_pauli_to_state(state: np.ndarray, n: int, q: int, p: str) -> np.ndarray:
    return _apply_single_qubit_gate(state, n, q, _pauli_matrix(p))


def baseline_expectation(state: np.ndarray, n: int, obs: ObservableSpec) -> float:
    if obs.kind == "single":
        transformed = _apply_pauli_to_state(state, n, obs.q, obs.pauli)
    else:
        assert obs.q2 is not None
        transformed = _apply_pauli_to_state(state, n, obs.q, obs.pauli[0])
        transformed = _apply_pauli_to_state(transformed, n, obs.q2, obs.pauli[1])
    return float(np.real(np.vdot(state, transformed)))


def _build_local_back_maps():
    I = _pauli_matrix("I")
    X = _pauli_matrix("X")
    Y = _pauli_matrix("Y")
    Z = _pauli_matrix("Z")
    basis = {"I": I, "X": X, "Y": Y, "Z": Z}

    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    gates = {"H": H, "S": S, "SDG": SDG}

    maps: dict[str, dict[str, tuple[str, int]]] = {}
    for gname, U in gates.items():
        m: dict[str, tuple[str, int]] = {}
        for name, P in basis.items():
            Q = U.conj().T @ P @ U  # Heisenberg back-propagation: U^dagger O U
            found = None
            for tgt_name, T in basis.items():
                if np.allclose(Q, T):
                    found = (tgt_name, 1)
                    break
                if np.allclose(Q, -T):
                    found = (tgt_name, -1)
                    break
            if found is None:
                raise RuntimeError(f"Cannot map {gname} on {name}")
            m[name] = found
        maps[gname] = m
    return maps


def _build_cnot_back_map():
    I = _pauli_matrix("I")
    X = _pauli_matrix("X")
    Y = _pauli_matrix("Y")
    Z = _pauli_matrix("Z")
    basis = {"I": I, "X": X, "Y": Y, "Z": Z}
    labels = ["I", "X", "Y", "Z"]

    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )

    m: dict[tuple[str, str], tuple[str, str, int]] = {}
    for a in labels:
        for b in labels:
            P = np.kron(basis[a], basis[b])
            Q = CNOT.conj().T @ P @ CNOT
            found = None
            for aa in labels:
                for bb in labels:
                    T = np.kron(basis[aa], basis[bb])
                    if np.allclose(Q, T):
                        found = (aa, bb, 1)
                        break
                    if np.allclose(Q, -T):
                        found = (aa, bb, -1)
                        break
                if found is not None:
                    break
            if found is None:
                raise RuntimeError(f"Cannot map CNOT on {a},{b}")
            m[(a, b)] = found
    return m


LOCAL_BACK_MAP = _build_local_back_maps()
CNOT_BACK_MAP = _build_cnot_back_map()


class DASHeisenbergLazySimulator:
    """
    Tensor-free DAS lazy evaluator for this RCS subset:
    - Stores only gate history (DAG/path instructions).
    - Evaluates observables by reverse-path projection in Heisenberg picture.
    - Never materializes global 2^n state.
    """

    def __init__(self, n: int, circuit: list[GateOp]) -> None:
        self.n = n
        self.circuit = circuit

    def expectation(self, obs: ObservableSpec) -> float:
        paulis = ["I"] * self.n
        sign = 1

        if obs.kind == "single":
            paulis[obs.q] = obs.pauli
        else:
            assert obs.q2 is not None
            paulis[obs.q] = obs.pauli[0]
            paulis[obs.q2] = obs.pauli[1]

        for op in reversed(self.circuit):
            if op.name in ("H", "S", "SDG"):
                p = paulis[op.q]
                p2, s = LOCAL_BACK_MAP[op.name][p]
                paulis[op.q] = p2
                sign *= s
            elif op.name == "CNOT":
                assert op.q2 is not None
                a, b = paulis[op.q], paulis[op.q2]
                aa, bb, s = CNOT_BACK_MAP[(a, b)]
                paulis[op.q], paulis[op.q2] = aa, bb
                sign *= s

        # Evaluate on |0...0>: only I/Z strings survive.
        for p in paulis:
            if p in ("X", "Y"):
                return 0.0
        return float(sign)

    def approx_memory_bytes(self) -> int:
        # O(number_of_ops) metadata, no global tensor.
        return int(len(self.circuit) * 24 + self.n * 16)


def build_observables(n: int) -> list[ObservableSpec]:
    obs: list[ObservableSpec] = []
    singles = min(3, n)
    for q in range(singles):
        for p in ["X", "Y", "Z"]:
            obs.append(ObservableSpec(kind="single", pauli=p, q=q))

    for q in range(min(2, n - 1)):
        for p in ["XX", "YY", "ZZ"]:
            obs.append(ObservableSpec(kind="pair", pauli=p, q=q, q2=q + 1))
    return obs


@dataclass
class NLevelStats:
    n_qubits: int
    depth: int
    sample_count: int
    mean_abs_error: float
    ci95_low_mae: float
    ci95_high_mae: float
    rmse: float
    max_abs_error: float
    equivalence_margin: float
    alpha: float
    tost_p1: float
    tost_p2: float
    equivalence_pass: bool
    baseline_time_mean_sec: float
    das_time_mean_sec: float
    baseline_state_bytes: int
    das_bytes_mean: float


@dataclass
class SeedRobustRow:
    seed: int
    sample_count: int
    mean_abs_error: float
    rmse: float
    max_abs_error: float
    equivalence_pass: bool


@dataclass
class RCSStatsReport:
    n_list: list[int]
    seeds: list[int]
    depth_factor: int
    equivalence_margin: float
    alpha: float
    per_n_stats: list[NLevelStats]
    per_seed_summary: list[SeedRobustRow]


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    n = int(values.size)
    if n <= 1:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half


def _tost_equivalence(diffs: np.ndarray, eps: float, alpha: float) -> tuple[float, float, bool]:
    n = int(diffs.size)
    if n <= 1:
        passed = bool(abs(float(np.mean(diffs))) < eps)
        return 1.0, 1.0, passed

    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0

    if se < 1e-15:
        passed = bool(abs(mean) < eps)
        return 0.0 if passed else 1.0, 0.0 if passed else 1.0, passed

    z1 = (mean + eps) / se
    z2 = (mean - eps) / se
    p1 = 1.0 - _normal_cdf(z1)  # H01: mean <= -eps
    p2 = _normal_cdf(z2)        # H02: mean >=  eps
    passed = (p1 < alpha) and (p2 < alpha)
    return float(p1), float(p2), bool(passed)


def run_rcs_stats(
    n_list: list[int],
    seeds: list[int],
    depth_factor: int,
    equivalence_margin: float,
    alpha: float,
) -> RCSStatsReport:
    per_n: list[NLevelStats] = []
    seed_errors: dict[int, list[float]] = {s: [] for s in seeds}

    for n in n_list:
        depth = depth_factor * n
        all_deltas: list[float] = []
        all_abs: list[float] = []
        all_sq: list[float] = []
        bt: list[float] = []
        dt: list[float] = []
        db: list[float] = []

        observables = build_observables(n)

        for s in seeds:
            circuit = generate_rcs_subset(n=n, depth=depth, seed=s)

            t0 = time.perf_counter()
            state = simulate_baseline_state(n, circuit)
            t1 = time.perf_counter()
            das = DASHeisenbergLazySimulator(n, circuit)
            t2 = time.perf_counter()

            bt.append(float(t1 - t0))
            dt.append(float(t2 - t1))
            db.append(float(das.approx_memory_bytes()))

            for obs in observables:
                b = baseline_expectation(state, n, obs)
                d = das.expectation(obs)
                delta = float(d - b)
                all_deltas.append(delta)
                all_abs.append(abs(delta))
                all_sq.append(delta * delta)
                seed_errors[s].append(abs(delta))

        arr_d = np.asarray(all_deltas, dtype=float)
        arr_abs = np.asarray(all_abs, dtype=float)
        arr_sq = np.asarray(all_sq, dtype=float)

        mean_abs, ci_low, ci_high = _mean_ci95(arr_abs)
        rmse = float(np.sqrt(np.mean(arr_sq)))
        p1, p2, eq_pass = _tost_equivalence(arr_d, eps=equivalence_margin, alpha=alpha)

        per_n.append(
            NLevelStats(
                n_qubits=n,
                depth=depth,
                sample_count=int(arr_d.size),
                mean_abs_error=mean_abs,
                ci95_low_mae=ci_low,
                ci95_high_mae=ci_high,
                rmse=rmse,
                max_abs_error=float(np.max(arr_abs)) if arr_abs.size > 0 else 0.0,
                equivalence_margin=equivalence_margin,
                alpha=alpha,
                tost_p1=p1,
                tost_p2=p2,
                equivalence_pass=eq_pass,
                baseline_time_mean_sec=float(np.mean(np.asarray(bt, dtype=float))),
                das_time_mean_sec=float(np.mean(np.asarray(dt, dtype=float))),
                baseline_state_bytes=_complexity_state_bytes(n),
                das_bytes_mean=float(np.mean(np.asarray(db, dtype=float))),
            )
        )

    per_seed: list[SeedRobustRow] = []
    for s in seeds:
        vals = np.asarray(seed_errors[s], dtype=float)
        mae = float(np.mean(vals)) if vals.size else 0.0
        rmse = float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0
        maxe = float(np.max(vals)) if vals.size else 0.0
        per_seed.append(
            SeedRobustRow(
                seed=s,
                sample_count=int(vals.size),
                mean_abs_error=mae,
                rmse=rmse,
                max_abs_error=maxe,
                equivalence_pass=maxe <= equivalence_margin,
            )
        )

    return RCSStatsReport(
        n_list=n_list,
        seeds=seeds,
        depth_factor=depth_factor,
        equivalence_margin=equivalence_margin,
        alpha=alpha,
        per_n_stats=per_n,
        per_seed_summary=per_seed,
    )


def _plot_confidence_band(path: Path, report: RCSStatsReport) -> None:
    ns = np.asarray([r.n_qubits for r in report.per_n_stats], dtype=float)
    mae = np.asarray([r.mean_abs_error for r in report.per_n_stats], dtype=float)
    lo = np.asarray([r.ci95_low_mae for r in report.per_n_stats], dtype=float)
    hi = np.asarray([r.ci95_high_mae for r in report.per_n_stats], dtype=float)
    rmse = np.asarray([r.rmse for r in report.per_n_stats], dtype=float)

    plt.figure(figsize=(8.4, 5.4), dpi=140)
    plt.plot(ns, mae, marker="o", linewidth=2.0, label="Mean |delta| (MAE)")
    plt.fill_between(ns, lo, hi, alpha=0.20, label="95% CI band (MAE)")
    plt.plot(ns, rmse, marker="s", linewidth=1.8, label="RMSE")
    plt.axhline(report.equivalence_margin, linestyle="--", linewidth=1.5, label=f"Equivalence margin eps={report.equivalence_margin:g}")

    plt.yscale("log")
    plt.xlabel("Qubit count n")
    plt.ylabel("Error (log scale)")
    plt.title("RCS Subset: DAS vs Baseline Multi-Observable Consistency")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _render_md(report: RCSStatsReport) -> str:
    lines: list[str] = []
    lines.append("# DAS-GQS RCS Subset Statistical Benchmark")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- n list: {report.n_list}")
    lines.append(f"- seeds: {report.seeds}")
    lines.append(f"- depth factor: {report.depth_factor}")
    lines.append(f"- equivalence margin: {report.equivalence_margin}")
    lines.append(f"- alpha: {report.alpha}")
    lines.append("")
    lines.append("## Per-n Statistical Summary")
    lines.append("| n | depth | samples | MAE | 95% CI low | 95% CI high | RMSE | max | p1 | p2 | TOST pass |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in report.per_n_stats:
        lines.append(
            f"| {r.n_qubits} | {r.depth} | {r.sample_count} | {r.mean_abs_error:.3e} | {r.ci95_low_mae:.3e} | "
            f"{r.ci95_high_mae:.3e} | {r.rmse:.3e} | {r.max_abs_error:.3e} | {r.tost_p1:.3e} | {r.tost_p2:.3e} | {r.equivalence_pass} |"
        )
    lines.append("")
    lines.append("## Cross-seed Robustness")
    lines.append("| seed | samples | MAE | RMSE | max | pass(max<=eps) |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for r in report.per_seed_summary:
        lines.append(
            f"| {r.seed} | {r.sample_count} | {r.mean_abs_error:.3e} | {r.rmse:.3e} | {r.max_abs_error:.3e} | {r.equivalence_pass} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, report: RCSStatsReport) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "depth",
                "samples",
                "mae",
                "ci95_low",
                "ci95_high",
                "rmse",
                "max_abs_error",
                "tost_p1",
                "tost_p2",
                "equivalence_pass",
                "baseline_time_mean_sec",
                "das_time_mean_sec",
                "baseline_state_bytes",
                "das_bytes_mean",
            ]
        )
        for r in report.per_n_stats:
            w.writerow(
                [
                    r.n_qubits,
                    r.depth,
                    r.sample_count,
                    r.mean_abs_error,
                    r.ci95_low_mae,
                    r.ci95_high_mae,
                    r.rmse,
                    r.max_abs_error,
                    r.tost_p1,
                    r.tost_p2,
                    r.equivalence_pass,
                    r.baseline_time_mean_sec,
                    r.das_time_mean_sec,
                    r.baseline_state_bytes,
                    r.das_bytes_mean,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="RCS subset statistical consistency benchmark")
    parser.add_argument("--n-list", type=str, default="6,8,10,12,14")
    parser.add_argument("--seed-list", type=str, default="11,17,23,31,43,59,71,83")
    parser.add_argument("--depth-factor", type=int, default=3)
    parser.add_argument("--equiv-margin", type=float, default=1e-9)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]

    report = run_rcs_stats(
        n_list=n_list,
        seeds=seeds,
        depth_factor=args.depth_factor,
        equivalence_margin=args.equiv_margin,
        alpha=args.alpha,
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "das_gqs_rcs_subset_stats.json"
    md_path = out_dir / "das_gqs_rcs_subset_stats.md"
    csv_path = out_dir / "das_gqs_rcs_subset_stats.csv"
    fig_path = out_dir / "das_gqs_rcs_subset_stats_band.png"

    json_path.write_text(json.dumps(asdict(report), ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    _write_csv(csv_path, report)
    _plot_confidence_band(fig_path, report)

    total_pass = all(r.equivalence_pass for r in report.per_n_stats)
    print("=== DAS-GQS RCS subset statistical benchmark done ===")
    print(f"json: {json_path}")
    print(f"md:   {md_path}")
    print(f"csv:  {csv_path}")
    print(f"fig:  {fig_path}")
    print("all_n_tost_pass:", total_pass)


if __name__ == "__main__":
    main()
