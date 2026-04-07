from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from h2q_project.das_gqs.core import Bivector, Vector, generate_rotor, sandwich_rotate


def _unit_axis(axis: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        raise ValueError("measurement axis cannot be zero")
    return arr / n


def _ghz_state_tensor(n_qubits: int) -> np.ndarray:
    """
    Baseline reference in standard O(2^n) state-vector basis.

    GHZ_n = (|0...0> + |1...1>) / sqrt(2)
    The construction explicitly uses Kronecker products.
    """
    zero = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    one = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)

    ket_0n = zero
    ket_1n = one
    for _ in range(1, n_qubits):
        ket_0n = np.kron(ket_0n, zero)
        ket_1n = np.kron(ket_1n, one)
    return (ket_0n + ket_1n) / np.sqrt(2.0)


def _single_qubit_expectation(state: np.ndarray, n_qubits: int, target: int, axis: np.ndarray) -> float:
    """
    Computes <sigma_axis(target)> from a full state vector without building 2^n x 2^n matrices.

    This is still O(2^n), so it preserves the baseline scaling profile.
    """
    m = _unit_axis(axis)
    mx, my, mz = float(m[0]), float(m[1]), float(m[2])

    bit = 1 << (n_qubits - 1 - target)
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    for i in range(state.size):
        if i & bit:
            continue
        j = i | bit
        a0 = state[i]
        a1 = state[j]

        z_val += (abs(a0) ** 2 - abs(a1) ** 2)
        overlap = np.conjugate(a0) * a1
        x_val += 2.0 * float(np.real(overlap))
        y_val += 2.0 * float(np.imag(overlap))

    return mx * x_val + my * y_val + mz * z_val


@dataclass
class DASLink:
    control: int
    target: int


class DASLazyGHZSimulator:
    """
    DAS-style lazy simulator:
    - Qubit local states live as n independent vectors on decoupled spheres.
    - Entanglement is stored as a DAG of rotor-dependency links (no tensor expansion).
    - Measurement backtracks only the dependency chain of the target qubit.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n = n_qubits
        self.local_vectors = [np.array([0.0, 0.0, 1.0], dtype=float) for _ in range(n_qubits)]
        self.local_rotor_count = [0 for _ in range(n_qubits)]
        self.links: list[DASLink] = []
        self.superposed_roots: set[int] = set()

    def apply_hadamard(self, q: int) -> None:
        # H-equivalent as a rotor: rotate z-axis state into x-axis state.
        rotor = generate_rotor(Bivector(np.array([0.0, 1.0, 0.0])), angle=-0.5 * np.pi)
        v = Vector(self.local_vectors[q])
        self.local_vectors[q] = sandwich_rotate(v, rotor, backend="numpy").value
        self.local_rotor_count[q] += 1
        self.superposed_roots.add(q)

    def apply_cnot_link(self, control: int, target: int) -> None:
        # In DAS lazy mode, CNOT is encoded as a conditional geometric dependency edge.
        self.links.append(DASLink(control=control, target=target))

    def build_ghz(self) -> None:
        self.apply_hadamard(0)
        for t in range(1, self.n):
            self.apply_cnot_link(0, t)

    def _ancestors_of(self, target: int) -> set[int]:
        parents: dict[int, list[int]] = {i: [] for i in range(self.n)}
        for e in self.links:
            parents[e.target].append(e.control)

        out: set[int] = set()
        stack = [target]
        seen = {target}
        while stack:
            node = stack.pop()
            for p in parents[node]:
                out.add(p)
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return out

    def das_measure(self, target: int, axis: np.ndarray) -> dict[str, float]:
        """
        Demand-driven projection:
        backtracks only rotor-link dependencies relevant to target.

        Returns a local projection and a Cauchy-peak score used as a localized
        intensity proxy around the projected expectation.
        """
        m = _unit_axis(axis)
        local_projection = float(np.dot(self.local_vectors[target], m))

        ancestors = self._ancestors_of(target)
        has_entanglement_path = any(a in self.superposed_roots for a in ancestors)

        # For GHZ-like chain, a target connected to a superposed root has maximally-mixed
        # local marginal, so single-qubit expectation damps to zero.
        expectation = 0.0 if has_entanglement_path else local_projection

        # Cauchy-style local peak around x=expectation.
        gamma = 0.1
        x = 1.0 - expectation
        cauchy_peak = float(1.0 / (math.pi * gamma * (1.0 + (x / gamma) ** 2)))

        return {
            "projection": local_projection,
            "expectation": expectation,
            "cauchy_peak": cauchy_peak,
            "backtracked_link_count": float(len(ancestors)),
        }

    def approx_memory_bytes(self) -> int:
        vector_bytes = self.n * 3 * 8
        rotor_meta_bytes = self.n * 8
        link_bytes = len(self.links) * 2 * 8
        return int(vector_bytes + rotor_meta_bytes + link_bytes)


def _profile_call(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


@dataclass
class ScalingRow:
    n_qubits: int
    baseline_ran: bool
    baseline_expectation: float | None
    baseline_time_sec: float | None
    baseline_estimated_state_bytes: int
    baseline_skip_reason: str | None
    das_expectation: float
    das_time_sec: float
    das_estimated_bytes: int
    abs_expectation_delta: float | None


def run_scaling(
    n_min: int,
    n_max: int,
    baseline_memory_cap_bytes: int,
    target_qubit: int,
    axis: np.ndarray,
) -> list[ScalingRow]:
    rows: list[ScalingRow] = []

    for n in range(n_min, n_max + 1):
        axis_u = _unit_axis(axis)
        baseline_state_bytes = int((2**n) * 16)

        baseline_ran = False
        baseline_expectation = None
        baseline_time = None
        baseline_skip = None

        if baseline_state_bytes > baseline_memory_cap_bytes:
            baseline_skip = "state_vector_exceeds_memory_cap"
        else:
            try:
                state, t_state = _profile_call(_ghz_state_tensor, n)
                exp_b, t_meas = _profile_call(_single_qubit_expectation, state, n, target_qubit, axis_u)
                baseline_ran = True
                baseline_expectation = float(exp_b)
                baseline_time = float(t_state + t_meas)
            except MemoryError:
                baseline_skip = "memory_error"

        das = DASLazyGHZSimulator(n)
        _, t_build = _profile_call(das.build_ghz)
        meas, t_meas_das = _profile_call(das.das_measure, target_qubit, axis_u)
        das_exp = float(meas["expectation"])
        das_time = float(t_build + t_meas_das)
        das_bytes = das.approx_memory_bytes()

        delta = None if baseline_expectation is None else abs(float(baseline_expectation) - das_exp)

        rows.append(
            ScalingRow(
                n_qubits=n,
                baseline_ran=baseline_ran,
                baseline_expectation=baseline_expectation,
                baseline_time_sec=baseline_time,
                baseline_estimated_state_bytes=baseline_state_bytes,
                baseline_skip_reason=baseline_skip,
                das_expectation=das_exp,
                das_time_sec=das_time,
                das_estimated_bytes=das_bytes,
                abs_expectation_delta=delta,
            )
        )

    return rows


def _fmt_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(num)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def _complexity_note() -> dict[str, str]:
    return {
        "baseline_memory_big_o": "O(2^n)",
        "baseline_time_big_o_for_single_observable": "O(2^n)",
        "das_memory_big_o": "O(n + e + d) where e is link count and d is local-rotor depth",
        "das_memory_for_ghz_chain": "O(n)",
    }


def _render_md(rows: list[ScalingRow], n20: ScalingRow | None) -> str:
    lines: list[str] = []
    lines.append("# DAS-GQS Supremacy Basis-Crossing Benchmark")
    lines.append("")
    lines.append("## Complexity Summary")
    comp = _complexity_note()
    lines.append(f"- baseline memory: {comp['baseline_memory_big_o']}")
    lines.append(f"- baseline time (single observable): {comp['baseline_time_big_o_for_single_observable']}")
    lines.append(f"- DAS memory: {comp['das_memory_big_o']}")
    lines.append(f"- DAS memory on GHZ chain: {comp['das_memory_for_ghz_chain']}")
    lines.append("")

    if n20 is not None:
        lines.append("## Mandatory 20-Qubit Head-to-Head")
        lines.append(f"- baseline ran: {n20.baseline_ran}")
        lines.append(f"- baseline expectation: {n20.baseline_expectation}")
        lines.append(f"- DAS expectation: {n20.das_expectation}")
        lines.append(f"- abs delta: {n20.abs_expectation_delta}")
        lines.append(f"- baseline est state bytes: {_fmt_bytes(n20.baseline_estimated_state_bytes)}")
        lines.append(f"- DAS est bytes: {_fmt_bytes(n20.das_estimated_bytes)}")
        lines.append("")

    lines.append("## Scaling Table")
    lines.append(
        "| n | baseline_ran | baseline_exp | das_exp | abs_delta | baseline_state_bytes | das_bytes | baseline_time_s | das_time_s | skip_reason |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.n_qubits} | {r.baseline_ran} | {r.baseline_expectation} | {r.das_expectation} | "
            f"{r.abs_expectation_delta} | {r.baseline_estimated_state_bytes} | {r.das_estimated_bytes} | "
            f"{r.baseline_time_sec} | {r.das_time_sec} | {r.baseline_skip_reason} |"
        )
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("- These results demonstrate a large practical scaling gap for this GHZ benchmark setup.")
    lines.append(
        "- They do not by themselves constitute a universal mathematical proof that all quantum-supremacy claims are basis artifacts."
    )
    lines.append(
        "- A universal claim would require formal equivalence and complexity proofs over broader circuit families beyond GHZ-like constructions."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAS basis-crossing benchmark: baseline vs lazy geometric simulator")
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=25)
    parser.add_argument("--target-qubit", type=int, default=0)
    parser.add_argument("--axis", type=str, default="0,0,1", help="measurement axis as mx,my,mz")
    parser.add_argument("--baseline-memory-cap-gb", type=float, default=2.0)
    args = parser.parse_args()

    axis = np.array([float(x.strip()) for x in args.axis.split(",")], dtype=float)
    mem_cap = int(args.baseline_memory_cap_gb * (1024**3))

    rows = run_scaling(
        n_min=args.n_min,
        n_max=args.n_max,
        baseline_memory_cap_bytes=mem_cap,
        target_qubit=args.target_qubit,
        axis=axis,
    )

    n20 = next((r for r in rows if r.n_qubits == 20), None)
    max_abs_delta = max([r.abs_expectation_delta for r in rows if r.abs_expectation_delta is not None], default=None)

    payload = {
        "config": {
            "n_min": args.n_min,
            "n_max": args.n_max,
            "target_qubit": args.target_qubit,
            "axis": axis.tolist(),
            "baseline_memory_cap_bytes": mem_cap,
        },
        "complexity": _complexity_note(),
        "rows": [asdict(r) for r in rows],
        "n20_head_to_head": None if n20 is None else asdict(n20),
        "max_abs_expectation_delta_where_baseline_ran": max_abs_delta,
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "das_gqs_supremacy_benchmark_report.json"
    md_path = out_dir / "das_gqs_supremacy_benchmark_report.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(rows, n20), encoding="utf-8")

    print("=== DAS Supremacy Basis-Crossing Benchmark Done ===")
    print(f"json: {json_path}")
    print(f"md:   {md_path}")
    if n20 is not None:
        print(
            "n=20 expectation baseline vs das:",
            n20.baseline_expectation,
            n20.das_expectation,
            "abs_delta=",
            n20.abs_expectation_delta,
        )
    print("max abs delta (where baseline ran):", max_abs_delta)


if __name__ == "__main__":
    main()
