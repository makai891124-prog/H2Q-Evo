from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from h2q_project.das_gqs.core import (
        Bivector,
        EntangledPair,
        G3,
        Scalar,
        Vector,
        assert_reversible,
        generate_rotor,
        geometric_correlation,
        measure_projection,
        sandwich_rotate,
    )
    from h2q_project.das_gqs.autocompute import (
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
except ModuleNotFoundError:
    from core import (  # type: ignore
        Bivector,
        EntangledPair,
        G3,
        Scalar,
        Vector,
        assert_reversible,
        generate_rotor,
        geometric_correlation,
        measure_projection,
        sandwich_rotate,
    )
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


def axis_from_degree(deg: float) -> Vector:
    rad = np.deg2rad(deg)
    return Vector(np.array([np.cos(rad), np.sin(rad), 0.0]))


@dataclass
class ChshReport:
    E_ab: float
    E_ab_prime: float
    E_a_prime_b: float
    E_a_prime_b_prime: float
    S: float
    abs_S: float
    tsirelson_target: float
    tsirelson_error: float


def evaluate_chsh() -> ChshReport:
    # Required angle set: 0, 45, 90, 135 degrees.
    A = axis_from_degree(0.0)
    A_prime = axis_from_degree(90.0)
    B = axis_from_degree(45.0)
    B_prime = axis_from_degree(135.0)

    E_ab = geometric_correlation(A, B)
    E_ab_prime = geometric_correlation(A, B_prime)
    E_a_prime_b = geometric_correlation(A_prime, B)
    E_a_prime_b_prime = geometric_correlation(A_prime, B_prime)

    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
    abs_S = abs(S)
    target = 2.0 * math.sqrt(2.0)
    return ChshReport(
        E_ab=E_ab,
        E_ab_prime=E_ab_prime,
        E_a_prime_b=E_a_prime_b,
        E_a_prime_b_prime=E_a_prime_b_prime,
        S=S,
        abs_S=abs_S,
        tsirelson_target=target,
        tsirelson_error=abs(abs_S - target),
    )


def memory_comparison(n_qubits: int = 2) -> dict:
    # Traditional complex matrix representation.
    dim = 2**n_qubits
    bytes_complex128 = 16
    tensor_gate_bytes = dim * dim * bytes_complex128

    # DAS-GQS geometric payload for a Bell-like pair:
    # 2 vectors (A/B) + 1 rotor (1 scalar + 3 bivector comps), all float64.
    bytes_float64 = 8
    das_bytes = (2 * 3 + 4) * bytes_float64

    return {
        "n_qubits": n_qubits,
        "traditional_gate_matrix_bytes": tensor_gate_bytes,
        "das_geometric_state_bytes": das_bytes,
        "compression_ratio_traditional_over_das": tensor_gate_bytes / das_bytes,
    }


def _resolve_compute_plan(
    compute_plan_cache: Path,
    refresh_compute_plan_cache: bool,
    autotune_threads: bool,
    thread_candidates: str | None,
    seed: int,
) -> tuple[dict, dict]:
    _ = seed
    hw = detect_hardware_profile()
    cands = (
        parse_thread_candidates(thread_candidates, cpu_count=max(1, hw.cpu_count))
        if autotune_threads
        else default_thread_candidates(max(1, hw.cpu_count))
    )
    if not cands:
        cands = [max(1, torch.get_num_threads())]

    cache_key = make_cache_key(
        namespace="chsh_validation",
        hardware=hw,
        params={
            "autotune_threads": bool(autotune_threads),
            "thread_candidates": cands,
            "probe": "matmul",
        },
    )
    cache = load_compute_plan_cache(compute_plan_cache)
    cache_hit = False
    probe_timings: dict[str, float] | None = None

    cached = None if refresh_compute_plan_cache else cache_lookup(cache, cache_key)
    if cached is not None:
        cache_hit = True
        selected_threads = int(cached.get("selected_torch_threads", max(1, torch.get_num_threads())))
        probe_timings = None
    else:
        if autotune_threads:
            selected_threads, probe_timings = autotune_threads_with_matmul(
                thread_candidates=cands,
                matmul_size=896,
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

    torch.set_num_threads(max(1, selected_threads))
    plan = {
        "selected_torch_threads": int(torch.get_num_threads()),
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "autotune_timings_ms": probe_timings,
    }
    return profile_as_dict(hw), plan


def run_demo(hardware_profile: dict | None = None, compute_plan: dict | None = None) -> dict:
    print("=== DAS-GQS: Geometric Quantum Simulation Demo ===")

    # Phase 1: Core algebraic entities.
    scalar = Scalar(1.0)
    vector = Vector(np.array([1.0, 0.0, 0.0]))
    bivector = Bivector(np.array([0.0, 0.0, 1.0]))  # e12 plane
    print("[Phase 1] Grades initialized:")
    print(f"  scalar={scalar.value}, vector={vector.value}, bivector={bivector.value}, I={G3.I.value}")

    # Phase 2: Rotor time-slice evolution and reversibility.
    rotor = generate_rotor(bivector, angle=np.pi / 3.0)
    evolved = sandwich_rotate(vector, rotor)
    rev_metrics = assert_reversible(vector, rotor)
    print("[Phase 2] Rotor evolution:")
    print(f"  v_old={vector.value}")
    print(f"  v_new={evolved.value}")
    print(f"  reversibility={rev_metrics}")

    # Phase 3: Geometric entanglement preparation.
    pair = EntangledPair(a_state=G3.e3)
    pair.apply_global_correlated_rotor(rotor)
    a_state, b_state = pair.poles()
    print("[Phase 3] Entanglement lock (antipodal poles):")
    print(f"  A={a_state.value}")
    print(f"  B={b_state.value}")
    print(f"  A+B (should ~0)={(a_state.value + b_state.value)}")

    # Phase 4: Projection and masking measurement.
    m_axis = axis_from_degree(45.0)
    p, out, collapsed = measure_projection(a_state, m_axis)
    print("[Phase 4] Projection measurement:")
    print(f"  projection P=v·m={p:.6f}, outcome={out}, collapsed={collapsed.value}")

    report = evaluate_chsh()
    print("[Validation] CHSH with pure geometric correlation E(a,b)=-a·b")
    for k, v in asdict(report).items():
        print(f"  {k}: {v}")

    resource = memory_comparison(n_qubits=2)
    print("[Validation] Memory structure comparison")
    for k, v in resource.items():
        print(f"  {k}: {v}")

    payload = {
        "chsh": asdict(report),
        "memory": resource,
        "hardware_profile": hardware_profile or {},
        "compute_plan": compute_plan or {},
    }
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS-GQS CHSH validation with autocompute integration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--autotune-threads",
        action="store_true",
        help="Autotune CPU thread count and cache the selected plan.",
    )
    p.add_argument(
        "--thread-candidates",
        type=str,
        default="",
        help="Comma-separated thread candidates, e.g. 1,2,4,6,8,10",
    )
    p.add_argument(
        "--compute-plan-cache",
        type=Path,
        default=Path("reports/compute_plan_cache.json"),
        help="Offline compute plan cache file shared across DAS scripts.",
    )
    p.add_argument(
        "--refresh-compute-plan-cache",
        action="store_true",
        help="Ignore cache and re-probe thread plan.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/das_gqs_chsh_validation.json"),
        help="Write full validation payload to JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hardware, plan = _resolve_compute_plan(
        compute_plan_cache=args.compute_plan_cache,
        refresh_compute_plan_cache=args.refresh_compute_plan_cache,
        autotune_threads=args.autotune_threads,
        thread_candidates=args.thread_candidates,
        seed=args.seed,
    )
    payload = run_demo(hardware_profile=hardware, compute_plan=plan)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[CHSH] report json: {args.output_json}")
    print(
        "[CHSH] compute plan: "
        f"threads={payload['compute_plan'].get('selected_torch_threads')}, "
        f"cache_hit={payload['compute_plan'].get('cache_hit')}"
    )


if __name__ == "__main__":
    main()
