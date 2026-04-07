from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

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
from .core import Vector, geometric_correlation


def axis_from_degree(deg: float) -> Vector:
    rad = np.deg2rad(deg)
    return Vector(np.array([np.cos(rad), np.sin(rad), 0.0]))


@dataclass
class EstimateWithCI:
    mean: float
    ci_low: float
    ci_high: float
    std_err: float
    n: int


@dataclass
class ChshBatchEstimate:
    E_ab: EstimateWithCI
    E_ab_prime: EstimateWithCI
    E_a_prime_b: EstimateWithCI
    E_a_prime_b_prime: EstimateWithCI
    S: EstimateWithCI
    abs_S: float
    hardware_profile: dict[str, object]
    compute_plan: dict[str, object]


@dataclass
class NoiseScenario:
    name: str
    axis_jitter_deg: float
    outcome_flip_prob: float


@dataclass
class NoiseRobustnessRow:
    scenario: str
    axis_jitter_deg: float
    outcome_flip_prob: float
    S_mean: float
    S_ci_low: float
    S_ci_high: float
    abs_S: float
    violates_classical_limit_with_95ci: bool


def resolve_batch_sampling_compute_plan(
    n_pairs: int,
    compute_plan_cache: Path,
    refresh_compute_plan_cache: bool = False,
    autotune_threads: bool = False,
    thread_candidates: str | None = None,
    matmul_size: int = 896,
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
        namespace="batch_sampling",
        hardware=hw,
        params={
            "n_pairs": int(n_pairs),
            "autotune_threads": bool(autotune_threads),
            "thread_candidates": cands,
            "probe": "matmul",
            "matmul_size": int(matmul_size),
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
                matmul_size=matmul_size,
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


def _estimate_with_ci(samples: np.ndarray, z: float = 1.96) -> EstimateWithCI:
    n = int(samples.size)
    mean = float(np.mean(samples))
    if n <= 1:
        se = 0.0
    else:
        se = float(np.std(samples, ddof=1) / np.sqrt(n))
    half = z * se
    return EstimateWithCI(
        mean=mean,
        ci_low=mean - half,
        ci_high=mean + half,
        std_err=se,
        n=n,
    )


def _draw_correlated_outcomes(E: float, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    p_same = float(np.clip((1.0 + E) * 0.5, 0.0, 1.0))
    a = rng.choice(np.array([-1, 1], dtype=int), size=n_pairs)
    same = rng.random(size=n_pairs) < p_same
    b = np.where(same, a, -a)
    return a * b


def _apply_outcome_noise(products: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    if flip_prob <= 0.0:
        return products
    flip_a = rng.random(size=products.size) < flip_prob
    flip_b = rng.random(size=products.size) < flip_prob
    sign = np.where(flip_a, -1, 1) * np.where(flip_b, -1, 1)
    return products * sign


def _noisy_axis(base_deg: float, jitter_deg: float, rng: np.random.Generator) -> Vector:
    if jitter_deg <= 0.0:
        return axis_from_degree(base_deg)
    noisy = base_deg + float(rng.normal(loc=0.0, scale=jitter_deg))
    return axis_from_degree(noisy)


def estimate_chsh_batch(
    n_pairs: int = 20000,
    seed: int = 7,
    axis_jitter_deg: float = 0.0,
    outcome_flip_prob: float = 0.0,
    hardware_profile: dict[str, object] | None = None,
    compute_plan: dict[str, object] | None = None,
) -> ChshBatchEstimate:
    rng = np.random.default_rng(seed)

    A, A_p = 0.0, 90.0
    B, B_p = 45.0, 135.0

    def estimate_pair(a_deg: float, b_deg: float) -> EstimateWithCI:
        a_axis = _noisy_axis(a_deg, axis_jitter_deg, rng)
        b_axis = _noisy_axis(b_deg, axis_jitter_deg, rng)
        E = geometric_correlation(a_axis, b_axis)
        products = _draw_correlated_outcomes(E, n_pairs=n_pairs, rng=rng)
        noisy_products = _apply_outcome_noise(products, flip_prob=outcome_flip_prob, rng=rng)
        return _estimate_with_ci(noisy_products)

    e_ab = estimate_pair(A, B)
    e_ab_p = estimate_pair(A, B_p)
    e_a_p_b = estimate_pair(A_p, B)
    e_a_p_b_p = estimate_pair(A_p, B_p)

    s_mean = e_ab.mean - e_ab_p.mean + e_a_p_b.mean + e_a_p_b_p.mean
    s_se = math.sqrt(
        e_ab.std_err**2 + e_ab_p.std_err**2 + e_a_p_b.std_err**2 + e_a_p_b_p.std_err**2
    )
    s_half = 1.96 * s_se
    s_est = EstimateWithCI(
        mean=s_mean,
        ci_low=s_mean - s_half,
        ci_high=s_mean + s_half,
        std_err=s_se,
        n=n_pairs,
    )

    return ChshBatchEstimate(
        E_ab=e_ab,
        E_ab_prime=e_ab_p,
        E_a_prime_b=e_a_p_b,
        E_a_prime_b_prime=e_a_p_b_p,
        S=s_est,
        abs_S=abs(s_mean),
        hardware_profile=hardware_profile or {},
        compute_plan=compute_plan or {},
    )


def noise_robustness_report(
    n_pairs: int = 20000,
    seed: int = 11,
    scenarios: Iterable[NoiseScenario] | None = None,
    hardware_profile: dict[str, object] | None = None,
    compute_plan: dict[str, object] | None = None,
) -> list[NoiseRobustnessRow]:
    if scenarios is None:
        scenarios = [
            NoiseScenario("ideal", axis_jitter_deg=0.0, outcome_flip_prob=0.0),
            NoiseScenario("mild", axis_jitter_deg=1.0, outcome_flip_prob=0.02),
            NoiseScenario("moderate", axis_jitter_deg=3.0, outcome_flip_prob=0.05),
            NoiseScenario("strong", axis_jitter_deg=5.0, outcome_flip_prob=0.10),
        ]

    rows: list[NoiseRobustnessRow] = []
    for i, sc in enumerate(scenarios):
        est = estimate_chsh_batch(
            n_pairs=n_pairs,
            seed=seed + i,
            axis_jitter_deg=sc.axis_jitter_deg,
            outcome_flip_prob=sc.outcome_flip_prob,
            hardware_profile=hardware_profile,
            compute_plan=compute_plan,
        )
        abs_ci_low = min(abs(est.S.ci_low), abs(est.S.ci_high)) if est.S.ci_low * est.S.ci_high > 0 else 0.0
        rows.append(
            NoiseRobustnessRow(
                scenario=sc.name,
                axis_jitter_deg=sc.axis_jitter_deg,
                outcome_flip_prob=sc.outcome_flip_prob,
                S_mean=est.S.mean,
                S_ci_low=est.S.ci_low,
                S_ci_high=est.S.ci_high,
                abs_S=est.abs_S,
                violates_classical_limit_with_95ci=abs_ci_low > 2.0,
            )
        )
    return rows


def as_serializable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [as_serializable(x) for x in obj]
    return obj
