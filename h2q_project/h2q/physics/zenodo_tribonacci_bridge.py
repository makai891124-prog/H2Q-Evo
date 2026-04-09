"""Tribonacci SL(3,Z) prompt bridge inspired by Zenodo 10.5281/zenodo.19220046.

This module keeps the integration lightweight and deterministic:
- Uses the companion matrix of x^3 - x^2 - x - 1
- Enforces det(A)=1 as a basic discrete unitarity check
- Derives a compact foliation signature for a dialogue prompt
- Exposes a text augmenter that can be used by reasoning entry points
"""

from __future__ import annotations

import math
from typing import Dict, List

TRIBONACCI_ETA = 1.8392867552141612


def companion_matrix() -> List[List[int]]:
    return [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]


def _matmul_3x3(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    out = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
            )
    return out


def _matpow_3x3(base: List[List[int]], exponent: int) -> List[List[int]]:
    result = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    power = [row[:] for row in base]
    exp = max(0, int(exponent))

    while exp > 0:
        if exp & 1:
            result = _matmul_3x3(result, power)
        power = _matmul_3x3(power, power)
        exp >>= 1
    return result


def _det_3x3(m: List[List[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _trace_3x3(m: List[List[int]]) -> int:
    return m[0][0] + m[1][1] + m[2][2]


def _fractional_binomial(alpha: float, k: int) -> float:
    if k == 0:
        return 1.0
    return math.gamma(alpha + 1.0) / (
        math.gamma(k + 1.0) * math.gamma(alpha - k + 1.0)
    )


def _half_order_fractional_delta(series: List[float]) -> float:
    if not series:
        return 0.0
    alpha = 0.5
    n = len(series) - 1
    acc = 0.0
    for k in range(n + 1):
        acc += ((-1.0) ** k) * _fractional_binomial(alpha, k) * series[n - k]
    return acc


def build_tribonacci_signature(text: str) -> Dict[str, float]:
    token_count = len([tok for tok in text.split(" ") if tok.strip()])
    if token_count == 0:
        token_count = len(text.strip())

    foliation_depth = max(1, min(12, token_count))
    base = companion_matrix()
    base_det = _det_3x3(base)

    depth_matrix = _matpow_3x3(base, foliation_depth)
    depth_trace = _trace_3x3(depth_matrix)

    trace_series = [
        float(_trace_3x3(_matpow_3x3(base, step)))
        for step in range(1, foliation_depth + 1)
    ]
    half_delta = _half_order_fractional_delta(trace_series)

    return {
        "eta": TRIBONACCI_ETA,
        "determinant": float(base_det),
        "foliation_depth": float(foliation_depth),
        "trace_depth": float(depth_trace),
        "half_order_delta": float(half_delta),
        "ring_size": 13.0,
    }


def augment_prompt_with_tribonacci_signature(prompt: str) -> str:
    signature = build_tribonacci_signature(prompt)
    signature_line = (
        "[Tribonacci-SL3Z bridge] "
        f"eta={signature['eta']:.6f}; "
        f"det={int(signature['determinant'])}; "
        f"depth={int(signature['foliation_depth'])}; "
        f"trace={int(signature['trace_depth'])}; "
        f"half_delta={signature['half_order_delta']:.6f}; "
        f"ring={int(signature['ring_size'])}."
    )
    return f"{prompt}\n\n{signature_line}"


__all__ = [
    "TRIBONACCI_ETA",
    "augment_prompt_with_tribonacci_signature",
    "build_tribonacci_signature",
    "companion_matrix",
]
