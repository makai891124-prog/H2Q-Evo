#!/usr/bin/env python3
"""Formalize bootstrap plateau diagnostics and dual-conjugate projection analysis.

This script links:
1) Self-bootstrap execution evidence (keep/discard causes),
2) High-dimensional dual-conjugate embedding,
3) Orthogonal projection geometry for AP/GP radial laws,
4) Golden-ratio convergence on the AP-GP boundary (Fibonacci recurrence).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


@dataclass
class BootstrapSummary:
    keep: int
    discard: int
    crash: int
    distill_deltas: List[float]
    research_deltas: List[float]
    systemic_deltas: List[float]
    unique_cmd_signatures: int


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def summarize_bootstrap(payload: Dict[str, Any]) -> BootstrapSummary:
    summary = payload.get("summary") or {}
    experiments = payload.get("experiments") or []

    distill_deltas: List[float] = []
    research_deltas: List[float] = []
    systemic_deltas: List[float] = []
    signatures = set()

    for row in experiments:
        name = str(row.get("name", ""))
        delta = _safe_float(row.get("delta", 0.0), 0.0)
        cmd = tuple(row.get("cmd") or [])
        signatures.add((name, cmd))

        if name == "distillation_uplift":
            distill_deltas.append(delta)
        elif name == "research_cross_validation":
            research_deltas.append(delta)
        elif name == "systemic_joint_capability":
            systemic_deltas.append(delta)

    return BootstrapSummary(
        keep=int(summary.get("keep", 0)),
        discard=int(summary.get("discard", 0)),
        crash=int(summary.get("crash", 0)),
        distill_deltas=distill_deltas,
        research_deltas=research_deltas,
        systemic_deltas=systemic_deltas,
        unique_cmd_signatures=len(signatures),
    )


def complex_channels(radius: np.ndarray, omega: float) -> np.ndarray:
    n = np.arange(radius.shape[0], dtype=np.float64)
    z_plus = radius * np.exp(1j * omega * n)
    z_minus = radius * np.exp(-1j * omega * n)

    # R^4 embedding: [Re z+, Im z+, Re z-, Im z-]
    emb = np.stack([z_plus.real, z_plus.imag, z_minus.real, z_minus.imag], axis=1)
    return emb.astype(np.float64)


def build_projection_basis() -> np.ndarray:
    # Conjugate-sum and conjugate-difference axes.
    e1 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2], axis=1)  # shape (4, 2)


def project_to_plane(embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return embedding @ basis


def radial_sequences(n_steps: int) -> Dict[str, np.ndarray]:
    n = np.arange(n_steps, dtype=np.float64)

    # AP and GP representatives.
    r_ap = 1.0 + 0.05 * n
    r_gp = 1.0 * (1.03 ** n)

    # AP-GP boundary candidate via Fibonacci recurrence.
    fib = np.zeros(n_steps + 2, dtype=np.float64)
    fib[0] = 1.0
    fib[1] = 1.0
    for i in range(2, n_steps + 2):
        fib[i] = fib[i - 1] + fib[i - 2]
    r_boundary = fib[1 : n_steps + 1]

    return {
        "ap": r_ap,
        "gp": r_gp,
        "boundary": r_boundary,
    }


def mean_second_diff(radius: np.ndarray) -> float:
    if radius.shape[0] < 3:
        return 0.0
    d2 = radius[2:] - 2.0 * radius[1:-1] + radius[:-2]
    return float(np.mean(d2))


def projection_growth_ratios(points: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(points, axis=1)
    prev = np.maximum(radii[:-1], 1e-12)
    return radii[1:] / prev


def analyze_formal_geometry(n_steps: int = 80, omega: float = 0.23) -> Dict[str, Any]:
    seq = radial_sequences(n_steps)
    basis = build_projection_basis()

    out: Dict[str, Any] = {
        "basis": {
            "vectors": basis.tolist(),
            "dot_product": float(np.dot(basis[:, 0], basis[:, 1])),
        },
        "modes": {},
    }

    for mode, radius in seq.items():
        emb = complex_channels(radius, omega)
        pts = project_to_plane(emb, basis)
        ratios = projection_growth_ratios(pts)
        out["modes"][mode] = {
            "mean_second_diff_radius": mean_second_diff(radius),
            "mean_growth_ratio": float(np.mean(ratios)),
            "tail_growth_ratio": float(np.mean(ratios[-10:])),
            "tail_growth_std": float(np.std(ratios[-10:])),
        }

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    boundary_tail_ratio = out["modes"]["boundary"]["tail_growth_ratio"]
    out["golden_ratio_test"] = {
        "phi": phi,
        "boundary_tail_growth_ratio": boundary_tail_ratio,
        "absolute_error": abs(boundary_tail_ratio - phi),
    }
    return out


def explain_plateau(bs: BootstrapSummary, baseline: Dict[str, Any], final: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    distill_sat = len(bs.distill_deltas) > 0 and max(abs(x) for x in bs.distill_deltas) < 1e-12
    research_sat = len(bs.research_deltas) > 0 and max(abs(x) for x in bs.research_deltas) < 1e-12

    if distill_sat and _safe_float(baseline.get("distill_delta", 0.0), 0.0) >= 0.999:
        lines.append("Distillation metric is saturated near ceiling (delta_schema_valid_rate ~= 1.0), so replaying same teacher/session settings yields no gain.")
    if research_sat and _safe_float(baseline.get("research_aggregate", 0.0), 0.0) >= 0.99:
        lines.append("Research aggregate is already in high-score plateau (>0.99), so deterministic reruns produce near-zero marginal improvement.")

    if bs.unique_cmd_signatures <= 3:
        lines.append("Exploration bandwidth is low: each round repeats the same 3 command signatures without parameter mutation.")

    systemic_gain = _safe_float(final.get("systemic_score", 0.0), 0.0) - _safe_float(baseline.get("systemic_score", 0.0), 0.0)
    if systemic_gain > 0:
        lines.append("Current incremental headroom is mostly in systemic joint capability; this should receive adaptive budget and parameter sweeps.")

    if not lines:
        lines.append("No dominant single failure mode detected; likely multi-factor coupling and measurement noise.")

    return lines


def render_markdown(payload: Dict[str, Any]) -> str:
    b = payload["bootstrap"]
    g = payload["formal_geometry"]
    phi_test = g["golden_ratio_test"]

    lines = [
        "# Bootstrap Plateau and Dual-Conjugate Formalization",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- source_bootstrap_json: `{payload['source_bootstrap_json']}`",
        "",
        "## 1. Bootstrap Replay Diagnosis",
        "",
        f"- keep/discard/crash: `{b['keep']}/{b['discard']}/{b['crash']}`",
        f"- unique command signatures: `{b['unique_cmd_signatures']}`",
        f"- distillation deltas: `{b['distill_deltas']}`",
        f"- research deltas: `{b['research_deltas']}`",
        f"- systemic deltas: `{b['systemic_deltas']}`",
        "",
        "### Why capability gain stalls",
    ]

    for item in payload["plateau_hypotheses"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 2. Dual-Conjugate High-Dimensional Formalization",
            "",
            "State embedding in R^4:",
            "- v_n = [Re(z+_n), Im(z+_n), Re(z-_n), Im(z-_n)]",
            "- z+_n = r_n * exp(i * omega * n), z-_n = r_n * exp(-i * omega * n)",
            "",
            "Orthogonal projection basis (linearly independent):",
            "- e1 = normalize([1, 0, 1, 0])",
            "- e2 = normalize([0, 1, 0, -1])",
            f"- dot(e1, e2): `{g['basis']['dot_product']:.6f}`",
            "",
            "## 3. AP/GP Split and Motion Semantics",
            "",
            f"- AP mean second-diff radius: `{g['modes']['ap']['mean_second_diff_radius']:.6e}`",
            f"- GP mean second-diff radius: `{g['modes']['gp']['mean_second_diff_radius']:.6e}`",
            "- Interpretation: AP behaves as near-uniform radial drift; GP yields positive radial acceleration in projected circular motion.",
            "",
            "## 4. Golden-Ratio Limit on AP-GP Boundary",
            "",
            "Boundary sequence uses Fibonacci recurrence r_{n+1} = r_n + r_{n-1}.",
            f"- phi: `{phi_test['phi']:.12f}`",
            f"- observed tail growth ratio: `{phi_test['boundary_tail_growth_ratio']:.12f}`",
            f"- absolute error: `{phi_test['absolute_error']:.3e}`",
            "- Interpretation: under this boundary construction, projected growth ratio converges toward phi as n increases.",
            "",
            "## 5. Actionable Next Steps",
            "",
            "- Add parameter mutation in distillation and research steps (sessions, teacher provider, aggregation folds).",
            "- Allocate adaptive iteration budget to components with recent positive delta (currently systemic).",
            "- Track the AP/GP-boundary growth-ratio metric together with MeaningScore to detect geometric regime shifts.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    src = REPORTS / "autoresearch_h2q_bootstrap_fusion_latest.json"
    bootstrap = _load_json(src)
    if not bootstrap:
        raise SystemExit(f"Missing or invalid bootstrap report: {src}")

    bs = summarize_bootstrap(bootstrap)
    formal = analyze_formal_geometry(n_steps=100, omega=0.23)

    baseline = bootstrap.get("baseline_snapshot") or {}
    final = bootstrap.get("final_snapshot") or {}

    payload: Dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_bootstrap_json": str(src),
        "bootstrap": {
            "keep": bs.keep,
            "discard": bs.discard,
            "crash": bs.crash,
            "distill_deltas": bs.distill_deltas,
            "research_deltas": bs.research_deltas,
            "systemic_deltas": bs.systemic_deltas,
            "unique_cmd_signatures": bs.unique_cmd_signatures,
            "baseline_snapshot": baseline,
            "final_snapshot": final,
        },
        "plateau_hypotheses": explain_plateau(bs, baseline, final),
        "formal_geometry": formal,
    }

    ts = int(time.time())
    out_json = REPORTS / f"dual_conjugate_bootstrap_formalization_{ts}.json"
    out_md = REPORTS / f"dual_conjugate_bootstrap_formalization_{ts}.md"
    latest_json = REPORTS / "dual_conjugate_bootstrap_formalization_latest.json"
    latest_md = REPORTS / "dual_conjugate_bootstrap_formalization_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")


if __name__ == "__main__":
    main()
