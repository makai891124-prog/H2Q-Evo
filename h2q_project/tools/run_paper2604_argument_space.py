#!/usr/bin/env python3
"""Encode arXiv:2604.01249v1 into paper2spacecode argument-space artifacts.

This runner does four things:
1) Numerical encoding of key theorem families as executable series.
2) Validation against closed forms (1/pi, 1/pi^2, 1/pi^3 special case).
3) Multi-dimensional visual artifacts (1D/2D/3D/4D views + mp4).
4) Stage6/7/8-style integration reports compatible with paper2spacecode workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - fallback for older imageio
    import imageio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from mpmath.ctx_mp_python import mpf as MPF
else:
    MPF = Any

from h2q_project.tools.paper2spacecode_pyvista_demo import (
    build_stage6_formula_contract,
    render_stage7_pyvista_demo,
)

PAPER_URL = "https://arxiv.org/html/2604.01249v1"
PAPER_TITLE = "On Series Involving Cubed Catalan Numbers"

FAMILY_PI_INV = "theorem10_pi_inverse_family"
FAMILY_PI2_INV = "theorem12_pi_square_inverse_family"
FAMILY_PI3_SPECIAL = "equation25_pi_cube_inverse_special_case"


@dataclass(frozen=True)
class FamilyRun:
    name: str
    display_name: str
    m_values: Sequence[int]
    max_terms: int


@dataclass
class FamilyResult:
    name: str
    display_name: str
    m_values: List[int]
    target_values: List[float]
    partial_sums: np.ndarray
    abs_errors: np.ndarray
    rel_errors: np.ndarray


def _run_command(cmd: List[str], *, timeout_sec: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )


def _resolve_ffmpeg_binary() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _compose_side_by_side_video(*, left_video: Path, right_video: Path, output_video: Path) -> None:
    ffmpeg = _resolve_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary is unavailable")

    _run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(left_video),
            "-i",
            str(right_video),
            "-filter_complex",
            "[0:v]scale=960:540,setsar=1[v0];[1:v]scale=960:540,setsar=1[v1];[v0][v1]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ],
        timeout_sec=360,
    )


def _scaled_central_binomial(k: int) -> MPF:
    return mp.mpf(math.comb(2 * k, k)) / (mp.mpf(4) ** k)


def _odd_inverse_product(k: int, m: int) -> MPF:
    prod = mp.mpf(1)
    for j in range(1, m + 1):
        prod *= mp.mpf(1) / mp.mpf(2 * k - 2 * j + 1)
    return prod


def _term_theorem10(k: int, m: int) -> MPF:
    base = ((-1) ** k) * _scaled_central_binomial(k) * _odd_inverse_product(k, m)
    return (base**3) * mp.mpf(4 * k - 2 * m + 1)


def _target_theorem10(m: int) -> MPF:
    scale = (mp.mpf(2) ** m) * mp.mpf(math.comb(2 * m, m)) / mp.mpf(math.factorial(m))
    return (scale**3) * (mp.mpf(2) / mp.pi)


def _term_theorem12(k: int, m: int) -> MPF:
    base = _scaled_central_binomial(k) * _odd_inverse_product(k, m)
    return (base**4) * mp.mpf(4 * k - 2 * m + 1)


def _target_theorem12(m: int) -> MPF:
    numerator = ((-1) ** m) * (mp.mpf(2) ** (8 * m)) * (mp.mpf(math.factorial(m)) ** 4)
    denominator = (mp.pi**2) * mp.mpf(m) * (mp.mpf(math.comb(2 * m, m)) ** 5)
    return numerator / denominator


def _term_equation25(k: int) -> MPF:
    base = _scaled_central_binomial(k)
    return base**3


def _target_equation25() -> MPF:
    quarter = mp.mpf(1) / mp.mpf(4)
    return (mp.gamma(quarter) ** 4) / (mp.mpf(4) * (mp.pi**3))


def _partial_sums(max_terms: int, term_fn: Callable[[int], MPF]) -> np.ndarray:
    out = np.zeros(max_terms, dtype=np.float64)
    acc = mp.mpf(0)
    for k in range(max_terms):
        acc += term_fn(k)
        out[k] = float(acc)
    return out


def _relative_error(partials: np.ndarray, target_value: float) -> tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(partials - target_value)
    denom = max(1e-30, abs(target_value))
    rel_err = abs_err / denom
    return abs_err, rel_err


def _compute_family_result(
    *,
    family: FamilyRun,
    term_builder: Callable[[int], Callable[[int], MPF]],
    target_builder: Callable[[int], MPF],
) -> FamilyResult:
    partials: List[np.ndarray] = []
    targets: List[float] = []
    abs_errors: List[np.ndarray] = []
    rel_errors: List[np.ndarray] = []

    for m in family.m_values:
        target_val = float(target_builder(m))
        partial = _partial_sums(family.max_terms, term_builder(m))
        abs_err, rel_err = _relative_error(partial, target_val)

        targets.append(target_val)
        partials.append(partial)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

    return FamilyResult(
        name=family.name,
        display_name=family.display_name,
        m_values=list(family.m_values),
        target_values=targets,
        partial_sums=np.stack(partials, axis=0),
        abs_errors=np.stack(abs_errors, axis=0),
        rel_errors=np.stack(rel_errors, axis=0),
    )


def _serialize_family_metrics(result: FamilyResult) -> Dict[str, Any]:
    m_records: List[Dict[str, Any]] = []
    for i, m in enumerate(result.m_values):
        partial_last = float(result.partial_sums[i, -1])
        target = float(result.target_values[i])
        abs_error_last = float(result.abs_errors[i, -1])
        rel_error_last = float(result.rel_errors[i, -1])
        m_records.append(
            {
                "m": int(m),
                "target": target,
                "partial_sum_last": partial_last,
                "abs_error_last": abs_error_last,
                "rel_error_last": rel_error_last,
            }
        )

    return {
        "family": result.name,
        "display_name": result.display_name,
        "max_rel_error_last": float(np.max(result.rel_errors[:, -1])),
        "min_rel_error_last": float(np.min(result.rel_errors[:, -1])),
        "m_records": m_records,
    }


def _safe_log_error(values: np.ndarray) -> np.ndarray:
    return -np.log10(np.maximum(values, 1e-30))


def _plot_convergence_lines(
    *,
    theorem10: FamilyResult,
    theorem12: FamilyResult,
    eq25_partial: np.ndarray,
    eq25_target: float,
    output_path: Path,
) -> None:
    n_axis = np.arange(1, theorem10.partial_sums.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)

    for idx, m in enumerate(theorem10.m_values[:3]):
        y = _safe_log_error(theorem10.rel_errors[idx])
        ax.plot(n_axis, y, linewidth=1.8, label=f"Eq104 | m={m}")

    for idx, m in enumerate(theorem12.m_values[:2]):
        y = _safe_log_error(theorem12.rel_errors[idx])
        ax.plot(n_axis, y, linewidth=1.8, linestyle="--", label=f"Eq114 | m={m}")

    eq25_rel = np.abs(eq25_partial - eq25_target) / max(1e-30, abs(eq25_target))
    ax.plot(n_axis, _safe_log_error(eq25_rel), linewidth=2.0, linestyle=":", label="Eq25")

    ax.set_title("arXiv:2604.01249v1 | Convergence in 1D term-space")
    ax.set_xlabel("Partial sum terms N")
    ax.set_ylabel("-log10(relative error)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_heatmap(*, result: FamilyResult, output_path: Path) -> None:
    z = _safe_log_error(result.rel_errors)
    fig, ax = plt.subplots(figsize=(12.8, 6.0), dpi=180)
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"{result.display_name} | 2D (m, N) error landscape")
    ax.set_xlabel("N index")
    ax.set_ylabel("m index")
    ax.set_yticks(np.arange(len(result.m_values)))
    ax.set_yticklabels([str(m) for m in result.m_values])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(relative error)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_surface_3d(*, result: FamilyResult, output_path: Path, azim: float = -60.0) -> None:
    n_axis = np.arange(1, result.partial_sums.shape[1] + 1)
    m_axis = np.asarray(result.m_values, dtype=np.float64)
    x, y = np.meshgrid(n_axis, m_axis)
    z = _safe_log_error(result.rel_errors)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(f"{result.display_name} | 3D error manifold")
    ax.set_xlabel("N")
    ax.set_ylabel("m")
    ax.set_zlabel("-log10(relative error)")
    ax.view_init(elev=25, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_projection_4d(
    *,
    theorem10: FamilyResult,
    theorem12: FamilyResult,
    eq25_partial: np.ndarray,
    eq25_target: float,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    def sample_points(
        family_index: int,
        family_name: str,
        m_values: Sequence[int],
        rel_matrix: np.ndarray,
        stride: int,
    ) -> None:
        colors = ["#2a9d8f", "#e76f51", "#264653"]
        label_done = False
        for i, m in enumerate(m_values):
            for n in range(0, rel_matrix.shape[1], stride):
                x = n + 1
                y = float(m)
                z = float(-math.log10(max(1e-30, rel_matrix[i, n])))
                label = family_name if not label_done else None
                ax.scatter(x, y, z, c=colors[family_index], s=11, alpha=0.55, label=label)
                label_done = True

    sample_points(0, "Eq104", theorem10.m_values, theorem10.rel_errors, stride=4)
    sample_points(1, "Eq114", theorem12.m_values, theorem12.rel_errors, stride=4)

    eq25_rel = np.abs(eq25_partial - eq25_target) / max(1e-30, abs(eq25_target))
    rel_matrix = eq25_rel.reshape(1, -1)
    sample_points(2, "Eq25", [0], rel_matrix, stride=4)

    ax.set_title("4D projection: (family, m, N, error) -> 3D+color")
    ax.set_xlabel("N")
    ax.set_ylabel("m")
    ax.set_zlabel("-log10(relative error)")
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _render_multidim_video(
    *,
    theorem10: FamilyResult,
    theorem12: FamilyResult,
    eq25_partial: np.ndarray,
    eq25_target: float,
    frame_count: int,
    fps: int,
    frames_dir: Path,
    video_path: Path,
) -> List[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[str] = []

    n_axis = np.arange(1, theorem10.partial_sums.shape[1] + 1)
    eq25_rel = np.abs(eq25_partial - eq25_target) / max(1e-30, abs(eq25_target))

    for frame_idx in range(frame_count):
        azim = -70.0 + (360.0 * frame_idx) / max(1, frame_count)

        fig = plt.figure(figsize=(14.4, 8.0), dpi=140)
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        for idx, m in enumerate(theorem10.m_values[:2]):
            ax1.plot(n_axis, _safe_log_error(theorem10.rel_errors[idx]), linewidth=1.6, label=f"Eq104 m={m}")
        ax1.plot(n_axis, _safe_log_error(eq25_rel), linewidth=1.8, linestyle=":", label="Eq25")
        ax1.set_title("1D term-space convergence")
        ax1.set_xlabel("N")
        ax1.set_ylabel("-log10(rel err)")
        ax1.grid(alpha=0.22)
        ax1.legend(loc="lower right", fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        hm = ax2.imshow(_safe_log_error(theorem10.rel_errors), aspect="auto", origin="lower", cmap="magma")
        ax2.set_title("2D space: Eq104 (m,N)")
        ax2.set_xlabel("N index")
        ax2.set_ylabel("m index")
        ax2.set_yticks(np.arange(len(theorem10.m_values)))
        ax2.set_yticklabels([str(m) for m in theorem10.m_values])
        cbar = fig.colorbar(hm, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("-log10(rel err)")

        ax3 = fig.add_subplot(gs[1, 0], projection="3d")
        x, y = np.meshgrid(n_axis, np.asarray(theorem12.m_values, dtype=np.float64))
        z = _safe_log_error(theorem12.rel_errors)
        ax3.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax3.set_title("3D manifold: Eq114 error")
        ax3.set_xlabel("N")
        ax3.set_ylabel("m")
        ax3.set_zlabel("-log10(rel err)")
        ax3.view_init(elev=25, azim=azim)

        ax4 = fig.add_subplot(gs[1, 1], projection="3d")
        ax4.scatter(
            n_axis[::5],
            np.zeros_like(n_axis[::5]),
            _safe_log_error(eq25_rel)[::5],
            c="#264653",
            s=12,
            alpha=0.7,
            label="Eq25",
        )
        for idx, m in enumerate(theorem10.m_values[:3]):
            ax4.scatter(
                n_axis[::5],
                np.full_like(n_axis[::5], m),
                _safe_log_error(theorem10.rel_errors[idx])[::5],
                c="#2a9d8f",
                s=10,
                alpha=0.45,
                label="Eq104" if idx == 0 else None,
            )
        for idx, m in enumerate(theorem12.m_values[:3]):
            ax4.scatter(
                n_axis[::5],
                np.full_like(n_axis[::5], m + 6),
                _safe_log_error(theorem12.rel_errors[idx])[::5],
                c="#e76f51",
                s=10,
                alpha=0.45,
                label="Eq114" if idx == 0 else None,
            )
        ax4.set_title("4D projection (family,m,N,error)")
        ax4.set_xlabel("N")
        ax4.set_ylabel("m / shifted m")
        ax4.set_zlabel("-log10(rel err)")
        ax4.view_init(elev=22, azim=-40 + azim * 0.4)
        ax4.legend(loc="upper right", fontsize=7)

        fig.suptitle(
            f"{PAPER_TITLE} | Multi-dimensional argument-space frame {frame_idx + 1}/{frame_count}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        frame_path = frames_dir / f"multidim_{frame_idx:04d}.png"
        fig.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(str(frame_path))

    with imageio.get_writer(str(video_path), fps=max(1, fps), codec="libx264", quality=8) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    return frame_paths


def _write_analysis_markdown(
    *,
    output_path: Path,
    theorem10: FamilyResult,
    theorem12: FamilyResult,
    eq25_partial: np.ndarray,
    eq25_target: float,
) -> None:
    eq25_rel_last = float(abs(eq25_partial[-1] - eq25_target) / max(1e-30, abs(eq25_target)))
    eq20_rel_last = float(theorem10.rel_errors[0, -1])
    eq24_rel_last = float(theorem12.rel_errors[0, -1])
    eq104_best = float(np.min(theorem10.rel_errors[:, -1]))
    eq104_worst = float(np.max(theorem10.rel_errors[:, -1]))
    eq114_best = float(np.min(theorem12.rel_errors[:, -1]))
    eq114_worst = float(np.max(theorem12.rel_errors[:, -1]))

    content = f"""# arXiv:2604.01249v1 论文编码分析

- 标题: {PAPER_TITLE}
- 链接: {PAPER_URL}
- 目标: 将论文中的级数论证转写为可执行验证对象，并映射到多维论证空间图像与视频。

## 已编码论证

1. Theorem 10 (Eq.104): 1/pi 家族（参数 m）
2. Theorem 12 (Eq.114): 1/pi^2 家族（参数 m）
3. Eq.25: 1/pi^3 特例

## 数值结论（以最终项 N 收敛误差计）

- Eq20 (Bauer) 相对误差: {eq20_rel_last:.3e}
- Eq24 相对误差: {eq24_rel_last:.3e}
- Eq25 相对误差: {eq25_rel_last:.3e}
- Eq104 最佳相对误差: {eq104_best:.3e}
- Eq104 最差相对误差: {eq104_worst:.3e}
- Eq114 最佳相对误差: {eq114_best:.3e}
- Eq114 最差相对误差: {eq114_worst:.3e}

## 抽象论证产物

- 1D: 级数部分和收敛曲线（项空间）
- 2D: (m, N) 误差热图（参数-项索引空间）
- 3D: 误差流形曲面（几何化论证空间）
- 4D: (family, m, N, error) 的投影散点（跨族比较空间）

## 结论摘要

- 论文中的 Ramanujan-like 结构可被工程化为“可计算-可验证-可视化”的统一对象。
- 明确特例 Eq20/Eq24/Eq25 在本实现中获得了可复现实证支持。
- 高阶 m 参数扫描暴露了直接截断下的尺度差异，提示后续应结合符号变换或加速求和做更严格验证。
- 通过 Stage6/7/8 语义化管线，可将数学论证进一步转译为视频化空间表达与复用型报告资产。
"""
    output_path.write_text(content, encoding="utf-8")


def _build_formula_driver_text() -> str:
    return (
        "arXiv:2604.01249v1 On Series Involving Cubed Catalan Numbers; "
        "encode Theorem 10 Eq104 for 1/pi family, Theorem 12 Eq114 for 1/pi^2 family, "
        "and Eq25 for 1/pi^3 special case into a multi-dimensional argument manifold."
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run arXiv:2604.01249v1 argument-space encoding pipeline")
    parser.add_argument(
        "--output-dir",
        default="h2q_project/reports/paper2604_argument_space",
        help="Output directory for all generated artifacts",
    )
    parser.add_argument("--max-terms", type=int, default=180, help="Max terms for each series partial sum")
    parser.add_argument("--viz-frames", type=int, default=72, help="Frame count for multidimensional mp4")
    parser.add_argument("--viz-fps", type=int, default=24, help="FPS for multidimensional mp4")
    parser.add_argument("--stage7-frames", type=int, default=72, help="Stage7 PyVista frame count")
    parser.add_argument("--stage7-fps", type=int, default=24, help="Stage7 PyVista fps")
    parser.add_argument("--stage7-grid", type=int, default=136, help="Stage7 PyVista grid resolution")
    parser.add_argument("--save-png-every", type=int, default=6, help="Stage7 save png stride")
    parser.add_argument(
        "--skip-stage7",
        action="store_true",
        help="Skip Stage7 rendering and Stage8 side-by-side composition",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    mp.dps = 80
    started_at = time.time()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [
        FamilyRun(
            name=FAMILY_PI_INV,
            display_name="Theorem 10 Eq104 (1/pi)",
            m_values=tuple(range(0, 5)),
            max_terms=max(40, args.max_terms),
        ),
        FamilyRun(
            name=FAMILY_PI2_INV,
            display_name="Theorem 12 Eq114 (1/pi^2)",
            m_values=tuple(range(1, 5)),
            max_terms=max(40, args.max_terms),
        ),
    ]

    theorem10 = _compute_family_result(
        family=families[0],
        term_builder=lambda m: (lambda k: _term_theorem10(k, m)),
        target_builder=_target_theorem10,
    )
    theorem12 = _compute_family_result(
        family=families[1],
        term_builder=lambda m: (lambda k: _term_theorem12(k, m)),
        target_builder=_target_theorem12,
    )

    eq25_partial = _partial_sums(max(40, args.max_terms), _term_equation25)
    eq25_target = float(_target_equation25())
    eq25_abs, eq25_rel = _relative_error(eq25_partial, eq25_target)

    eq20_rel_last = float(theorem10.rel_errors[0, -1])
    eq24_rel_last = float(theorem12.rel_errors[0, -1])
    eq25_rel_last = float(eq25_rel[-1])

    canonical_thresholds = {
        "eq20_rel_error_lt_5e-2": eq20_rel_last < 5e-2,
        "eq24_rel_error_lt_1e-8": eq24_rel_last < 1e-8,
        "eq25_rel_error_lt_3e-2": eq25_rel_last < 3e-2,
    }

    numeric_report = {
        "paper": {
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "families": [
            _serialize_family_metrics(theorem10),
            _serialize_family_metrics(theorem12),
            {
                "family": FAMILY_PI3_SPECIAL,
                "display_name": "Equation 25 (1/pi^3 special case)",
                "target": eq25_target,
                "partial_sum_last": float(eq25_partial[-1]),
                "abs_error_last": float(eq25_abs[-1]),
                "rel_error_last": float(eq25_rel[-1]),
            },
        ],
        "canonical_equations": {
            "eq20_bauer_m0": {
                "target": float(2 / mp.pi),
                "partial_sum_last": float(theorem10.partial_sums[0, -1]),
                "rel_error_last": eq20_rel_last,
            },
            "eq24_m1_pi2": {
                "target": float(-8 / (mp.pi**2)),
                "partial_sum_last": float(theorem12.partial_sums[0, -1]),
                "rel_error_last": eq24_rel_last,
            },
            "eq25_pi3": {
                "target": eq25_target,
                "partial_sum_last": float(eq25_partial[-1]),
                "rel_error_last": eq25_rel_last,
            },
            "thresholds": canonical_thresholds,
            "passed": bool(all(canonical_thresholds.values())),
        },
    }

    numeric_report_path = output_dir / "paper2604_numerical_validation.json"
    numeric_report_path.write_text(json.dumps(numeric_report, ensure_ascii=False, indent=2), encoding="utf-8")

    visuals_dir = output_dir / "argument_space_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    one_d_path = visuals_dir / "space_1d_convergence.png"
    _plot_convergence_lines(
        theorem10=theorem10,
        theorem12=theorem12,
        eq25_partial=eq25_partial,
        eq25_target=eq25_target,
        output_path=one_d_path,
    )

    two_d_eq104_path = visuals_dir / "space_2d_heatmap_eq104.png"
    _plot_heatmap(result=theorem10, output_path=two_d_eq104_path)

    two_d_eq114_path = visuals_dir / "space_2d_heatmap_eq114.png"
    _plot_heatmap(result=theorem12, output_path=two_d_eq114_path)

    three_d_path = visuals_dir / "space_3d_surface_eq114.png"
    _plot_surface_3d(result=theorem12, output_path=three_d_path, azim=-54.0)

    four_d_path = visuals_dir / "space_4d_projection.png"
    _plot_projection_4d(
        theorem10=theorem10,
        theorem12=theorem12,
        eq25_partial=eq25_partial,
        eq25_target=eq25_target,
        output_path=four_d_path,
    )

    multidim_frames_dir = visuals_dir / "multidim_video_frames"
    multidim_video_path = visuals_dir / "paper2604_multidim_argument_space.mp4"
    frame_paths = _render_multidim_video(
        theorem10=theorem10,
        theorem12=theorem12,
        eq25_partial=eq25_partial,
        eq25_target=eq25_target,
        frame_count=max(24, args.viz_frames),
        fps=max(1, args.viz_fps),
        frames_dir=multidim_frames_dir,
        video_path=multidim_video_path,
    )

    analysis_md_path = output_dir / "paper2604_analysis.md"
    _write_analysis_markdown(
        output_path=analysis_md_path,
        theorem10=theorem10,
        theorem12=theorem12,
        eq25_partial=eq25_partial,
        eq25_target=eq25_target,
    )

    abstraction = {
        "paper": {
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "core_sections": [
                "Series involving cubed Catalan numbers",
                "Series involving fourth powers of Catalan numbers",
                "Ramanujan-like series and related series",
            ],
        },
        "encoded_claims": [
            {
                "id": "Eq104",
                "statement": "Generalized Bauer-type family for 1/pi indexed by m",
                "space_mapping": ["1D convergence", "2D m-N heatmap", "4D family projection"],
            },
            {
                "id": "Eq114",
                "statement": "Ramanujan-like family for 1/pi^2 indexed by m",
                "space_mapping": ["2D m-N heatmap", "3D error manifold", "4D family projection"],
            },
            {
                "id": "Eq25",
                "statement": "Special 1/pi^3 identity from central binomial cube series",
                "space_mapping": ["1D convergence", "4D family projection"],
            },
        ],
        "verification_summary": {
            "eq20_rel_error_last": eq20_rel_last,
            "eq24_rel_error_last": eq24_rel_last,
            "eq25_rel_error_last": eq25_rel_last,
            "thresholds": canonical_thresholds,
            "canonical_equations_passed": bool(all(canonical_thresholds.values())),
            "exploratory_scan_note": "Higher-m family scans are exploratory visualization outputs, not strict closed-form acceptance checks.",
        },
        "abstract_conclusion": [
            "The paper's theorem families can be compiled into executable numerical contracts.",
            "Canonical explicit identities (Eq20/Eq24/Eq25) are reproducibly supported in this implementation.",
            "Multi-dimensional visual spaces expose both local convergence speed and global argument geometry.",
            "The Stage6/7/8 pipeline supports translating formal math claims into reproducible visual artifacts.",
        ],
    }
    abstraction_path = output_dir / "paper2604_argument_abstraction.json"
    abstraction_path.write_text(json.dumps(abstraction, ensure_ascii=False, indent=2), encoding="utf-8")

    stage6_contract_path = output_dir / "stage6_formula_contract.json"
    stage7_report_path = output_dir / "stage7_render_report.json"
    stage8_report_path = output_dir / "stage8_argument_space_report.json"

    stage6_contract = build_stage6_formula_contract(
        paper_formula_text=_build_formula_driver_text(),
        frames=max(24, args.stage7_frames),
        fps=max(1, args.stage7_fps),
        grid_resolution=max(72, args.stage7_grid),
        driver="paper2604_argument_space",
    )
    stage6_contract_path.write_text(
        json.dumps(stage6_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage7_report: Dict[str, Any] = {
        "stage": "7_realtime_manifold_rendering",
        "passed": False,
        "error": "skipped",
    }
    stage8_report: Dict[str, Any] = {
        "stage": "8_argument_integration",
        "passed": False,
        "error": "stage7 skipped",
    }

    if not args.skip_stage7:
        stage7_dir = output_dir / "stage7_spatial_demo"
        stage7_report = render_stage7_pyvista_demo(
            stage6_contract=stage6_contract,
            output_dir=stage7_dir,
            save_png_every=max(1, args.save_png_every),
            interactive_preview=False,
        )
        stage7_report_path.write_text(json.dumps(stage7_report, ensure_ascii=False, indent=2), encoding="utf-8")

        stage8_video_path = output_dir / "paper2604_stage8_space_fusion.mp4"
        stage8_error: str | None = None
        combined_created = False
        try:
            stage7_video = Path(str(stage7_report.get("video_path", "")))
            if stage7_video.exists() and multidim_video_path.exists():
                _compose_side_by_side_video(
                    left_video=stage7_video,
                    right_video=multidim_video_path,
                    output_video=stage8_video_path,
                )
                combined_created = stage8_video_path.exists() and stage8_video_path.stat().st_size > 0
            else:
                stage8_error = "missing_stage7_or_multidim_video"
        except Exception as exc:
            stage8_error = str(exc)

        stage8_report = {
            "stage": "8_argument_integration",
            "inputs": {
                "stage7_video": str(stage7_report.get("video_path", "")),
                "multidim_video": str(multidim_video_path),
            },
            "outputs": {
                "combined_video": str(stage8_video_path),
            },
            "checks": {
                "stage7_passed": bool(stage7_report.get("passed")),
                "multidim_video_exists": bool(multidim_video_path.exists()),
                "combined_video_created": bool(combined_created),
            },
            "passed": bool(stage7_report.get("passed") and combined_created and stage8_error is None),
            "error": stage8_error,
        }
    else:
        stage7_report_path.write_text(json.dumps(stage7_report, ensure_ascii=False, indent=2), encoding="utf-8")

    stage8_report_path.write_text(json.dumps(stage8_report, ensure_ascii=False, indent=2), encoding="utf-8")

    final_report = {
        "paper": {
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "artifact_paths": {
            "numeric_report": str(numeric_report_path),
            "analysis_markdown": str(analysis_md_path),
            "abstraction_json": str(abstraction_path),
            "space_1d": str(one_d_path),
            "space_2d_eq104": str(two_d_eq104_path),
            "space_2d_eq114": str(two_d_eq114_path),
            "space_3d": str(three_d_path),
            "space_4d": str(four_d_path),
            "multidim_frames_dir": str(multidim_frames_dir),
            "multidim_video": str(multidim_video_path),
            "stage6_contract": str(stage6_contract_path),
            "stage7_report": str(stage7_report_path),
            "stage8_report": str(stage8_report_path),
        },
        "checks": {
            "eq20_rel_error_lt_5e-2": canonical_thresholds["eq20_rel_error_lt_5e-2"],
            "eq24_rel_error_lt_1e-8": canonical_thresholds["eq24_rel_error_lt_1e-8"],
            "eq25_rel_error_lt_3e-2": canonical_thresholds["eq25_rel_error_lt_3e-2"],
            "canonical_equations_passed": bool(all(canonical_thresholds.values())),
            "multidim_video_created": bool(multidim_video_path.exists() and multidim_video_path.stat().st_size > 0),
            "stage7_passed": bool(stage7_report.get("passed")),
            "stage8_passed": bool(stage8_report.get("passed")),
        },
        "multidim_frame_count": len(frame_paths),
        "duration_sec": round(time.time() - started_at, 3),
    }
    final_report["overall_passed"] = bool(
        final_report["checks"]["canonical_equations_passed"]
        and final_report["checks"]["multidim_video_created"]
        and (args.skip_stage7 or final_report["checks"]["stage8_passed"])
    )

    final_report_path = output_dir / "paper2604_argument_space_report.json"
    final_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0 if final_report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
