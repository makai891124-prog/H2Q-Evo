#!/usr/bin/env python3
"""paper2spacecode Stage 6/7 runner.

Stage 6:
- Build a machine-readable formula contract from paper formula text.

Stage 7:
- Render a real-time manifold animation with PyVista.
- Save periodic PNG frames and an MP4 video.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from h2q_project.h2q.physics.zenodo_tribonacci_bridge import build_tribonacci_signature

try:
    import pyvista as pv
except Exception:
    pv = None

if TYPE_CHECKING:
    import pyvista as pyvista_types

    PolyDataType = pyvista_types.PolyData
else:
    PolyDataType = Any


def build_stage6_formula_contract(
    *,
    paper_formula_text: str,
    frames: int,
    fps: int,
    grid_resolution: int,
    driver: str = "tribonacci_sl3z",
) -> Dict[str, Any]:
    signature = build_tribonacci_signature(paper_formula_text)
    validation = {
        "determinant_is_unitary": abs(float(signature["determinant"]) - 1.0) < 1e-9,
        "foliation_depth_in_range": 1.0 <= float(signature["foliation_depth"]) <= 12.0,
        "trace_depth_positive": float(signature["trace_depth"]) > 0.0,
    }
    validation["contract_ready"] = all(validation.values())

    return {
        "stage": "6_formula_contract",
        "driver": driver,
        "input": {
            "paper_formula_text": paper_formula_text,
        },
        "derived": signature,
        "validation": validation,
        "render_plan": {
            "frames": int(max(1, frames)),
            "fps": int(max(1, fps)),
            "grid_resolution": int(max(32, grid_resolution)),
        },
    }


def _generate_formula_manifold(
    *,
    signature: Dict[str, float],
    frame_index: int,
    total_frames: int,
    grid_resolution: int,
) -> Tuple[PolyDataType, Dict[str, float]]:
    if pv is None:
        raise RuntimeError("PyVista is unavailable")

    u = np.linspace(0.0, 2.0 * math.pi, grid_resolution, dtype=np.float32)
    v = np.linspace(0.0, 2.0 * math.pi, grid_resolution, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    phase = (2.0 * math.pi * frame_index) / max(1, total_frames - 1)
    eta = float(signature["eta"])
    trace_depth = float(signature["trace_depth"])
    half_delta = float(signature["half_order_delta"])

    major_radius = 1.45 + 0.22 * np.sin(eta * phase + 0.04 * trace_depth)
    minor_radius = 0.42 + 0.12 * np.cos(0.50 * phase + 0.15 * half_delta)
    twist = 0.55 * np.sin(phase + 0.6 * vv) + 0.10 * np.sin(0.07 * trace_depth)

    x = (major_radius + minor_radius * np.cos(vv + twist)) * np.cos(uu + 0.45 * phase)
    y = (major_radius + minor_radius * np.cos(vv + twist)) * np.sin(uu + 0.45 * phase)
    z = (
        minor_radius * np.sin(vv + twist)
        + 0.28 * np.sin(2.0 * uu + eta * phase)
        + 0.10 * np.cos(3.0 * vv - phase)
    )

    grad_u = np.gradient(z, axis=0)
    grad_v = np.gradient(z, axis=1)
    curvature_proxy = np.sqrt(grad_u * grad_u + grad_v * grad_v).astype(np.float32)

    grid = pv.StructuredGrid(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()

    flat_curvature = curvature_proxy.ravel(order="F")
    if flat_curvature.size != surface.n_points:
        flat_curvature = np.resize(flat_curvature, surface.n_points)
    surface["curvature_proxy"] = flat_curvature

    frame_metrics = {
        "frame_index": float(frame_index),
        "phase": float(phase),
        "curvature_proxy_mean": float(np.mean(flat_curvature)),
        "curvature_proxy_max": float(np.max(flat_curvature)),
    }
    return surface, frame_metrics


def render_stage7_pyvista_demo(
    *,
    stage6_contract: Dict[str, Any],
    output_dir: Path,
    save_png_every: int = 6,
    interactive_preview: bool = False,
) -> Dict[str, Any]:
    if pv is None:
        raise RuntimeError("PyVista is unavailable. Install with: pip install pyvista imageio")

    started_at = time.time()
    render_plan = stage6_contract["render_plan"]
    signature = stage6_contract["derived"]

    total_frames = int(render_plan["frames"])
    fps = int(render_plan["fps"])
    grid_resolution = int(render_plan["grid_resolution"])
    save_stride = max(1, int(save_png_every))

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "paper_formula_manifold_demo.mp4"

    plotter = pv.Plotter(off_screen=not interactive_preview, window_size=(1280, 720))
    plotter.set_background("black")
    plotter.open_movie(str(video_path), framerate=fps, quality=8)

    if interactive_preview:
        plotter.show(auto_close=False, interactive_update=True)

    saved_png_frames: List[str] = []
    frame_metrics: List[Dict[str, float]] = []
    error: str | None = None

    try:
        for frame_index in range(total_frames):
            mesh, metrics = _generate_formula_manifold(
                signature=signature,
                frame_index=frame_index,
                total_frames=total_frames,
                grid_resolution=grid_resolution,
            )
            frame_metrics.append(metrics)

            plotter.clear()
            clim_hi = max(0.01, float(np.percentile(mesh["curvature_proxy"], 99.0)))
            plotter.add_mesh(
                mesh,
                scalars="curvature_proxy",
                cmap="viridis",
                clim=[0.0, clim_hi],
                smooth_shading=True,
                specular=0.25,
            )
            plotter.add_text(
                f"paper2spacecode stage7 | frame {frame_index + 1}/{total_frames}",
                position="upper_left",
                font_size=10,
            )
            plotter.show_axes()
            plotter.camera_position = "iso"
            plotter.write_frame()

            if frame_index % save_stride == 0 or frame_index == total_frames - 1:
                png_path = frames_dir / f"frame_{frame_index:04d}.png"
                plotter.screenshot(str(png_path))
                saved_png_frames.append(str(png_path))

            if interactive_preview:
                plotter.update()
    except Exception as exc:
        error = str(exc)
    finally:
        plotter.close()

    mean_curvature = float(np.mean([m["curvature_proxy_mean"] for m in frame_metrics])) if frame_metrics else 0.0
    max_curvature = float(np.max([m["curvature_proxy_max"] for m in frame_metrics])) if frame_metrics else 0.0

    passed = error is None and video_path.exists() and video_path.stat().st_size > 0 and len(saved_png_frames) > 0

    return {
        "stage": "7_realtime_manifold_rendering",
        "driver": "pyvista",
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "saved_png_frames": saved_png_frames,
        "frame_metrics": {
            "count": len(frame_metrics),
            "curvature_proxy_mean": mean_curvature,
            "curvature_proxy_max": max_curvature,
        },
        "duration_sec": round(time.time() - started_at, 3),
        "passed": passed,
        "error": error,
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper2spacecode Stage 6/7 with PyVista")
    parser.add_argument(
        "--formula-text",
        default="Companion matrix A of x^3 - x^2 - x - 1 with Tribonacci foliation dynamics.",
        help="Paper formula text used to derive Stage 6 contract",
    )
    parser.add_argument("--frames", type=int, default=72, help="Number of animation frames")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate")
    parser.add_argument("--grid-resolution", type=int, default=144, help="Manifold grid resolution")
    parser.add_argument("--save-png-every", type=int, default=6, help="Save one png every N frames")
    parser.add_argument("--interactive-preview", action="store_true", help="Show interactive real-time preview")
    parser.add_argument(
        "--output-dir",
        default="h2q_project/reports/paper2spacecode_stage7",
        help="Output directory for Stage 6/7 artifacts",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage6_contract = build_stage6_formula_contract(
        paper_formula_text=args.formula_text,
        frames=args.frames,
        fps=args.fps,
        grid_resolution=args.grid_resolution,
    )
    stage6_path = output_dir / "stage6_formula_contract.json"
    stage6_path.write_text(json.dumps(stage6_contract, ensure_ascii=False, indent=2), encoding="utf-8")

    stage7_report = render_stage7_pyvista_demo(
        stage6_contract=stage6_contract,
        output_dir=output_dir,
        save_png_every=args.save_png_every,
        interactive_preview=args.interactive_preview,
    )
    stage7_path = output_dir / "stage7_render_report.json"
    stage7_path.write_text(json.dumps(stage7_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "stage6_contract_path": str(stage6_path),
                "stage7_report_path": str(stage7_path),
                "passed": bool(stage7_report.get("passed")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if stage7_report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
