#!/usr/bin/env python3
"""paper2spacecode Stage 6/7/8 end-to-end runner.

Pipeline:
1) Run local voice/video cognition chain.
2) Build Stage 6 formula contract.
3) Render Stage 7 real-time manifold demo (png + mp4).
4) Integrate Stage 7 with chain video (Stage 8) and emit final report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from h2q_project.tools.paper2spacecode_pyvista_demo import (
    build_stage6_formula_contract,
    render_stage7_pyvista_demo,
)
from h2q_project.tools.run_voice_video_dialogue_chain_test import run_full_chain


def _run_command(cmd: List[str], *, timeout_sec: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )


def _resolve_ffmpeg_binary() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _compose_side_by_side_video(
    *,
    left_video: Path,
    right_video: Path,
    output_video: Path,
) -> None:
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
            "-map",
            "0:a?",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_video),
        ],
        timeout_sec=240,
    )


def _pick_formula_driver_text(chain_report: Dict[str, Any], fallback_text: str, override: str) -> str:
    if override.strip():
        return override.strip()

    audio_stage = chain_report.get("audio_stage") or {}
    video_stage = chain_report.get("video_stage") or {}
    for candidate in (
        audio_stage.get("transcript"),
        video_stage.get("transcript"),
        audio_stage.get("response_text"),
        fallback_text,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback_text


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper2spacecode Stage 6/7/8 end-to-end")
    parser.add_argument(
        "--input-text",
        default="hello h2q, render a tribonacci manifold from this paper formula and explain it",
        help="Input text for the voice-video cognition chain",
    )
    parser.add_argument("--formula-text", default="", help="Optional explicit paper formula text")
    parser.add_argument("--language", default="en-US", help="ASR language code")
    parser.add_argument("--play-tts", action="store_true", help="Play local TTS for chain responses")
    parser.add_argument("--use-cloud-fallback", action="store_true", help="Allow cloud ASR fallback")
    parser.add_argument("--use-tribonacci-bridge", action="store_true", help="Enable Tribonacci prompt bridge")
    parser.add_argument("--frames", type=int, default=72, help="Stage 7 frame count")
    parser.add_argument("--fps", type=int, default=24, help="Stage 7 video fps")
    parser.add_argument("--grid-resolution", type=int, default=144, help="Stage 7 manifold grid")
    parser.add_argument("--save-png-every", type=int, default=6, help="Save one png every N frames")
    parser.add_argument(
        "--output-dir",
        default="h2q_project/reports/paper2spacecode_e2e",
        help="Output directory for stage artifacts and final report",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    started_at = time.time()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voice_artifacts_dir = output_dir / "stage8_chain_media"
    voice_report_path = output_dir / "voice_video_dialogue_chain_report.json"

    chain_report = run_full_chain(
        input_text=args.input_text,
        language=args.language,
        play_tts=args.play_tts,
        use_cloud_fallback=args.use_cloud_fallback,
        use_tribonacci_bridge=args.use_tribonacci_bridge,
        output_report_path=voice_report_path,
        artifact_dir=voice_artifacts_dir,
    )

    formula_driver_text = _pick_formula_driver_text(
        chain_report=chain_report,
        fallback_text=args.input_text,
        override=args.formula_text,
    )

    stage6_contract = build_stage6_formula_contract(
        paper_formula_text=formula_driver_text,
        frames=args.frames,
        fps=args.fps,
        grid_resolution=args.grid_resolution,
    )
    stage6_path = output_dir / "stage6_formula_contract.json"
    stage6_path.write_text(json.dumps(stage6_contract, ensure_ascii=False, indent=2), encoding="utf-8")

    stage7_dir = output_dir / "stage7_spatial_demo"
    stage7_report = render_stage7_pyvista_demo(
        stage6_contract=stage6_contract,
        output_dir=stage7_dir,
        save_png_every=args.save_png_every,
        interactive_preview=False,
    )
    stage7_path = output_dir / "stage7_render_report.json"
    stage7_path.write_text(json.dumps(stage7_report, ensure_ascii=False, indent=2), encoding="utf-8")

    chain_video_path = None
    if isinstance(chain_report.get("artifacts"), dict):
        chain_video_raw = chain_report["artifacts"].get("video_input")
        if isinstance(chain_video_raw, str) and chain_video_raw:
            chain_video_path = Path(chain_video_raw)

    stage8_output_video = output_dir / "paper_formula_space_av_e2e.mp4"
    stage8_error = None
    stage8_combined_created = False
    try:
        if chain_video_path and chain_video_path.exists() and stage7_report.get("video_path"):
            _compose_side_by_side_video(
                left_video=chain_video_path,
                right_video=Path(stage7_report["video_path"]),
                output_video=stage8_output_video,
            )
            stage8_combined_created = stage8_output_video.exists() and stage8_output_video.stat().st_size > 0
    except Exception as exc:
        stage8_error = str(exc)

    stage8_report = {
        "stage": "8_av_integration_e2e",
        "inputs": {
            "voice_chain_report_path": str(voice_report_path),
            "stage7_render_report_path": str(stage7_path),
            "stage6_formula_contract_path": str(stage6_path),
        },
        "outputs": {
            "combined_video_path": str(stage8_output_video),
            "formula_driver_text": formula_driver_text,
        },
        "checks": {
            "voice_chain_passed": bool(chain_report.get("overall_passed")),
            "stage7_passed": bool(stage7_report.get("passed")),
            "combined_video_created": stage8_combined_created,
        },
        "passed": bool(
            chain_report.get("overall_passed")
            and stage7_report.get("passed")
            and stage8_combined_created
            and stage8_error is None
        ),
        "error": stage8_error,
    }
    stage8_path = output_dir / "stage8_e2e_report.json"
    stage8_path.write_text(json.dumps(stage8_report, ensure_ascii=False, indent=2), encoding="utf-8")

    final_report = {
        "stage6_contract_path": str(stage6_path),
        "stage7_report_path": str(stage7_path),
        "stage8_report_path": str(stage8_path),
        "voice_chain_report_path": str(voice_report_path),
        "overall_passed": bool(stage8_report["passed"]),
        "duration_sec": round(time.time() - started_at, 3),
    }
    final_path = output_dir / "paper2spacecode_e2e_report.json"
    final_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0 if final_report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
