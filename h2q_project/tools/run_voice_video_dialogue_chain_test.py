#!/usr/bin/env python3
"""Run a full local chain test for audio/video dialogue recognition.

Chain under test:
1) Local audio generation (macOS say)
2) Local ASR transcription (speech_recognition)
3) Cognitive reasoning generation (_voice_prompt_handler)
4) Local TTS response playback (macOS say)
5) Video path via ffmpeg audio extraction + ASR + reasoning + TTS
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from h2q_project.h2q.services.voice_io_service import (
    MacOSSayTTSBackend,
    SpeechRecognitionASRBackend,
)
from h2q_project.h2q_server import _voice_prompt_handler


def _run_command(cmd: list[str], *, timeout_sec: int = 120) -> subprocess.CompletedProcess[str]:
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
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _synthesize_audio_with_say(text: str, output_wav: Path) -> None:
    say_binary = shutil.which("say")
    if not say_binary:
        raise RuntimeError("macOS 'say' binary is unavailable")
    _run_command(
        [
            say_binary,
            "-o",
            str(output_wav),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            text,
        ],
        timeout_sec=30,
    )


def _create_video_with_audio(audio_path: Path, video_path: Path) -> None:
    ffmpeg = _resolve_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary is unavailable")

    _run_command(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=960x540:d=8",
            "-i",
            str(audio_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video_path),
        ],
        timeout_sec=120,
    )


def _extract_audio_from_video(video_path: Path, output_wav: Path) -> None:
    ffmpeg = _resolve_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary is unavailable")

    _run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ],
        timeout_sec=120,
    )


def _transcribe_audio_file(
    *,
    audio_path: Path,
    language: str,
    use_cloud_fallback: bool,
) -> Optional[str]:
    asr_backend = SpeechRecognitionASRBackend(use_cloud_fallback=use_cloud_fallback)
    if not asr_backend.is_available():
        return None
    return asr_backend.transcribe_file(str(audio_path), language=language)


def _run_cognitive_turn(transcript: str, *, play_tts: bool) -> str:
    response = _voice_prompt_handler(transcript)
    if play_tts:
        tts = MacOSSayTTSBackend()
        if tts.is_available():
            tts.speak(response)
    return response


def _stage_report(
    *,
    stage_name: str,
    media_path: Path,
    transcript: Optional[str],
    response_text: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    passed = bool(transcript and response_text and not error)
    return {
        "stage": stage_name,
        "media_path": str(media_path),
        "transcript": transcript,
        "response_text": response_text,
        "passed": passed,
        "error": error,
    }


def run_full_chain(
    *,
    input_text: str,
    language: str,
    play_tts: bool,
    use_cloud_fallback: bool,
    use_tribonacci_bridge: bool,
    output_report_path: Path,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    started_at = time.time()
    previous_bridge_flag = os.environ.get("VOICE_IO_ENABLE_TRIBONACCI_BRIDGE")
    os.environ["VOICE_IO_ENABLE_TRIBONACCI_BRIDGE"] = "1" if use_tribonacci_bridge else "0"

    try:
        with tempfile.TemporaryDirectory(prefix="h2q_voice_chain_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_input = tmp_path / "dialogue_input.wav"
            video_input = tmp_path / "dialogue_input.mp4"
            video_audio = tmp_path / "dialogue_video_audio.wav"

            report: Dict[str, Any] = {
                "input_text": input_text,
                "language": language,
                "use_cloud_fallback": use_cloud_fallback,
                "use_tribonacci_bridge": use_tribonacci_bridge,
                "tooling": {
                    "say": bool(shutil.which("say")),
                    "ffmpeg": bool(_resolve_ffmpeg_binary()),
                },
                "audio_stage": None,
                "video_stage": None,
                "artifacts": {},
                "overall_passed": False,
            }

            try:
                _synthesize_audio_with_say(input_text, audio_input)
                audio_transcript = _transcribe_audio_file(
                    audio_path=audio_input,
                    language=language,
                    use_cloud_fallback=use_cloud_fallback,
                )
                audio_response = (
                    _run_cognitive_turn(audio_transcript, play_tts=play_tts)
                    if audio_transcript
                    else None
                )
                report["audio_stage"] = _stage_report(
                    stage_name="audio",
                    media_path=audio_input,
                    transcript=audio_transcript,
                    response_text=audio_response,
                )
            except Exception as exc:
                report["audio_stage"] = _stage_report(
                    stage_name="audio",
                    media_path=audio_input,
                    transcript=None,
                    response_text=None,
                    error=str(exc),
                )

            try:
                _create_video_with_audio(audio_input, video_input)
                _extract_audio_from_video(video_input, video_audio)
                video_transcript = _transcribe_audio_file(
                    audio_path=video_audio,
                    language=language,
                    use_cloud_fallback=use_cloud_fallback,
                )
                video_response = (
                    _run_cognitive_turn(video_transcript, play_tts=play_tts)
                    if video_transcript
                    else None
                )
                report["video_stage"] = _stage_report(
                    stage_name="video",
                    media_path=video_input,
                    transcript=video_transcript,
                    response_text=video_response,
                )
            except Exception as exc:
                report["video_stage"] = _stage_report(
                    stage_name="video",
                    media_path=video_input,
                    transcript=None,
                    response_text=None,
                    error=str(exc),
                )

            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                if audio_input.exists():
                    saved_audio = artifact_dir / "dialogue_input.wav"
                    shutil.copy2(audio_input, saved_audio)
                    report["artifacts"]["audio_input"] = str(saved_audio)
                if video_input.exists():
                    saved_video = artifact_dir / "dialogue_input.mp4"
                    shutil.copy2(video_input, saved_video)
                    report["artifacts"]["video_input"] = str(saved_video)
                if video_audio.exists():
                    saved_video_audio = artifact_dir / "dialogue_video_audio.wav"
                    shutil.copy2(video_audio, saved_video_audio)
                    report["artifacts"]["video_audio"] = str(saved_video_audio)

            audio_ok = bool(report["audio_stage"] and report["audio_stage"]["passed"])
            video_ok = bool(report["video_stage"] and report["video_stage"]["passed"])
            report["overall_passed"] = audio_ok and video_ok
            report["duration_sec"] = round(time.time() - started_at, 3)

            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            output_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
    finally:
        if previous_bridge_flag is None:
            os.environ.pop("VOICE_IO_ENABLE_TRIBONACCI_BRIDGE", None)
        else:
            os.environ["VOICE_IO_ENABLE_TRIBONACCI_BRIDGE"] = previous_bridge_flag


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local audio/video dialogue chain test")
    parser.add_argument(
        "--input-text",
        default="hello h2q this is a local speech recognition cognitive chain test",
        help="Text to synthesize into input audio before ASR",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="ASR language code, e.g. en-US or zh-CN",
    )
    parser.add_argument(
        "--play-tts",
        action="store_true",
        help="Play response via local macOS say backend",
    )
    parser.add_argument(
        "--use-cloud-fallback",
        action="store_true",
        help="Allow Google Web Speech fallback if local Sphinx transcription fails",
    )
    parser.add_argument(
        "--use-tribonacci-bridge",
        action="store_true",
        help="Enable prompt augmentation via the Tribonacci SL(3,Z) bridge",
    )
    parser.add_argument(
        "--report",
        default="h2q_project/reports/voice_video_dialogue_chain_report.json",
        help="Path for JSON test report",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Optional directory to persist generated media artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report_path = Path(args.report).resolve()
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None

    report = run_full_chain(
        input_text=args.input_text,
        language=args.language,
        play_tts=args.play_tts,
        use_cloud_fallback=args.use_cloud_fallback,
        use_tribonacci_bridge=args.use_tribonacci_bridge,
        output_report_path=report_path,
        artifact_dir=artifact_dir,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("overall_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
