"""PersonaPlex integration bridge.

This module provides a thin adapter to:
- probe local PersonaPlex runtime availability,
- build reproducible server/offline commands,
- optionally run offline evaluation jobs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional


def _pick_python() -> str:
    return os.getenv("PERSONAPLEX_PYTHON", "python3")


@dataclass
class PersonaPlexRunResult:
    ok: bool
    command: str
    exit_code: int
    elapsed_sec: float
    stdout: str
    stderr: str


@dataclass
class PersonaPlexConfig:
    python_executable: str = _pick_python()
    hf_repo: str = "nvidia/personaplex-7b-v1"
    default_device: str = "cpu"
    default_voice_prompt: str = "NATF0.pt"
    enable_cpu_offload: bool = True


class PersonaPlexBridge:
    """Command-level bridge for PersonaPlex server/offline workflows."""

    def __init__(self, config: Optional[PersonaPlexConfig] = None) -> None:
        self.config = config or PersonaPlexConfig()

    def build_server_command(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8998,
        ssl_dir: Optional[str] = None,
        device: Optional[str] = None,
        cpu_offload: Optional[bool] = None,
    ) -> list[str]:
        command = [
            self.config.python_executable,
            "-m",
            "moshi.server",
            "--host",
            host,
            "--port",
            str(port),
            "--device",
            device or self.config.default_device,
            "--hf-repo",
            self.config.hf_repo,
        ]
        use_cpu_offload = self.config.enable_cpu_offload if cpu_offload is None else cpu_offload
        if ssl_dir:
            command.extend(["--ssl", ssl_dir])
        if use_cpu_offload:
            command.append("--cpu-offload")
        return command

    def build_offline_command(
        self,
        *,
        input_wav: str,
        output_wav: str,
        output_text: str,
        text_prompt: str,
        voice_prompt: Optional[str] = None,
        voice_prompt_dir: Optional[str] = None,
        seed: int = 42424242,
        temp_audio: float = 0.8,
        temp_text: float = 0.7,
        topk_audio: int = 250,
        topk_text: int = 25,
        device: Optional[str] = None,
        cpu_offload: Optional[bool] = None,
    ) -> list[str]:
        command = [
            self.config.python_executable,
            "-m",
            "moshi.offline",
            "--input-wav",
            input_wav,
            "--output-wav",
            output_wav,
            "--output-text",
            output_text,
            "--text-prompt",
            text_prompt,
            "--voice-prompt",
            voice_prompt or self.config.default_voice_prompt,
            "--hf-repo",
            self.config.hf_repo,
            "--seed",
            str(seed),
            "--temp-audio",
            str(temp_audio),
            "--temp-text",
            str(temp_text),
            "--topk-audio",
            str(topk_audio),
            "--topk-text",
            str(topk_text),
            "--device",
            device or self.config.default_device,
        ]
        if voice_prompt_dir:
            command.extend(["--voice-prompt-dir", voice_prompt_dir])
        use_cpu_offload = self.config.enable_cpu_offload if cpu_offload is None else cpu_offload
        if use_cpu_offload:
            command.append("--cpu-offload")
        return command

    def command_to_string(self, command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)

    def check_runtime_available(self, *, timeout_sec: int = 10) -> Dict[str, object]:
        command = [self.config.python_executable, "-m", "moshi.offline", "--help"]
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.perf_counter() - start
            return {
                "available": proc.returncode == 0,
                "exit_code": proc.returncode,
                "elapsed_sec": elapsed,
                "command": self.command_to_string(command),
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return {
                "available": False,
                "exit_code": -1,
                "elapsed_sec": elapsed,
                "command": self.command_to_string(command),
                "error": str(exc),
            }

    def run_offline(self, *, command: list[str], timeout_sec: int = 0) -> PersonaPlexRunResult:
        start = time.perf_counter()
        timeout = None if timeout_sec <= 0 else timeout_sec
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start
            return PersonaPlexRunResult(
                ok=proc.returncode == 0,
                command=self.command_to_string(command),
                exit_code=proc.returncode,
                elapsed_sec=elapsed,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return PersonaPlexRunResult(
                ok=False,
                command=self.command_to_string(command),
                exit_code=-1,
                elapsed_sec=elapsed,
                stdout="",
                stderr=str(exc),
            )