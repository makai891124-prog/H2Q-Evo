"""Voice I/O broker for half-duplex local speech workflows.

This module provides a low-intrusion service layer to wire together:
- text input queue (API and optional microphone ASR),
- response generation callback,
- optional local TTS output.

Design goals:
- Optional dependencies only; service remains usable without ASR packages.
- Threaded execution to avoid blocking FastAPI request handlers.
- Simple status/reporting hooks for incremental rollout.
"""

from __future__ import annotations

import importlib
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Protocol

logger = logging.getLogger("voice-io-service")


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class VoiceIOConfig:
    queue_max_size: int = 64
    history_size: int = 100
    microphone_enabled: bool = False
    tts_enabled: bool = True
    asr_backend: str = "speech_recognition"
    asr_use_cloud_fallback: bool = False
    asr_language: str = "zh-CN"
    asr_timeout_sec: float = 2.0
    asr_phrase_time_limit_sec: float = 6.0

    @classmethod
    def from_env(cls) -> "VoiceIOConfig":
        return cls(
            queue_max_size=max(8, int(os.getenv("VOICE_IO_QUEUE_MAX_SIZE", "64"))),
            history_size=max(20, int(os.getenv("VOICE_IO_HISTORY_SIZE", "100"))),
            microphone_enabled=_as_bool(os.getenv("VOICE_IO_MICROPHONE_ENABLED"), False),
            tts_enabled=_as_bool(os.getenv("VOICE_IO_TTS_ENABLED"), True),
            asr_backend=os.getenv("VOICE_IO_ASR_BACKEND", "speech_recognition").strip().lower(),
            asr_use_cloud_fallback=_as_bool(
                os.getenv("VOICE_IO_ASR_USE_CLOUD_FALLBACK"),
                False,
            ),
            asr_language=os.getenv("VOICE_IO_ASR_LANGUAGE", "zh-CN"),
            asr_timeout_sec=max(0.5, float(os.getenv("VOICE_IO_ASR_TIMEOUT_SEC", "2.0"))),
            asr_phrase_time_limit_sec=max(
                1.0, float(os.getenv("VOICE_IO_ASR_PHRASE_TIME_LIMIT_SEC", "6.0"))
            ),
        )


class TextToSpeechBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def speak(self, text: str) -> None:
        ...


class ASRBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def listen_once(self, *, timeout_sec: float, phrase_time_limit_sec: float, language: str) -> Optional[str]:
        ...


class NoopTTSBackend:
    name = "noop"

    def is_available(self) -> bool:
        return True

    def speak(self, text: str) -> None:
        _ = text


class NoopASRBackend:
    name = "noop"

    def is_available(self) -> bool:
        return False

    def listen_once(
        self,
        *,
        timeout_sec: float,
        phrase_time_limit_sec: float,
        language: str,
    ) -> Optional[str]:
        _ = timeout_sec
        _ = phrase_time_limit_sec
        _ = language
        return None


class MacOSSayTTSBackend:
    name = "macos-say"

    def __init__(self) -> None:
        self._say_binary = shutil.which("say")

    def is_available(self) -> bool:
        return self._say_binary is not None

    def speak(self, text: str) -> None:
        if not text.strip() or not self._say_binary:
            return
        try:
            subprocess.run(
                [self._say_binary, text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("TTS playback failed")


class SpeechRecognitionASRBackend:
    name = "speech-recognition"

    def __init__(self, *, use_cloud_fallback: bool = False) -> None:
        self._use_cloud_fallback = use_cloud_fallback
        self._sr = None
        self._recognizer = None
        self._microphone = None
        try:
            speech_recognition = importlib.import_module("speech_recognition")
            self._sr = speech_recognition
            self._recognizer = speech_recognition.Recognizer()
        except Exception:
            logger.info(
                "speech_recognition package unavailable; microphone ASR disabled until installed"
            )

    def is_available(self) -> bool:
        return self._sr is not None and self._recognizer is not None

    def _ensure_microphone(self) -> None:
        if self._microphone is None and self._sr is not None:
            self._microphone = self._sr.Microphone()

    def _recognize_audio(self, audio: Any, language: str) -> Optional[str]:
        if not self.is_available():
            return None

        recognizer = self._recognizer
        try:
            return recognizer.recognize_sphinx(audio, language=language).strip()
        except Exception:
            if not self._use_cloud_fallback:
                return None
        try:
            return recognizer.recognize_google(audio, language=language).strip()
        except Exception:
            return None

    def transcribe_file(self, file_path: str, *, language: str) -> Optional[str]:
        if not self.is_available():
            return None
        sr = self._sr
        recognizer = self._recognizer
        try:
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
        except Exception:
            logger.exception("Failed to read audio file for ASR: %s", file_path)
            return None
        return self._recognize_audio(audio, language)

    def listen_once(self, *, timeout_sec: float, phrase_time_limit_sec: float, language: str) -> Optional[str]:
        if not self.is_available():
            return None
        self._ensure_microphone()
        if self._microphone is None:
            return None

        sr = self._sr
        recognizer = self._recognizer
        with self._microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = recognizer.listen(
                    source,
                    timeout=timeout_sec,
                    phrase_time_limit=phrase_time_limit_sec,
                )
            except sr.WaitTimeoutError:
                return None
        return self._recognize_audio(audio, language)


class VoiceIOService:
    """Threaded voice broker for local ASR + response + TTS."""

    def __init__(
        self,
        *,
        prompt_handler: Callable[[str], str],
        config: Optional[VoiceIOConfig] = None,
        tts_backend: Optional[TextToSpeechBackend] = None,
        asr_backend: Optional[ASRBackend] = None,
    ) -> None:
        self.config = config or VoiceIOConfig.from_env()
        self._prompt_handler = prompt_handler

        selected_tts = tts_backend or MacOSSayTTSBackend()
        if not selected_tts.is_available():
            selected_tts = NoopTTSBackend()
        self._tts_backend: TextToSpeechBackend = selected_tts

        selected_asr = asr_backend or self._build_asr_backend(self.config)
        self._asr_backend: ASRBackend = selected_asr

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

        self._request_count = 0
        self._error_count = 0

        self._input_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=self.config.queue_max_size
        )
        self._output_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=self.config.queue_max_size
        )
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.history_size)

        self._broker_thread: Optional[threading.Thread] = None
        self._microphone_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._broker_thread is not None and self._broker_thread.is_alive()

    def configure(
        self,
        *,
        microphone_enabled: Optional[bool] = None,
        tts_enabled: Optional[bool] = None,
        asr_backend: Optional[str] = None,
        asr_use_cloud_fallback: Optional[bool] = None,
        asr_language: Optional[str] = None,
    ) -> None:
        rebuild_asr = False
        with self._state_lock:
            if microphone_enabled is not None:
                self.config.microphone_enabled = bool(microphone_enabled)
            if tts_enabled is not None:
                self.config.tts_enabled = bool(tts_enabled)
            if asr_backend:
                normalized = asr_backend.strip().lower()
                if normalized and normalized != self.config.asr_backend:
                    self.config.asr_backend = normalized
                    rebuild_asr = True
            if asr_use_cloud_fallback is not None:
                fallback = bool(asr_use_cloud_fallback)
                if fallback != self.config.asr_use_cloud_fallback:
                    self.config.asr_use_cloud_fallback = fallback
                    rebuild_asr = True
            if asr_language:
                self.config.asr_language = asr_language
            if rebuild_asr:
                self._asr_backend = self._build_asr_backend(self.config)

    def start(self) -> bool:
        with self._state_lock:
            if self.is_running:
                return False
            self._stop_event.clear()
            self._broker_thread = threading.Thread(
                target=self._broker_loop,
                daemon=True,
                name="voice-io-broker",
            )
            self._broker_thread.start()
            self._microphone_thread = threading.Thread(
                target=self._microphone_loop,
                daemon=True,
                name="voice-io-microphone",
            )
            self._microphone_thread.start()
        logger.info("VoiceIO service started")
        return True

    def stop(self) -> bool:
        with self._state_lock:
            if not self.is_running:
                return False
            self._stop_event.set()
            broker_thread = self._broker_thread
            microphone_thread = self._microphone_thread

        if broker_thread:
            broker_thread.join(timeout=2.0)
        if microphone_thread:
            microphone_thread.join(timeout=2.0)

        with self._state_lock:
            self._broker_thread = None
            self._microphone_thread = None
        logger.info("VoiceIO service stopped")
        return True

    def submit_text(self, text: str, *, source: str = "api", speak_response: bool = True) -> str:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        if not self.is_running:
            raise RuntimeError("VoiceIO service is not running")

        request_id = uuid.uuid4().hex
        payload = {
            "request_id": request_id,
            "text": text.strip(),
            "source": source,
            "speak_response": bool(speak_response),
            "timestamp": time.time(),
        }
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full as exc:
            raise RuntimeError("VoiceIO input queue is full") from exc
        return request_id

    def drain_responses(self, *, max_items: int = 20) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        max_items = max(1, max_items)
        for _ in range(max_items):
            try:
                out.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def get_status(self) -> Dict[str, Any]:
        with self._state_lock:
            history_items = list(self._history)
            return {
                "running": self.is_running,
                "microphone_enabled": self.config.microphone_enabled,
                "tts_enabled": self.config.tts_enabled,
                "asr_backend_config": self.config.asr_backend,
                "asr_use_cloud_fallback": self.config.asr_use_cloud_fallback,
                "asr_language": self.config.asr_language,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "input_queue_size": self._input_queue.qsize(),
                "output_queue_size": self._output_queue.qsize(),
                "asr_backend": self._asr_backend.name,
                "asr_available": self._asr_backend.is_available(),
                "tts_backend": self._tts_backend.name,
                "tts_available": self._tts_backend.is_available(),
                "recent_history": history_items[-10:],
            }

    def _build_asr_backend(self, config: VoiceIOConfig) -> ASRBackend:
        backend_key = (config.asr_backend or "speech_recognition").strip().lower()
        if backend_key in {"none", "disabled", "off", "noop"}:
            return NoopASRBackend()

        if backend_key in {"speech_recognition", "speech-recognition", "sr", "auto"}:
            return SpeechRecognitionASRBackend(
                use_cloud_fallback=config.asr_use_cloud_fallback,
            )

        logger.warning(
            "Unknown ASR backend '%s'; falling back to speech_recognition",
            backend_key,
        )
        return SpeechRecognitionASRBackend(
            use_cloud_fallback=config.asr_use_cloud_fallback,
        )

    def _push_output(self, item: Dict[str, Any]) -> None:
        try:
            self._output_queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self._output_queue.get_nowait()
                self._output_queue.put_nowait(item)
            except queue.Empty:
                pass

    def _broker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            request_id = payload["request_id"]
            transcript = payload["text"]
            source = payload["source"]
            speak_response = payload["speak_response"]
            started = time.perf_counter()

            response_text = ""
            error = None
            try:
                response_text = self._prompt_handler(transcript) or ""
                if self.config.tts_enabled and speak_response:
                    self._tts_backend.speak(response_text)
            except Exception as exc:
                self._error_count += 1
                error = str(exc)
                logger.exception("VoiceIO broker failed for request %s", request_id)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            item = {
                "request_id": request_id,
                "source": source,
                "transcript": transcript,
                "response_text": response_text,
                "latency_ms": elapsed_ms,
                "error": error,
                "timestamp": time.time(),
            }

            with self._state_lock:
                self._request_count += 1
                self._history.append(item)

            self._push_output(item)

    def _microphone_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.config.microphone_enabled:
                time.sleep(0.25)
                continue

            if not self._asr_backend.is_available():
                time.sleep(1.0)
                continue

            transcript = self._asr_backend.listen_once(
                timeout_sec=self.config.asr_timeout_sec,
                phrase_time_limit_sec=self.config.asr_phrase_time_limit_sec,
                language=self.config.asr_language,
            )
            if not transcript:
                continue
            try:
                self.submit_text(
                    transcript,
                    source="microphone",
                    speak_response=True,
                )
            except Exception:
                self._error_count += 1
                logger.exception("VoiceIO microphone ingestion failed")
                time.sleep(0.25)