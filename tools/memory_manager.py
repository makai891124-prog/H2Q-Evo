#!/usr/bin/env python3
"""Memory management utilities for long-running AGI evolution processes.

Prevents buffer overflow and memory leaks in continuous CLI runs.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """Maintains a bounded in-memory buffer with automatic archival to disk."""

    def __init__(self, max_size: int = 100, archive_dir: Optional[Path] = None):
        """
        Args:
            max_size: Maximum number of items to keep in memory.
            archive_dir: Directory for archived items. If None, items are dropped.
        """
        self.max_size = max(1, max_size)
        self.archive_dir = archive_dir
        if archive_dir:
            archive_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self.archive_count = 0
        self.dropped_count = 0

    def append(self, item: Dict[str, Any]) -> None:
        """Add item to buffer; older items are archived or dropped."""
        # If buffer is at max size, archive the oldest item before adding new one.
        if len(self.buffer) >= self.max_size and self.archive_dir:
            try:
                oldest = dict(self.buffer[0])  # Make a copy before deque rotates
                self._archive_item(oldest, self.archive_count)
                self.archive_count += 1
            except Exception as e:
                logger.warning(f"Failed to archive item: {e}")
                self.dropped_count += 1

        self.buffer.append(item)

    def _archive_item(self, item: Dict[str, Any], idx: int) -> None:
        """Write item to disk archive."""
        if not self.archive_dir:
            return
        archive_file = self.archive_dir / f"archived_item_{idx:06d}.json"
        try:
            archive_file.write_text(
                json.dumps(item, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to write archive file {archive_file}: {e}")

    def get_all(self) -> List[Dict[str, Any]]:
        """Return current buffer contents."""
        return list(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        return {
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "archived_count": self.archive_count,
            "dropped_count": self.dropped_count,
        }


class MetricsRotator:
    """Rotates historical metrics to prevent unbounded growth."""

    def __init__(
        self,
        max_records: int = 1000,
        checkpoint_size: int = 200,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            max_records: Maximum metrics records to keep in memory.
            checkpoint_size: Write to disk after this many new records.
        """
        self.max_records = max(1, max_records)
        self.checkpoint_size = max(1, checkpoint_size)
        self.records: Deque[Dict[str, Any]] = deque(maxlen=self.max_records)
        self.checkpoint_idx = 0
        self.checkpoint_dir = checkpoint_dir or Path(".")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._pending_checkpoint: List[Dict[str, Any]] = []

    def append(self, record: Dict[str, Any]) -> Optional[Path]:
        """Add record; returns checkpoint file if limit exceeded."""
        self.records.append(record)
        self._pending_checkpoint.append(record)
        checkpoint_file = None

        # Checkpoint periodically.
        if len(self._pending_checkpoint) >= self.checkpoint_size:
            checkpoint_file = self._write_checkpoint()
            self.checkpoint_idx += len(self._pending_checkpoint)
            self._pending_checkpoint.clear()

        return checkpoint_file

    def _write_checkpoint(self) -> Path:
        """Write checkpoint file with metrics since last checkpoint."""
        ts = int(time.time())
        checkpoint_file = self.checkpoint_dir / f"metrics_checkpoint_{ts}.json"
        try:
            checkpoint_file.write_text(
                json.dumps(self._pending_checkpoint, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"Metrics checkpoint written: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
        return checkpoint_file

    def get_records(self) -> List[Dict[str, Any]]:
        """Return current records."""
        return list(self.records)

    def get_stats(self) -> Dict[str, Any]:
        """Return rotator statistics."""
        return {
            "current_count": len(self.records),
            "max_records": self.max_records,
            "checkpoint_idx": self.checkpoint_idx,
        }


class StreamingJSONWriter:
    """Writes large JSON structures incrementally to prevent memory buildup."""

    def __init__(self, filepath: Path, mode: str = "w"):
        """
        Args:
            filepath: Output file path.
            mode: "w" for write, "a" for append.
        """
        self.filepath = filepath
        self.mode = mode
        self.file = None
        self.first_item = True

    def __enter__(self):
        self.file = open(self.filepath, self.mode, encoding="utf-8")
        self.file.write("{\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.write("\n}\n")
            self.file.close()

    def write_key_value(self, key: str, value: Any) -> None:
        """Write a top-level key-value pair."""
        if not self.file:
            raise RuntimeError("StreamingJSONWriter not used as context manager")

        if not self.first_item:
            self.file.write(",\n")
        self.first_item = False

        self.file.write(f'  "{key}": ')
        json.dump(value, self.file, ensure_ascii=False, separators=(",", ":"))
        self.file.flush()

    def write_iterable_array(self, key: str, values: Iterable[Any]) -> None:
        """Write a top-level key whose value is an array streamed item-by-item."""
        if not self.file:
            raise RuntimeError("StreamingJSONWriter not used as context manager")

        if not self.first_item:
            self.file.write(",\n")
        self.first_item = False

        self.file.write(f'  "{key}": [')
        first = True
        for item in values:
            if not first:
                self.file.write(",")
            first = False
            json.dump(item, self.file, ensure_ascii=False, separators=(",", ":"))
        self.file.write("]")
        self.file.flush()


class LogBufferFlusher:
    """Periodically flushes logger handlers to prevent log buffer buildup."""

    def __init__(self, flush_interval: float = 10.0):
        """
        Args:
            flush_interval: Flush every N seconds.
        """
        self.flush_interval = flush_interval
        self.last_flush = time.time()

    def maybe_flush(self) -> bool:
        """Flush if interval exceeded; returns True if flushed."""
        now = time.time()
        if now - self.last_flush >= self.flush_interval:
            self.flush()
            self.last_flush = now
            return True
        return False

    @staticmethod
    def flush() -> None:
        """Flush all logger handlers."""
        for handler in logging.root.handlers:
            if hasattr(handler, "flush"):
                try:
                    handler.flush()
                except Exception as e:
                    logger.warning(f"Failed to flush handler {handler}: {e}")


class MemoryWatchdog:
    """Monitors process memory usage and alerts on threshold breach."""

    def __init__(self, warn_mb: float = 500.0, critical_mb: float = 1000.0):
        """
        Args:
            warn_mb: Warn if memory exceeds this (MB).
            critical_mb: Critical alert if memory exceeds this (MB).
        """
        self.warn_mb = warn_mb
        self.critical_mb = critical_mb
        self.last_check = time.time()
        self.check_interval = 30.0

    def check_and_warn(self) -> Dict[str, Any]:
        """Check memory usage; return status."""
        now = time.time()
        if now - self.last_check < self.check_interval:
            return {"checked": False}

        self.last_check = now
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            status = "ok"
            if mem_mb >= self.critical_mb:
                status = "critical"
                logger.error(f"🚨 CRITICAL: Memory usage {mem_mb:.1f} MB exceeds {self.critical_mb} MB")
            elif mem_mb >= self.warn_mb:
                status = "warn"
                logger.warning(f"⚠️ Memory usage {mem_mb:.1f} MB exceeds warn threshold {self.warn_mb} MB")
            return {
                "checked": True,
                "memory_mb": mem_mb,
                "status": status,
            }
        except ImportError:
            logger.debug("psutil not available; memory watchdog disabled")
            return {"checked": False, "reason": "psutil-not-available"}
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return {"checked": False, "reason": str(e)}


def enable_unbuffered_logging() -> None:
    """Configure Python and logging for unbuffered output."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Reconfigure logging to use unbuffered handlers.
    for handler in logging.root.handlers:
        if hasattr(handler, "stream"):
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.__stdout__
            if hasattr(handler, "flush"):
                try:
                    handler.flush()
                except Exception as e:
                    logger.debug(f"Failed to flush handler: {e}")
