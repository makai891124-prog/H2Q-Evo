#!/usr/bin/env python3
"""Memory and buffering helpers for long-running CLI/services."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


_LOGGER = logging.getLogger(__name__)


def append_with_limit(items: List[Any], value: Any, max_items: int) -> None:
    """Append a value and keep only the latest ``max_items`` entries."""
    items.append(value)
    if max_items > 0 and len(items) > max_items:
        del items[: len(items) - max_items]


def json_write_round_payload(
    out_path: Path,
    round_payload: Dict[str, Any],
    trust_summary: Dict[str, Any],
    trust_report: Optional[Path],
    stream: Optional[TextIO] = None,
) -> None:
    """Write a round report JSON without building a large serialized buffer."""
    handle = stream
    should_close = False
    if handle is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle = out_path.open("w", encoding="utf-8")
        should_close = True
    try:
        handle.write("{\n")
        handle.write('  "meta": ')
        json.dump(
            {
                "created_at_utc": round_payload.get("timestamp_utc", ""),
                "trust_report": str(trust_report) if trust_report else "",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write(",\n  \"trust\": ")
        json.dump(trust_summary, handle, ensure_ascii=False, indent=2)
        handle.write(",\n  \"round\": ")
        json.dump(round_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n}\n")
        handle.flush()
    finally:
        if should_close:
            handle.close()


def flush_logger_handlers(logger_obj: logging.Logger) -> None:
    """Flush all handlers attached to a logger."""
    for handler in logger_obj.handlers:
        try:
            handler.flush()
        except Exception as exc:
            _LOGGER.debug("Failed to flush logger handler %r: %s", handler, exc)
