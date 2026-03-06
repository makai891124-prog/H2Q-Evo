#!/usr/bin/env python3
"""Generate one-click KPI dashboard from latest run artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_latest_session(min_ts: Optional[int]) -> Optional[Path]:
    sessions = sorted(REPORTS.glob("trusted_local_agi_chat_session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if min_ts is None:
        return sessions[0] if sessions else None

    for p in sessions:
        try:
            stamp = int(p.stem.rsplit("_", 1)[-1])
        except Exception:
            stamp = 0
        if stamp >= min_ts:
            return p
    return sessions[0] if sessions else None


def _find_latest_release_gate(prefer_deepseek: bool) -> Optional[Path]:
    candidates: List[Path] = []
    if prefer_deepseek:
        preferred = REPORTS / "deepseek_assisted_release_gate_latest.json"
        if preferred.exists():
            return preferred
    for name in [
        "deepseek_assisted_release_gate_latest.json",
        "release_gate_latest.json",
        "openclaw_bridge_release_gate_latest.json",
    ]:
        p = REPORTS / name
        if p.exists():
            candidates.append(p)
    return _latest_by_mtime(candidates)


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(1.0, float(value)))
    fill = int(round(v * width))
    return "#" * fill + "-" * (width - fill)


def _extract_epoch_from_name(path: Path, prefix: str) -> int:
    stem = path.stem
    key = f"{prefix}_"
    if not stem.startswith(key):
        return 0
    try:
        return int(stem[len(key) :])
    except Exception:
        return 0


def _load_trend_history(output_prefix: str, limit: int = 20) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    latest_name = f"{output_prefix}_latest.json"
    for p in REPORTS.glob(f"{output_prefix}_*.json"):
        if p.name == latest_name:
            continue
        epoch = _extract_epoch_from_name(p, output_prefix)
        if epoch <= 0:
            continue
        try:
            payload = _load_json(p)
            kpis = payload.get("kpis") or {}
            points.append(
                {
                    "epoch": epoch,
                    "label": datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%m-%d %H:%M"),
                    "strict_json_attempt_count": float(kpis.get("strict_json_attempt_count", 0.0) or 0.0),
                    "hard_fail_count": float(kpis.get("hard_fail_count", 0.0) or 0.0),
                    "fallback_ratio_self_eval": float(kpis.get("fallback_ratio_self_eval", 0.0) or 0.0),
                    "teacher_assist_dependency_ratio": float(kpis.get("teacher_assist_dependency_ratio", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    points.sort(key=lambda x: x["epoch"])
    if len(points) > limit:
        points = points[-limit:]
    return points


def _plot_dashboard_chart(history: List[Dict[str, Any]], out_png: Path, out_latest_png: Path) -> None:
    labels = [p["label"] for p in history]
    fallback = [p["fallback_ratio_self_eval"] for p in history]
    teacher = [p["teacher_assist_dependency_ratio"] for p in history]
    strict_counts = [p["strict_json_attempt_count"] for p in history]
    hard_fail_counts = [p["hard_fail_count"] for p in history]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    ax1 = axes[0]
    x = range(len(labels))
    ax1.plot(x, fallback, marker="o", linewidth=2.0, color="#c0392b", label="fallback_ratio_self_eval")
    ax1.plot(x, teacher, marker="o", linewidth=2.0, color="#2e86c1", label="teacher_assist_dependency_ratio")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("ratio")
    ax1.set_title("One-Click KPI Trend (Ratios)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax2 = axes[1]
    width = 0.38
    x2 = list(x)
    ax2.bar([i - width / 2 for i in x2], strict_counts, width=width, color="#27ae60", label="strict_json_attempt_count")
    ax2.bar([i + width / 2 for i in x2], hard_fail_counts, width=width, color="#8e44ad", label="hard_fail_count")
    ax2.set_ylabel("count")
    ax2.set_title("One-Click KPI Trend (Counts)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.savefig(out_png, dpi=160)
    shutil.copy2(out_png, out_latest_png)
    plt.close(fig)


def _collect_session_kpis(session_payload: Dict[str, Any]) -> Dict[str, Any]:
    transcript = session_payload.get("transcript") or []
    strict_json_attempt_count = 0
    hard_fail_count = 1 if bool(((session_payload.get("meta") or {}).get("hard_fail_triggered", False))) else 0

    self_eval_total = 0
    self_eval_fallback = 0

    for turn in transcript:
        runtime = turn.get("runtime") or {}
        route = str(runtime.get("route", ""))
        schema = runtime.get("schema") or {}

        if bool(schema.get("strict_json_pre_fallback_attempted", False)):
            strict_json_attempt_count += 1

        if bool(schema.get("required", False)):
            self_eval_total += 1
            if route == "local_fallback_template":
                self_eval_fallback += 1

    fallback_ratio_self_eval = _safe_ratio(self_eval_fallback, self_eval_total)
    return {
        "strict_json_attempt_count": strict_json_attempt_count,
        "hard_fail_count": hard_fail_count,
        "self_eval_total": self_eval_total,
        "self_eval_fallback_count": self_eval_fallback,
        "fallback_ratio_self_eval": fallback_ratio_self_eval,
    }


def _collect_teacher_assist_dependency(release_payload: Dict[str, Any]) -> Dict[str, Any]:
    signals = release_payload.get("signals") or {}
    meta = release_payload.get("meta") or {}
    assist_enabled = bool(signals.get("assist_enabled", False))
    assist_calls = int(signals.get("assist_calls", 0) or 0)
    assist_success_calls = int(signals.get("assist_success_calls", 0) or 0)

    if not assist_enabled:
        ratio = 0.0
    else:
        # Dependency proxy: use success-call ratio to estimate effective reliance.
        ratio = _safe_ratio(assist_success_calls, max(assist_calls, 1))

    return {
        "assist_provider": str(meta.get("assist_provider", "none")),
        "assist_enabled": assist_enabled,
        "assist_calls": assist_calls,
        "assist_success_calls": assist_success_calls,
        "teacher_assist_dependency_ratio": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one-click KPI dashboard from latest artifacts")
    parser.add_argument("--output-prefix", default="one_click_kpi_dashboard")
    parser.add_argument("--run-ts", type=int, default=0, help="Prefer chat session files whose timestamp >= run-ts")
    parser.add_argument("--prefer-deepseek-assist", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    min_ts = args.run_ts if args.run_ts > 0 else None

    session_path = _find_latest_session(min_ts=min_ts)
    release_gate_path = _find_latest_release_gate(prefer_deepseek=args.prefer_deepseek_assist)

    if session_path is None:
        raise SystemExit("No trusted_local_agi_chat_session_*.json found in reports/")

    session_payload = _load_json(session_path)
    session_kpis = _collect_session_kpis(session_payload)

    assist_kpis: Dict[str, Any] = {
        "assist_provider": "none",
        "assist_enabled": False,
        "assist_calls": 0,
        "assist_success_calls": 0,
        "teacher_assist_dependency_ratio": 0.0,
    }
    if release_gate_path is not None:
        try:
            assist_kpis = _collect_teacher_assist_dependency(_load_json(release_gate_path))
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "session": str(session_path),
            "release_gate": str(release_gate_path) if release_gate_path else None,
        },
        "kpis": {
            "strict_json_attempt_count": int(session_kpis["strict_json_attempt_count"]),
            "hard_fail_count": int(session_kpis["hard_fail_count"]),
            "fallback_ratio_self_eval": float(session_kpis["fallback_ratio_self_eval"]),
            "teacher_assist_dependency_ratio": float(assist_kpis["teacher_assist_dependency_ratio"]),
        },
        "supporting": {
            "self_eval_total": int(session_kpis["self_eval_total"]),
            "self_eval_fallback_count": int(session_kpis["self_eval_fallback_count"]),
            "assist_provider": assist_kpis["assist_provider"],
            "assist_enabled": bool(assist_kpis["assist_enabled"]),
            "assist_calls": int(assist_kpis["assist_calls"]),
            "assist_success_calls": int(assist_kpis["assist_success_calls"]),
        },
    }

    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    out_png = REPORTS / f"{args.output_prefix}_{ts}.png"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"
    latest_png = REPORTS / f"{args.output_prefix}_latest.png"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, latest_json)

    history = _load_trend_history(args.output_prefix, limit=20)
    # Ensure current run is included in the chart even when history was empty.
    history.append(
        {
            "epoch": ts,
            "label": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M"),
            "strict_json_attempt_count": float(payload["kpis"]["strict_json_attempt_count"]),
            "hard_fail_count": float(payload["kpis"]["hard_fail_count"]),
            "fallback_ratio_self_eval": float(payload["kpis"]["fallback_ratio_self_eval"]),
            "teacher_assist_dependency_ratio": float(payload["kpis"]["teacher_assist_dependency_ratio"]),
        }
    )
    history = sorted(history, key=lambda x: x["epoch"])[-20:]
    _plot_dashboard_chart(history=history, out_png=out_png, out_latest_png=latest_png)

    k = payload["kpis"]
    lines = [
        "# One-Click KPI Dashboard",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- session: `{payload['sources']['session']}`",
        f"- release_gate: `{payload['sources']['release_gate']}`",
        "",
        "## KPI Metrics",
        f"- strict_json_attempt_count: `{k['strict_json_attempt_count']}`",
        f"- hard_fail_count: `{k['hard_fail_count']}`",
        f"- fallback_ratio_self_eval: `{k['fallback_ratio_self_eval']:.6f}`",
        f"- teacher_assist_dependency_ratio: `{k['teacher_assist_dependency_ratio']:.6f}`",
        "",
        "## Quick Visual",
        "- fallback_ratio_self_eval",
        f"  `{_bar(k['fallback_ratio_self_eval'])}`",
        "- teacher_assist_dependency_ratio",
        f"  `{_bar(k['teacher_assist_dependency_ratio'])}`",
        "",
        "## Trend Chart",
        "![One-Click KPI Trend](one_click_kpi_dashboard_latest.png)",
        "",
        "## Supporting Signals",
        f"- self_eval_total: `{payload['supporting']['self_eval_total']}`",
        f"- self_eval_fallback_count: `{payload['supporting']['self_eval_fallback_count']}`",
        f"- assist_provider: `{payload['supporting']['assist_provider']}`",
        f"- assist_enabled: `{payload['supporting']['assist_enabled']}`",
        f"- assist_calls: `{payload['supporting']['assist_calls']}`",
        f"- assist_success_calls: `{payload['supporting']['assist_success_calls']}`",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(out_md, latest_md)

    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")
    print(f"PNG: {out_png}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest MD: {latest_md}")
    print(f"Latest PNG: {latest_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
