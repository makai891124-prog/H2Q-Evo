#!/usr/bin/env python3
"""Realtime monitor for AGI self-evolution daemon outputs.

Collects periodic summaries from round/alert reports:
- current success rate
- assist hit rate
- assist token consumption
- alert count
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_ts(path: Path, prefix: str) -> int:
    name = path.stem
    # Example: agi_self_evolution_round_1772742224
    part = name.replace(prefix, "")
    try:
        return int(part)
    except Exception:
        return 0


def _list_round_files() -> List[Path]:
    files = sorted(
        REPORTS.glob("agi_self_evolution_round_*.json"),
        key=lambda p: _extract_ts(p, "agi_self_evolution_round_"),
    )
    return files


def _list_alert_files() -> List[Path]:
    files = sorted(
        REPORTS.glob("agi_self_evolution_alert_*.json"),
        key=lambda p: _extract_ts(p, "agi_self_evolution_alert_"),
    )
    return files


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics(round_files: List[Path], alert_files: List[Path], lookback_rounds: int) -> Dict[str, Any]:
    if lookback_rounds > 0 and len(round_files) > lookback_rounds:
        round_files = round_files[-lookback_rounds:]

    total_rounds = 0
    successful_rounds = 0

    assist_enabled_calls = 0
    assist_ok_calls = 0
    assist_total_tokens = 0
    failed_entry_total = 0
    fail_status_counts: Dict[str, int] = {}
    fail_assist_reason_counts: Dict[str, int] = {}

    for p in round_files:
        data = _load_json(p)
        round_obj = data.get("round", {})
        total_rounds += 1
        if bool(round_obj.get("success", False)):
            successful_rounds += 1

        for entry in round_obj.get("entries", []):
            assist = entry.get("runtime", {}).get("assist", {})
            if not bool(assist.get("enabled", False)):
                continue
            assist_enabled_calls += 1
            if bool(assist.get("ok", False)):
                assist_ok_calls += 1
            assist_total_tokens += int(assist.get("tokens", 0) or 0)

        analysis = round_obj.get("failure_analysis", {})
        failed_entry_total += int(analysis.get("failed_entry_count", 0) or 0)
        for k, v in analysis.get("status_counts", {}).items():
            key = str(k)
            fail_status_counts[key] = fail_status_counts.get(key, 0) + int(v)
        for k, v in analysis.get("assist_reason_counts", {}).items():
            key = str(k)
            fail_assist_reason_counts[key] = fail_assist_reason_counts.get(key, 0) + int(v)

    success_rate = (successful_rounds / total_rounds) if total_rounds > 0 else 0.0
    assist_hit_rate = (assist_ok_calls / assist_enabled_calls) if assist_enabled_calls > 0 else 0.0

    return {
        "round_count": total_rounds,
        "success_rounds": successful_rounds,
        "success_rate": success_rate,
        "assist_enabled_calls": assist_enabled_calls,
        "assist_ok_calls": assist_ok_calls,
        "assist_hit_rate": assist_hit_rate,
        "assist_total_tokens": assist_total_tokens,
        "alert_count": len(alert_files),
        "failed_entry_total": failed_entry_total,
        "top_fail_statuses": dict(sorted(fail_status_counts.items(), key=lambda x: (-x[1], x[0]))[:5]),
        "top_fail_assist_reasons": dict(sorted(fail_assist_reason_counts.items(), key=lambda x: (-x[1], x[0]))[:5]),
    }


def _compute_hourly_diagnosis(round_files: List[Path], lookback_rounds: int, recent_hours: int = 8) -> Dict[str, Any]:
    if lookback_rounds > 0 and len(round_files) > lookback_rounds:
        round_files = round_files[-lookback_rounds:]

    buckets: Dict[str, Dict[str, Any]] = {}
    for p in round_files:
        ts = _extract_ts(p, "agi_self_evolution_round_")
        if ts <= 0:
            continue
        hour_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
        b = buckets.setdefault(
            hour_key,
            {
                "hour_utc": hour_key,
                "rounds": 0,
                "success_rounds": 0,
                "assist_tokens": 0,
                "failed_entries": 0,
                "fail_status_counts": {},
                "fail_assist_reason_counts": {},
            },
        )

        data = _load_json(p)
        round_obj = data.get("round", {})
        b["rounds"] += 1
        if bool(round_obj.get("success", False)):
            b["success_rounds"] += 1

        tokens_this_round = 0
        for entry in round_obj.get("entries", []):
            assist = entry.get("runtime", {}).get("assist", {})
            tokens_this_round += int(assist.get("tokens", 0) or 0)
        b["assist_tokens"] += tokens_this_round

        analysis = round_obj.get("failure_analysis", {})
        b["failed_entries"] += int(analysis.get("failed_entry_count", 0) or 0)
        for k, v in analysis.get("status_counts", {}).items():
            kk = str(k)
            b["fail_status_counts"][kk] = b["fail_status_counts"].get(kk, 0) + int(v)
        for k, v in analysis.get("assist_reason_counts", {}).items():
            kk = str(k)
            b["fail_assist_reason_counts"][kk] = b["fail_assist_reason_counts"].get(kk, 0) + int(v)

    hours = sorted(buckets.values(), key=lambda x: x["hour_utc"])
    if recent_hours > 0 and len(hours) > recent_hours:
        hours = hours[-recent_hours:]

    for h in hours:
        rounds = max(1, int(h["rounds"]))
        succ = int(h["success_rounds"])
        h["success_rate"] = float(succ) / float(rounds)
        h["token_efficiency"] = float(h["assist_tokens"]) / float(max(1, succ))
        h["top_fail_status"] = next(
            iter(sorted(h["fail_status_counts"].items(), key=lambda x: (-x[1], x[0]))),
            ("none", 0),
        )
        h["top_fail_assist_reason"] = next(
            iter(sorted(h["fail_assist_reason_counts"].items(), key=lambda x: (-x[1], x[0]))),
            ("none", 0),
        )

    trends: List[Dict[str, Any]] = []
    for idx in range(1, len(hours)):
        prev = hours[idx - 1]
        cur = hours[idx]
        trends.append(
            {
                "from_hour": prev["hour_utc"],
                "to_hour": cur["hour_utc"],
                "success_rate_delta": float(cur["success_rate"] - prev["success_rate"]),
                "token_efficiency_delta": float(cur["token_efficiency"] - prev["token_efficiency"]),
                "top_fail_status": cur["top_fail_status"],
                "top_fail_assist_reason": cur["top_fail_assist_reason"],
            }
        )

    return {
        "hours": hours,
        "trends": trends,
        "recent_hours": recent_hours,
    }


def _render_hourly_markdown(diag: Dict[str, Any]) -> str:
    lines = [
        "# AGI Hourly Trend Diagnosis",
        "",
        f"- recent_hours: `{diag.get('recent_hours', 0)}`",
        "",
        "## Hour Buckets",
        "",
        "| Hour(UTC) | Rounds | Success Rate | Token Efficiency(tokens/success) | Top Fail Status | Top Fail Assist Reason |",
        "|---|---:|---:|---:|---|---|",
    ]

    for h in diag.get("hours", []):
        top_status = h.get("top_fail_status", ("none", 0))
        top_reason = h.get("top_fail_assist_reason", ("none", 0))
        lines.append(
            f"| {h['hour_utc']} | {h['rounds']} | {h['success_rate']:.2%} | {h['token_efficiency']:.1f} | {top_status[0]}({top_status[1]}) | {top_reason[0]}({top_reason[1]}) |"
        )

    lines.extend(["", "## Hourly Deltas", "", "| From -> To | Success Rate Delta | Token Efficiency Delta |", "|---|---:|---:|"])
    for t in diag.get("trends", []):
        lines.append(
            f"| {t['from_hour']} -> {t['to_hour']} | {t['success_rate_delta']:+.2%} | {t['token_efficiency_delta']:+.1f} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(snapshot: Dict[str, Any]) -> str:
    m = snapshot["metrics"]
    lines = [
        "# AGI Realtime Monitor Snapshot",
        "",
        f"- timestamp_utc: `{snapshot['timestamp_utc']}`",
        f"- rounds: `{m['round_count']}`",
        f"- success_rate: `{m['success_rate']:.2%}` ({m['success_rounds']}/{m['round_count']})",
        f"- assist_hit_rate: `{m['assist_hit_rate']:.2%}` ({m['assist_ok_calls']}/{m['assist_enabled_calls']})",
        f"- assist_total_tokens: `{m['assist_total_tokens']}`",
        f"- alert_count: `{m['alert_count']}`",
        f"- failed_entry_total: `{m['failed_entry_total']}`",
    ]
    if m.get("top_fail_statuses"):
        lines.append(f"- top_fail_statuses: `{m['top_fail_statuses']}`")
    if m.get("top_fail_assist_reasons"):
        lines.append(f"- top_fail_assist_reasons: `{m['top_fail_assist_reasons']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime AGI evolution monitor")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--output-prefix", default="agi_realtime_monitor")
    parser.add_argument("--lookback-rounds", type=int, default=0, help="0 means all history")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    out_jsonl = REPORTS / f"{args.output_prefix}_{ts}.jsonl"
    out_latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_latest_md = REPORTS / f"{args.output_prefix}_latest.md"
    out_hourly_json = REPORTS / f"{args.output_prefix}_hourly_diagnosis_latest.json"
    out_hourly_md = REPORTS / f"{args.output_prefix}_hourly_diagnosis_latest.md"

    i = 0
    while True:
        i += 1
        round_files = _list_round_files()
        alert_files = _list_alert_files()
        metrics = _compute_metrics(round_files, alert_files, lookback_rounds=max(0, args.lookback_rounds))
        hourly_diag = _compute_hourly_diagnosis(round_files, lookback_rounds=max(0, args.lookback_rounds), recent_hours=8)

        snapshot = {
            "timestamp_utc": _now_utc(),
            "cycle": i,
            "interval_seconds": args.interval_seconds,
            "lookback_rounds": max(0, args.lookback_rounds),
            "metrics": metrics,
            "files": {
                "latest_round": str(round_files[-1]) if round_files else "",
                "latest_alert": str(alert_files[-1]) if alert_files else "",
            },
        }

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        out_latest_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        out_latest_md.write_text(_render_markdown(snapshot), encoding="utf-8")
        out_hourly_json.write_text(json.dumps(hourly_diag, ensure_ascii=False, indent=2), encoding="utf-8")
        out_hourly_md.write_text(_render_hourly_markdown(hourly_diag), encoding="utf-8")

        print(
            "[monitor] cycle={cycle} success_rate={success:.2%} assist_hit_rate={hit:.2%} tokens={tokens} alerts={alerts}".format(
                cycle=i,
                success=metrics["success_rate"],
                hit=metrics["assist_hit_rate"],
                tokens=metrics["assist_total_tokens"],
                alerts=metrics["alert_count"],
            ),
            flush=True,
        )

        if args.cycles > 0 and i >= args.cycles:
            break

        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    main()
