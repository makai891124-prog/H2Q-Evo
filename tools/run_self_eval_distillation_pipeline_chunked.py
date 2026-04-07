#!/usr/bin/env python3
"""Run self-eval distillation in chunks and keep the best chunk result as latest."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run(cmd: List[str], timeout_sec: int) -> Dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
        )
        return {
            "returncode": int(proc.returncode),
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
            "elapsed_sec": time.time() - start,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join(((exc.stderr or "") + "\nTIMEOUT").splitlines()[-20:]),
            "elapsed_sec": time.time() - start,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked distillation pipeline runner")
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--teacher-provider", choices=["deepseek", "heuristic"], default="deepseek")
    parser.add_argument("--teacher-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--max-samples-per-chunk", type=int, default=60)
    parser.add_argument("--synthetic-min-size-per-chunk", type=int, default=60)
    parser.add_argument("--include-valid-prompts", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--output-prefix", default="self_eval_distillation_pipeline_chunked")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    chunk_runs: List[Dict[str, Any]] = []
    best_payload: Dict[str, Any] = {}
    best_delta = -1e18

    chunk_count = max(1, int(args.chunks))
    for i in range(1, chunk_count + 1):
        cmd = [
            str(PY),
            "tools/run_self_eval_distillation_pipeline.py",
            "--sessions",
            str(max(1, int(args.sessions))),
            "--teacher-provider",
            args.teacher_provider,
            "--teacher-key-file",
            args.teacher_key_file,
            "--max-samples",
            str(max(1, int(args.max_samples_per_chunk))),
            "--synthetic-min-size",
            str(max(0, int(args.synthetic_min_size_per_chunk))),
        ]
        if args.include_valid_prompts:
            cmd.append("--include-valid-prompts")

        run_info = _run(cmd, timeout_sec=int(args.timeout_sec))
        latest = _read_json(REPORTS / "self_eval_distillation_pipeline_latest.json")
        delta = _f((latest.get("metrics") or {}).get("delta_schema_valid_rate", 0.0), 0.0)

        chunk_row = {
            "chunk": i,
            "cmd": cmd,
            "returncode": run_info["returncode"],
            "elapsed_sec": run_info["elapsed_sec"],
            "delta_schema_valid_rate": delta,
            "stdout_tail": run_info["stdout_tail"],
            "stderr_tail": run_info["stderr_tail"],
        }
        chunk_runs.append(chunk_row)

        if run_info["returncode"] == 0 and latest and delta >= best_delta:
            best_delta = delta
            best_payload = latest

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "chunks": chunk_count,
            "sessions": int(args.sessions),
            "teacher_provider": args.teacher_provider,
            "max_samples_per_chunk": int(args.max_samples_per_chunk),
            "synthetic_min_size_per_chunk": int(args.synthetic_min_size_per_chunk),
        },
        "chunk_runs": chunk_runs,
        "best_delta_schema_valid_rate": best_delta if best_payload else 0.0,
        "all_chunks_succeeded": all(int(r.get("returncode", 1)) == 0 for r in chunk_runs),
    }

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Self-Eval Distillation Pipeline Chunked",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- chunks: `{chunk_count}`",
        f"- best_delta_schema_valid_rate: `{summary['best_delta_schema_valid_rate']}`",
        f"- all_chunks_succeeded: `{summary['all_chunks_succeeded']}`",
        "",
        "## Chunk Runs",
    ]
    for r in chunk_runs:
        lines.append(
            f"- chunk={r['chunk']}, rc={r['returncode']}, elapsed_sec={r['elapsed_sec']:.1f}, delta={r['delta_schema_valid_rate']:+.6f}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    # Keep compatibility: bootstrap readers consume self_eval_distillation_pipeline_latest.json.
    if best_payload:
        (REPORTS / "self_eval_distillation_pipeline_latest.json").write_text(
            json.dumps(best_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")

    return 0 if summary["all_chunks_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
