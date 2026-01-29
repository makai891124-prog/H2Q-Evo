#!/usr/bin/env python3
"""Run public benchmarks against local H2Q server (no external APIs)."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests

from h2q_project.h2q.agi.standard_benchmarks import run_standard_benchmarks


SERVER_URL = os.getenv("H2Q_LOCAL_BENCH_URL", "http://127.0.0.1:8000/chat")
TIMEOUT_S = float(os.getenv("H2Q_LOCAL_BENCH_TIMEOUT", "20"))


def _parse_answer(payload_text: str, num_choices: int) -> int:
    """Extract a choice index from model output."""
    if not payload_text:
        return 0

    # Try to find digits in response
    digits = [int(ch) for ch in payload_text if ch.isdigit()]
    for d in digits:
        if 0 <= d < num_choices:
            return d

    # Fallback to 0
    return 0


def _build_prompt(question: str, choices: List[str]) -> str:
    return (
        "请从以下选项中选择最可能的正确答案，仅输出选项索引数字。\n"
        f"问题: {question}\n"
        f"选项: {choices}\n"
    )


def local_answer_func(question: str, choices: List[str]) -> Dict[str, Any]:
    prompt = _build_prompt(question, choices)
    try:
        resp = requests.post(
            SERVER_URL,
            json={"prompt": prompt, "max_tokens": 64, "temperature": 0.2, "use_das_arch": True},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        text = str(data.get("text", ""))
        idx = _parse_answer(text, len(choices))
        return {"selected": [idx], "ranked": [idx], "probs": []}
    except Exception as exc:
        return {"selected": [0], "error": str(exc)}


def main():
    os.environ.setdefault("H2Q_ALLOW_SMALL_PUBLIC_BENCH", "1")
    os.environ.setdefault("H2Q_PUBLIC_BENCH_MIN", "20")
    os.environ.setdefault("H2Q_PUBLIC_BENCH_N", "20")

    results = run_standard_benchmarks(
        answer_func=local_answer_func,
        n_per_benchmark=int(os.getenv("H2Q_PUBLIC_BENCH_N", "20")),
        public_only=True,
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "public_standard_benchmarks",
        "local_server_url": SERVER_URL,
        "anti_cheat": {
            "no_answer_key": True,
            "no_dataset_leak": True,
            "note": "仅提供问题与选项，不提供正确答案"
        },
        "results": results,
    }

    out_dir = Path("benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"local_public_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 本地公开基准结果已保存: {out_path}")


if __name__ == "__main__":
    main()
