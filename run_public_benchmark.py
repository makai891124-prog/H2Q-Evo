#!/usr/bin/env python3
"""Run public benchmark evaluation and save results."""
import json
import os
from datetime import datetime
from pathlib import Path

from h2q_project.h2q.agi.honest_capability_system import HonestCapabilityTester


def _load_env_key():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            os.environ.setdefault("GEMINI_API_KEY", line.split("=", 1)[1].strip().strip('"').strip("'"))
            return


def main():
    _load_env_key()
    os.environ.setdefault("H2Q_ALLOW_SMALL_PUBLIC_BENCH", "1")
    os.environ.setdefault("H2Q_PUBLIC_BENCH_MIN", "20")
    os.environ.setdefault("H2Q_PUBLIC_BENCH_N", "20")
    tester = HonestCapabilityTester()
    results = tester.run_honest_evaluation()

    out_dir = Path("benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"public_benchmark_{ts}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
