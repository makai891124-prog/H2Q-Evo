#!/usr/bin/env python3
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    latest = reports / "autoresearch_h2q_bootstrap_fusion_latest.json"
    sweep = reports / "bootstrap_gate_sweep_177315.txt"

    cmd = [
        str(root / ".venv" / "bin" / "python"),
        "tools/run_autoresearch_h2q_bootstrap.py",
        "--execute",
        "--max-iterations",
        "1",
        "--timeout-sec",
        "900",
    ]

    rounds = []
    for i in range(1, 4):
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        elapsed = time.time() - t0

        if not latest.exists():
            raise RuntimeError(f"Round {i}: latest report missing: {latest}")

        data = json.loads(latest.read_text(encoding="utf-8"))
        experiments = data.get("experiments") or []
        exp = experiments[-1] if experiments else {}
        decision = exp.get("decision") or data.get("decision") or {}
        hard = decision.get("hard_gate") or {}

        benchmark_gain = decision.get("benchmark_gain")
        benchmark_threshold = (
            hard.get("benchmark_gain_threshold")
            if isinstance(hard, dict)
            else None
        )
        benchmark_pass = (
            hard.get("benchmark_ok")
            if isinstance(hard, dict)
            else None
        )
        status = exp.get("status") or decision.get("status")

        rounds.append(
            {
                "round": i,
                "status": status,
                "benchmark_gain": benchmark_gain,
                "benchmark_threshold": benchmark_threshold,
                "benchmark_pass": benchmark_pass,
                "lora_replay_pass": decision.get("lora_replay_pass"),
                "output_quality_pass": decision.get("replay_quality_pass"),
                "returncode": proc.returncode,
                "elapsed_sec": round(elapsed, 3),
            }
        )

    keep_count = sum(1 for r in rounds if r.get("status") == "keep")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_count": len(rounds),
        "keep_count": keep_count,
        "pass_rate": keep_count / len(rounds),
        "defaults_applied": {
            "hard_gate_benchmark_lookback": 12,
            "hard_gate_benchmark_quantile": 0.25,
            "hard_gate_benchmark_floor": 1e-5,
        },
        "sweep_evidence": str(sweep),
        "rounds": rounds,
    }

    ts = int(time.time())
    versioned = reports / f"gate_calibration_{ts}.json"
    latest_out = reports / "gate_calibration_latest.json"
    versioned.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"written_versioned={versioned}")
    print(f"written_latest={latest_out}")
    print(f"pass_rate={payload['pass_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
