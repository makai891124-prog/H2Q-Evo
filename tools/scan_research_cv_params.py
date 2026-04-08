#!/usr/bin/env python3
"""Grid-scan research aggregation weights/thresholds and persist best config."""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "tools" / "run_research_aggregation_cross_validation.py"


@dataclass
class ScanRow:
    score: float
    loo_min: float
    loo_std: float
    robust: bool
    cfg: Dict[str, Any]
    artifact: str


def _run_once(cfg: Dict[str, Any]) -> ScanRow:
    cmd = [
        str(PY),
        str(SCRIPT),
        "--config-file",
        "reports/research_cv_tuned_config_tmp.json",
        "--w-distill",
        str(cfg["weights"]["distill_gain"]),
        "--w-consistency",
        str(cfg["weights"]["consistency_quality"]),
        "--w-robustness",
        str(cfg["weights"]["robustness_30_vs_50"]),
        "--w-public",
        str(cfg["weights"]["public_validation"]),
        "--w-formal",
        str(cfg["weights"]["formal_closure"]),
        "--thr-aggregate",
        str(cfg["thresholds"]["aggregate"]),
        "--thr-loo-min",
        str(cfg["thresholds"]["loo_min"]),
        "--thr-loo-std",
        str(cfg["thresholds"]["loo_std"]),
    ]
    subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    latest = json.loads((REPORTS / "research_aggregation_cross_validation_latest.json").read_text(encoding="utf-8"))
    agg = float(((latest.get("aggregate_effectiveness") or {}).get("score", 0.0) or 0.0))
    cv = latest.get("cross_validation") or {}
    loo_min = float(cv.get("min_score", 0.0) or 0.0)
    loo_std = float(cv.get("std_score", 1.0) or 1.0)
    robust = bool(((latest.get("proof_argument") or {}).get("robust_claim", False)))
    return ScanRow(
        score=agg,
        loo_min=loo_min,
        loo_std=loo_std,
        robust=robust,
        cfg=cfg,
        artifact=str(REPORTS / "research_aggregation_cross_validation_latest.json"),
    )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Keep scan compact to avoid long runtime.
    weight_sets: List[Dict[str, float]] = [
        {
            "distill_gain": 0.30,
            "consistency_quality": 0.20,
            "robustness_30_vs_50": 0.10,
            "public_validation": 0.25,
            "formal_closure": 0.15,
        },
        {
            "distill_gain": 0.25,
            "consistency_quality": 0.20,
            "robustness_30_vs_50": 0.15,
            "public_validation": 0.25,
            "formal_closure": 0.15,
        },
        {
            "distill_gain": 0.20,
            "consistency_quality": 0.25,
            "robustness_30_vs_50": 0.15,
            "public_validation": 0.25,
            "formal_closure": 0.15,
        },
        {
            "distill_gain": 0.20,
            "consistency_quality": 0.20,
            "robustness_30_vs_50": 0.10,
            "public_validation": 0.30,
            "formal_closure": 0.20,
        },
    ]
    thresholds = list(
        itertools.product(
            [0.85, 0.88],   # aggregate
            [0.80, 0.82],   # loo_min
            [0.07, 0.06],   # loo_std
        )
    )

    rows: List[ScanRow] = []
    for w in weight_sets:
        for agg_thr, loo_thr, std_thr in thresholds:
            cfg = {
                "weights": w,
                "thresholds": {
                    "aggregate": agg_thr,
                    "loo_min": loo_thr,
                    "loo_std": std_thr,
                },
            }
            rows.append(_run_once(cfg))

    # Prefer robust solutions, then higher score, then better LOO stats.
    rows.sort(key=lambda r: (1 if r.robust else 0, r.score, r.loo_min, -r.loo_std), reverse=True)
    best = rows[0]

    ts = int(time.time())
    out_json = REPORTS / f"research_cv_param_scan_{ts}.json"
    latest_json = REPORTS / "research_cv_param_scan_latest.json"
    best_cfg = REPORTS / "research_cv_tuned_config_latest.json"

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_candidates": len(rows),
        "best": {
            "score": best.score,
            "loo_min": best.loo_min,
            "loo_std": best.loo_std,
            "robust": best.robust,
            "cfg": best.cfg,
        },
        "top5": [
            {
                "score": r.score,
                "loo_min": r.loo_min,
                "loo_std": r.loo_std,
                "robust": r.robust,
                "cfg": r.cfg,
            }
            for r in rows[:5]
        ],
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    best_cfg.write_text(json.dumps(best.cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scan JSON: {out_json}")
    print(f"Latest scan JSON: {latest_json}")
    print(f"Tuned config: {best_cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
