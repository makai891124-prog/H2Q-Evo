#!/usr/bin/env python3
"""Train a lightweight self-eval distillation adapter from teacher/student samples."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text.lower()) if len(t) >= 2]


def _choose_default(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "capability_boundaries": ["当前自评能力覆盖有限，需要持续蒸馏更新。"],
            "failure_risks": ["结构化JSON输出在复杂自我评估任务中仍存在失败风险。"],
            "improvement_plan": [
                {"action": "执行失败样本蒸馏并更新适配器", "metric": "schema_valid_rate >= 0.20"}
            ],
            "confidence": 0.55,
        }
    return samples[0]["teacher_normalized"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train self-eval distillation adapter model")
    parser.add_argument("--dataset", default="reports/self_eval_distill_dataset_latest.json")
    parser.add_argument("--output-prefix", default="self_eval_distill_model")
    parser.add_argument("--min-similarity", type=float, default=0.08)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples") or []
    valid_samples = [x for x in raw_samples if bool(x.get("teacher_valid")) and isinstance(x.get("teacher_normalized"), dict)]

    prompt_map: Dict[str, Dict[str, Any]] = {}
    entries: List[Dict[str, Any]] = []
    keyword_counter: Counter[str] = Counter()

    for item in valid_samples:
        prompt = str(item.get("prompt", "")).strip()
        norm = item.get("teacher_normalized")
        if not prompt or not isinstance(norm, dict):
            continue
        prompt_map[prompt] = norm
        keywords = _tokenize(prompt)
        keyword_counter.update(keywords)
        entries.append(
            {
                "prompt": prompt,
                "keywords": keywords,
                "teacher_json": norm,
                "failure_count": int(item.get("failure_count", 1)),
                "teacher_source": item.get("teacher_source", "unknown"),
            }
        )

    entries.sort(key=lambda x: x.get("failure_count", 0), reverse=True)
    default_template = _choose_default(valid_samples)

    ts = int(time.time())
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_latest.md"

    model = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "sample_count": len(valid_samples),
        "min_similarity": float(args.min_similarity),
        "prompt_map": prompt_map,
        "entries": entries,
        "default_template": default_template,
        "top_keywords": [k for k, _ in keyword_counter.most_common(50)],
    }

    out_json.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Self-Eval Distillation Adapter Model",
        "",
        f"- generated_at_utc: `{model['generated_at_utc']}`",
        f"- dataset: `{dataset_path}`",
        f"- valid_sample_count: `{len(valid_samples)}`",
        f"- prompt_map_size: `{len(prompt_map)}`",
        f"- min_similarity: `{model['min_similarity']:.4f}`",
        "",
        "## Top Keywords",
        "- " + ", ".join(model["top_keywords"][:20]),
    ]

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    latest_md.write_text(text, encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
