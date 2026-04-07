#!/usr/bin/env python3
"""Train a nano core seed from open-source crystal weights and distillation data.

Pipeline:
1) Load crystal embeddings extracted from an open-source model.
2) Build token-frequency signal from local distillation dataset.
3) Learn a compact seed bank (prototype vectors) by weighted clustering.
4) Save seed checkpoint for downstream local training/bootstrapping.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_texts(dataset: Dict[str, Any], limit: int) -> List[str]:
    out: List[str] = []
    for row in dataset.get("samples", [])[: max(1, limit)]:
        prompt = str(row.get("prompt", "")).strip()
        if prompt:
            out.append(prompt)
        teacher = row.get("teacher_normalized")
        if isinstance(teacher, dict):
            out.append(json.dumps(teacher, ensure_ascii=False, sort_keys=True))
    return out


def _token_hist(tokenizer: Any, texts: List[str], vocab_size: int) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.float32)
    for text in texts:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            ids = []
        for idx in ids:
            if isinstance(idx, int) and 0 <= idx < vocab_size:
                counts[idx] += 1.0
    return counts


def _weighted_kmeans(emb: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 12) -> torch.Tensor:
    # emb: [N, D], weights: [N]
    top_idx = torch.topk(weights, k=min(max(k * 8, k), emb.size(0))).indices
    x = emb[top_idx]
    w = weights[top_idx].clamp_min(1e-9)

    # deterministic init: choose most weighted points
    init_idx = torch.topk(w, k=min(k, x.size(0))).indices
    centers = x[init_idx].clone()

    for _ in range(max(2, iters)):
        # [M, K]
        dist = torch.cdist(x, centers)
        assign = torch.argmin(dist, dim=1)

        new_centers = []
        for j in range(centers.size(0)):
            mask = assign == j
            if int(mask.sum().item()) == 0:
                new_centers.append(centers[j])
                continue
            xj = x[mask]
            wj = w[mask].unsqueeze(1)
            cj = (xj * wj).sum(dim=0) / wj.sum()
            new_centers.append(cj)
        centers = torch.stack(new_centers, dim=0)

    return centers


def main() -> int:
    parser = argparse.ArgumentParser(description="Train nano core seed from crystal + distillation dataset")
    parser.add_argument("--crystal", default="h2q_qwen_crystal.pt")
    parser.add_argument("--dataset", default="reports/self_eval_distill_dataset_latest.json")
    parser.add_argument("--seed-dim", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--output-prefix", default="nano_core_seed")
    args = parser.parse_args()

    crystal_path = Path(args.crystal)
    if not crystal_path.is_absolute():
        crystal_path = ROOT / crystal_path
    if not crystal_path.exists():
        raise SystemExit(f"Crystal file not found: {crystal_path}")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise SystemExit(f"Distillation dataset not found: {dataset_path}")

    crystal = torch.load(crystal_path, map_location="cpu")
    emb = crystal.get("embeddings")
    if emb is None:
        raise SystemExit("Crystal is missing `embeddings`")
    emb = emb.float()

    model_name = str(crystal.get("source", "Qwen/Qwen2.5-0.5B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dataset = _load_json(dataset_path)
    texts = _collect_texts(dataset, limit=max(1, args.max_samples))
    if not texts:
        raise SystemExit("No usable texts from distillation dataset")

    vocab_size = int(min(emb.size(0), int(crystal.get("vocab_size", emb.size(0)))))
    hist = _token_hist(tokenizer, texts, vocab_size)

    # Keep only tokens observed in local distillation traces.
    active = hist > 0
    if int(active.sum().item()) < max(64, args.seed_count):
        # Fallback: include top-frequency tokens even when sparse.
        topk = torch.topk(hist, k=min(max(args.seed_count * 4, 256), hist.numel())).indices
        active = torch.zeros_like(active)
        active[topk] = True

    active_idx = torch.nonzero(active).squeeze(1)
    x = emb[active_idx]
    w = hist[active_idx].clamp_min(1.0)

    # Learn compact prototype bank in crystal space.
    centers = _weighted_kmeans(x, w, k=max(8, args.seed_count), iters=14)

    # Optional projection to smaller seed dimension.
    if args.seed_dim < centers.size(1):
        _, _, v = torch.pca_lowrank(centers, q=max(args.seed_dim, 4), center=True)
        proj = v[:, : args.seed_dim]
        seed = centers @ proj
    else:
        seed = centers

    seed = torch.nn.functional.normalize(seed, dim=1)

    ts = int(time.time())
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_pt = REPORTS / f"{args.output_prefix}_{ts}.pt"
    out_json = REPORTS / f"{args.output_prefix}_{ts}.json"
    latest_pt = REPORTS / f"{args.output_prefix}_latest.pt"
    latest_json = REPORTS / f"{args.output_prefix}_latest.json"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": model_name,
        "crystal": str(crystal_path),
        "dataset": str(dataset_path),
        "text_count": len(texts),
        "active_token_count": int(active_idx.numel()),
        "seed_shape": [int(seed.size(0)), int(seed.size(1))],
        "seed_dim": int(seed.size(1)),
        "seed_count": int(seed.size(0)),
    }

    torch.save(
        {
            "meta": payload,
            "seed_vectors": seed.half(),
            "active_token_ids": active_idx,
            "token_hist": hist,
        },
        out_pt,
    )
    latest_pt.write_bytes(out_pt.read_bytes())

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"SEED PT: {out_pt}")
    print(f"Latest PT: {latest_pt}")
    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
