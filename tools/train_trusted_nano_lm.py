#!/usr/bin/env python3
"""Trusted tiny LM weight training on local distillation corpus.

- Downloads a small open-source LM (default: sshleifer/tiny-gpt2)
- Trains model weights with causal LM objective on local corpus
- Saves finetuned checkpoint and outward generation samples
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_corpus(dataset: Dict[str, Any], max_samples: int) -> List[str]:
    out: List[str] = []
    for item in (dataset.get("samples") or [])[: max(1, max_samples)]:
        prompt = str(item.get("prompt", "")).strip()
        if prompt:
            out.append(prompt)
        teacher = item.get("teacher_normalized")
        if isinstance(teacher, dict):
            out.append(json.dumps(teacher, ensure_ascii=False, sort_keys=True))
    return [t for t in out if t]


def _batchify(ids: List[int], block_size: int, stride: int) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    if len(ids) < block_size:
        return chunks
    for i in range(0, max(1, len(ids) - block_size), max(1, stride)):
        c = ids[i : i + block_size]
        if len(c) == block_size:
            chunks.append(torch.tensor(c, dtype=torch.long))
    return chunks


def _sample_generations(model: Any, tokenizer: Any, prompts: List[str], device: torch.device, max_new_tokens: int = 64) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    model.eval()
    for p in prompts:
        toks = tokenizer(p, return_tensors="pt", truncation=True, max_length=128)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"prompt": p, "output": text})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train trusted nano LM from local distillation corpus")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--dataset", default="reports/self_eval_distill_dataset_latest.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-prefix", default="trusted_nano_lm")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    ds = _load_json(dataset_path)
    corpus = _collect_corpus(ds, max_samples=args.max_samples)
    if not corpus:
        raise SystemExit("No text corpus from distillation dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    all_ids: List[int] = []
    for text in corpus:
        all_ids.extend(tokenizer.encode(text, add_special_tokens=True))

    chunks = _batchify(all_ids, block_size=args.block_size, stride=args.stride)
    if not chunks:
        raise SystemExit("Insufficient tokenized data for training chunks")

    random.seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=max(0.0, float(args.weight_decay)))
    total_steps = max(1, (len(chunks) + max(1, args.batch_size) - 1) // max(1, args.batch_size)) * max(1, args.epochs)
    warmup_steps = int(max(0.0, min(0.9, float(args.warmup_ratio))) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses: List[float] = []
    steps = 0
    for _ in range(max(1, args.epochs)):
        random.shuffle(chunks)
        for i in range(0, len(chunks), max(1, args.batch_size)):
            batch = chunks[i : i + args.batch_size]
            x = torch.stack(batch, dim=0).to(device)
            out = model(input_ids=x, labels=x)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.detach().cpu().item()))
            steps += 1

    ts = int(time.time())
    ckpt_dir = REPORTS / f"{args.output_prefix}_weights_{ts}"
    latest_ckpt = REPORTS / f"{args.output_prefix}_weights_latest"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Replace latest symlink-like directory via copy semantics.
    if latest_ckpt.exists():
        if latest_ckpt.is_dir():
            for p in latest_ckpt.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            latest_ckpt.unlink()
    latest_ckpt.mkdir(parents=True, exist_ok=True)
    for p in ckpt_dir.iterdir():
        if p.is_file():
            (latest_ckpt / p.name).write_bytes(p.read_bytes())

    samples = _sample_generations(
        model,
        tokenizer,
        prompts=[
            "请给出一个结构化JSON格式的自我评估摘要：",
            "Describe one concrete next-step experiment to improve robustness:",
        ],
        device=device,
        max_new_tokens=80,
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "dataset": str(dataset_path),
        "corpus_text_count": len(corpus),
        "token_count": len(all_ids),
        "chunk_count": len(chunks),
        "epochs": int(args.epochs),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "weights_dir": str(ckpt_dir),
        "weights_latest_dir": str(latest_ckpt),
        "samples": samples,
    }

    out_json = REPORTS / f"{args.output_prefix}_training_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_training_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_training_{ts}.md"
    latest_md = REPORTS / f"{args.output_prefix}_training_latest.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Trusted Nano LM Training",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- model_name: `{payload['model_name']}`",
        f"- dataset: `{payload['dataset']}`",
        f"- token_count: `{payload['token_count']}`",
        f"- chunk_count: `{payload['chunk_count']}`",
        f"- steps: `{payload['steps']}`",
        f"- loss_initial: `{payload['loss_initial']}`",
        f"- loss_final: `{payload['loss_final']}`",
        f"- weights_latest_dir: `{payload['weights_latest_dir']}`",
        "",
        "## Outward Samples",
    ]
    for s in samples:
        lines.append(f"- prompt: `{s['prompt']}`")
        lines.append(f"- output: `{s['output']}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md}")
    print(f"Weights dir: {ckpt_dir}")
    print(f"Weights latest dir: {latest_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
