#!/usr/bin/env python3
"""Trusted LoRA fine-tuning on local distillation corpus.

- Uses PEFT LoRA for parameter-efficient adaptation.
- Falls back gracefully with a clear error if peft is unavailable.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import shutil

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
    return [x for x in out if x]


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
                min_new_tokens=min(24, max_new_tokens),
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"prompt": p, "output": text})
    return rows


def _extract_completion(prompt: str, output: str) -> str:
    p = str(prompt or "").strip()
    o = str(output or "").strip()
    if p and o.startswith(p):
        # Keep empty completion as empty instead of falling back to full echoed prompt.
        return o[len(p) :].lstrip("\n\r\t :：-\"").strip()

    # Fallback: tolerate small whitespace/newline variations around prompt prefix.
    if p and o:
        p_norm = " ".join(p.split())
        head = o[: max(len(p) + 32, 96)]
        head_norm = " ".join(head.split())
        if head_norm.startswith(p_norm):
            idx = o.find(p)
            if idx >= 0:
                return o[idx + len(p) :].lstrip("\n\r\t :：-\"").strip()

        idx = o.find(p)
        if 0 <= idx <= 24:
            return o[idx + len(p) :].lstrip("\n\r\t :：-\"").strip()

    return o


def _required_fields_from_prompt(prompt: str) -> List[str]:
    p = str(prompt or "").lower()
    keys: List[str] = []
    for candidate in ["diagnosis", "action", "confidence", "metric", "failure mode", "next_step", "root_cause"]:
        if candidate in p:
            keys.append(candidate)
    return keys


def _score_replay_quality(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    if not samples:
        return {
            "score": 0.0,
            "structure_rate": 0.0,
            "density_rate": 0.0,
            "echo_rate": 1.0,
            "non_empty_rate": 0.0,
            "samples": [],
        }

    per_sample: List[Dict[str, Any]] = []
    total = len(samples)
    non_empty_hits = 0
    structure_hits = 0
    density_acc = 0.0
    echo_acc = 0.0
    score_acc = 0.0

    for row in samples:
        prompt = str(row.get("prompt", ""))
        output = str(row.get("output", ""))
        completion = _extract_completion(prompt, output)

        non_empty = len(completion) >= 24
        if non_empty:
            non_empty_hits += 1

        prompt_l = prompt.lower()
        wants_json = "json" in prompt_l or "字段" in prompt_l or "field" in prompt_l
        completion_s = completion.strip()
        structure_ok = True
        if wants_json:
            structure_ok = completion_s.startswith("{") and completion_s.endswith("}")
        if structure_ok:
            structure_hits += 1

        required = _required_fields_from_prompt(prompt)
        if required:
            field_hit = sum(1 for k in required if k.lower() in completion_s.lower())
            field_score = field_hit / max(1, len(required))
        else:
            field_score = 1.0

        if completion_s:
            alnum = sum(1 for ch in completion_s if ch.isalnum())
            density = alnum / max(1, len(completion_s))
            density = min(1.0, density / 0.45)
        else:
            density = 0.0
        density_acc += density

        if completion_s:
            sim = difflib.SequenceMatcher(None, completion_s.lower(), prompt.strip().lower()).ratio()
        else:
            # Empty completion is already penalized by non_empty/structure; do not mark as pure echo.
            sim = 0.0
        echo_acc += sim
        echo_penalty = min(1.0, sim)

        raw_score = (
            0.30 * (1.0 if non_empty else 0.0)
            + 0.25 * (1.0 if structure_ok else 0.0)
            + 0.25 * field_score
            + 0.20 * density
        )
        sample_score = raw_score * (1.0 - 0.40 * echo_penalty)
        score_acc += sample_score

        per_sample.append(
            {
                "prompt": prompt,
                "completion": completion,
                "non_empty": bool(non_empty),
                "structure_ok": bool(structure_ok),
                "field_score": float(field_score),
                "density": float(density),
                "echo_similarity": float(sim),
                "score": float(sample_score),
            }
        )

    avg_score = score_acc / max(1, total)
    structure_rate = structure_hits / max(1, total)
    density_rate = density_acc / max(1, total)
    echo_rate = echo_acc / max(1, total)
    non_empty_rate = non_empty_hits / max(1, total)

    return {
        "score": float(avg_score),
        "structure_rate": float(structure_rate),
        "density_rate": float(density_rate),
        "echo_rate": float(echo_rate),
        "non_empty_rate": float(non_empty_rate),
        "samples": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train trusted nano LoRA from local distillation corpus")
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
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="c_attn,c_proj")
    parser.add_argument("--output-prefix", default="trusted_nano_lora")
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise SystemExit(
            "Missing peft dependency. Install with: .venv/bin/pip install peft"
        ) from exc

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

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    target_modules = [x.strip() for x in str(args.target_modules).split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=max(1, int(args.lora_r)),
        lora_alpha=max(1, int(args.lora_alpha)),
        lora_dropout=max(0.0, float(args.lora_dropout)),
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=max(0.0, float(args.weight_decay)),
    )
    total_steps = max(1, (len(chunks) + max(1, args.batch_size) - 1) // max(1, args.batch_size)) * max(1, args.epochs)
    warmup_steps = int(max(0.0, min(0.9, float(args.warmup_ratio))) * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses: List[float] = []
    steps = 0
    stop_early = False

    best_loss = float("inf")
    best_step = 0
    no_improve = 0

    ts = int(time.time())
    best_ckpt_dir = REPORTS / f"{args.output_prefix}_weights_best_{ts}"
    final_ckpt_dir = REPORTS / f"{args.output_prefix}_weights_{ts}"
    latest_ckpt = REPORTS / f"{args.output_prefix}_weights_latest"

    def _save_best_checkpoint() -> None:
        if best_ckpt_dir.exists():
            shutil.rmtree(best_ckpt_dir)
        best_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_ckpt_dir)
        tokenizer.save_pretrained(best_ckpt_dir)

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

            cur = float(losses[-1])
            if cur < (best_loss - float(args.early_stop_min_delta)):
                best_loss = cur
                best_step = steps
                no_improve = 0
                _save_best_checkpoint()
            else:
                no_improve += 1

            if int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
                stop_early = True
                break
        if stop_early:
            break

    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)

    if not best_ckpt_dir.exists():
        # Safety fallback for edge cases where no best checkpoint was captured.
        shutil.copytree(final_ckpt_dir, best_ckpt_dir, dirs_exist_ok=True)
        best_loss = losses[-1] if losses else best_loss
        best_step = steps

    # Only advance latest when best checkpoint is truly better than initial loss.
    loss_initial = losses[0] if losses else None
    improved_vs_initial = bool(loss_initial is not None and best_loss < float(loss_initial))
    latest_strategy = "kept_existing"
    if improved_vs_initial:
        if latest_ckpt.exists():
            shutil.rmtree(latest_ckpt)
        shutil.copytree(best_ckpt_dir, latest_ckpt)
        latest_strategy = "updated_with_best"
    elif not latest_ckpt.exists():
        shutil.copytree(best_ckpt_dir, latest_ckpt)
        latest_strategy = "initialized_with_best"

    samples_final = _sample_generations(
        model,
        tokenizer,
        prompts=[
            "请给出一个结构化JSON格式的自我评估摘要：",
            "Describe one concrete next-step experiment to improve robustness:",
        ],
        device=device,
        max_new_tokens=80,
    )

    # Replay best checkpoint to validate that the chosen checkpoint is usable.
    replay_samples: List[Dict[str, str]] = []
    replay_pass = False
    replay_score = 0.0
    replay_quality = {
        "score": 0.0,
        "structure_rate": 0.0,
        "density_rate": 0.0,
        "echo_rate": 1.0,
        "non_empty_rate": 0.0,
        "samples": [],
    }
    replay_quality_pass = False
    try:
        from peft import PeftModel

        replay_base = AutoModelForCausalLM.from_pretrained(args.model_name)
        replay_model = PeftModel.from_pretrained(replay_base, str(best_ckpt_dir)).to(device)
        replay_samples = _sample_generations(
            replay_model,
            tokenizer,
            prompts=[
                "请输出一个JSON，字段: diagnosis, action, confidence。",
                "Describe one robustness experiment with metric and expected failure mode.",
            ],
            device=device,
            max_new_tokens=64,
        )
        non_empty = sum(1 for r in replay_samples if len(_extract_completion(r.get("prompt", ""), r.get("output", ""))) >= 24)
        replay_score = non_empty / max(1, len(replay_samples))
        replay_quality = _score_replay_quality(replay_samples)
        replay_quality_pass = bool(
            replay_quality.get("score", 0.0) >= 0.52
            and replay_quality.get("structure_rate", 0.0) >= 0.50
            and replay_quality.get("non_empty_rate", 0.0) >= 0.50
            and replay_quality.get("echo_rate", 1.0) <= 0.92
        )
        replay_pass = bool(improved_vs_initial and replay_score >= 0.5 and replay_quality_pass)
    except Exception:
        replay_pass = False

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "dataset": str(dataset_path),
        "corpus_text_count": len(corpus),
        "token_count": len(all_ids),
        "chunk_count": len(chunks),
        "epochs": int(args.epochs),
        "steps": steps,
        "stopped_early": bool(stop_early),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "warmup_steps": warmup_steps,
        "loss_initial": loss_initial,
        "loss_final": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "weights_dir": str(final_ckpt_dir),
        "weights_latest_dir": str(latest_ckpt),
        "latest_strategy": latest_strategy,
        "best_checkpoint": {
            "dir": str(best_ckpt_dir),
            "loss": best_loss if best_loss != float("inf") else None,
            "step": int(best_step),
            "improved_vs_initial": bool(improved_vs_initial),
            "replay": {
                "score": float(replay_score),
                "samples": replay_samples,
                "quality": replay_quality,
            },
            "replay_pass": bool(replay_pass),
            "replay_quality_pass": bool(replay_quality_pass),
        },
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": target_modules,
        },
        "samples": samples_final,
    }

    out_json = REPORTS / f"{args.output_prefix}_training_{ts}.json"
    latest_json = REPORTS / f"{args.output_prefix}_training_latest.json"
    out_md = REPORTS / f"{args.output_prefix}_training_{ts}.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Trusted Nano LoRA Training",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- model_name: `{payload['model_name']}`",
        f"- dataset: `{payload['dataset']}`",
        f"- token_count: `{payload['token_count']}`",
        f"- chunk_count: `{payload['chunk_count']}`",
        f"- steps: `{payload['steps']}`",
        f"- warmup_steps: `{payload['warmup_steps']}`",
        f"- loss_initial: `{payload['loss_initial']}`",
        f"- loss_final: `{payload['loss_final']}`",
        f"- best_loss: `{payload['best_checkpoint']['loss']}`",
        f"- best_step: `{payload['best_checkpoint']['step']}`",
        f"- replay_pass: `{payload['best_checkpoint']['replay_pass']}`",
        f"- weights_latest_dir: `{payload['weights_latest_dir']}`",
        f"- latest_strategy: `{payload['latest_strategy']}`",
        f"- lora_target_modules: `{','.join(target_modules)}`",
        "",
        "## Outward Samples",
    ]
    for s in samples_final:
        lines.append(f"- prompt: `{s['prompt']}`")
        lines.append(f"- output: `{s['output']}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md_path = REPORTS / f"{args.output_prefix}_training_latest.md"
    latest_md_path.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"JSON: {out_json}")
    print(f"Latest JSON: {latest_json}")
    print(f"MD: {out_md}")
    print(f"Latest MD: {latest_md_path}")
    print(f"Weights dir: {final_ckpt_dir}")
    print(f"Weights latest dir: {latest_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
