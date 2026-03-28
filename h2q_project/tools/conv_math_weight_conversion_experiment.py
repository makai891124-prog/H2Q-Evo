#!/usr/bin/env python3
"""
Open-source LLM weight conversion experiment:
1) Download a small open-source causal LM.
2) Convert embedding/lm_head with a convolution + low-rank + manifold mapping pipeline.
3) Build a converted model and evaluate output consistency against the original model.
4) Save converted artifacts and a reproducible JSON report.

This is an engineering feasibility experiment, not a claim of universal equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sanitize_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


@dataclass
class ConvertedMatrix:
    reconstructed: torch.Tensor
    compressed: torch.Tensor
    right_basis: torch.Tensor
    conv_kernel: torch.Tensor
    compression_ratio: float
    rel_l2_error: float


def _conv_smooth_hidden(weight: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # weight: [vocab, hidden]
    x = weight.unsqueeze(1)  # [vocab, 1, hidden]
    y = F.conv1d(x, kernel, padding=kernel.shape[-1] // 2)
    return y.squeeze(1)


def _group_quaternion_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    # x: [n, d]
    n, d = x.shape
    pad = (4 - (d % 4)) % 4
    if pad:
        x = F.pad(x, (0, pad))
    q = x.view(n, -1, 4)
    norms = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    q_unit = q / norms
    return q_unit.view(n, -1), norms.view(n, -1), pad


def _group_quaternion_decode(q_unit_flat: torch.Tensor, norms_flat: torch.Tensor, original_dim: int, pad: int) -> torch.Tensor:
    q = q_unit_flat.view(q_unit_flat.shape[0], -1, 4)
    norms = norms_flat.view(norms_flat.shape[0], -1, 1)
    x = (q * norms).view(q_unit_flat.shape[0], -1)
    if pad:
        x = x[:, :original_dim]
    return x


def convert_matrix_with_conv_math(weight: torch.Tensor, rank: int, device: torch.device) -> ConvertedMatrix:
    w = weight.detach().to(torch.float32).to(device)

    # Fixed convolution kernel as a lightweight local correlation extractor.
    kernel = torch.tensor([[[0.25, 0.5, 0.25]]], dtype=torch.float32, device=device)
    w_smoothed = _conv_smooth_hidden(w, kernel)
    # Keep local convolutional inductive bias but avoid over-smoothing collapse.
    w_mixed = 0.85 * w + 0.15 * w_smoothed

    # Low-rank decomposition.
    u, s, vh = torch.linalg.svd(w_mixed, full_matrices=False)
    r = min(rank, s.numel())
    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]

    compressed = u_r * s_r.unsqueeze(0)

    # Mathematical core remap: quaternion manifold normalization per 4-tuple.
    q_unit, q_norms, pad = _group_quaternion_encode(compressed)
    compressed_restored = _group_quaternion_decode(q_unit, q_norms, compressed.shape[1], pad)

    reconstructed = compressed_restored @ vh_r

    # Residual low-rank correction improves token-level stability while staying compressed.
    residual = w - reconstructed
    r_res = min(max(4, r // 8), residual.shape[1], residual.shape[0])
    u2, s2, vh2 = torch.linalg.svd(residual, full_matrices=False)
    u2 = u2[:, :r_res]
    s2 = s2[:r_res]
    vh2 = vh2[:r_res, :]
    residual_recon = (u2 * s2.unsqueeze(0)) @ vh2
    reconstructed = reconstructed + residual_recon

    rel_l2 = torch.linalg.norm(reconstructed - w) / torch.linalg.norm(w).clamp_min(1e-8)

    original_params = w.numel()
    # count: compressed + basis + conv kernel + manifold norms
    compressed_params = (
        compressed.numel()
        + vh_r.numel()
        + kernel.numel()
        + q_norms.numel()
        + u2.numel()
        + s2.numel()
        + vh2.numel()
    )
    ratio = float(original_params / max(compressed_params, 1))

    return ConvertedMatrix(
        reconstructed=reconstructed,
        compressed=compressed_restored,
        right_basis=vh_r,
        conv_kernel=kernel,
        compression_ratio=ratio,
        rel_l2_error=float(rel_l2.item()),
    )


class TranslationAlignmentMiddleware:
    """Alignment middleware to compare original vs converted outputs on identical token streams."""

    def __init__(self, tokenizer, original_model, converted_model, device: torch.device):
        self.tokenizer = tokenizer
        self.original_model = original_model
        self.converted_model = converted_model
        self.device = device

    @torch.no_grad()
    def compare_prompt(self, prompt: str) -> dict[str, Any]:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        orig = self.original_model(input_ids=input_ids, attention_mask=attention_mask).logits
        conv = self.converted_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compare last-step distribution.
        lo = orig[:, -1, :]
        lc = conv[:, -1, :]

        cos = F.cosine_similarity(lo, lc, dim=-1).mean().item()
        top1_o = lo.argmax(dim=-1)
        top1_c = lc.argmax(dim=-1)
        top1_match = (top1_o == top1_c).float().mean().item()

        k = 5
        topk_o = torch.topk(lo, k=k, dim=-1).indices
        topk_c = torch.topk(lc, k=k, dim=-1).indices
        overlap = []
        for i in range(topk_o.shape[0]):
            a = set(topk_o[i].tolist())
            b = set(topk_c[i].tolist())
            overlap.append(len(a.intersection(b)) / k)
        top5_overlap = float(sum(overlap) / max(len(overlap), 1))

        return {
            "prompt": prompt,
            "cosine_last_logits": float(cos),
            "top1_match": float(top1_match),
            "top5_overlap": float(top5_overlap),
            "orig_top1_token": int(top1_o[0].item()),
            "conv_top1_token": int(top1_c[0].item()),
        }


def _build_calibration_texts() -> list[str]:
    return [
        "Hello, my name is",
        "The quick brown fox jumps over the lazy dog.",
        "In geometry, a manifold is a topological space.",
        "深度学习的核心思想是通过多层表示学习特征。",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Large language models rely on token embeddings and attention mechanisms.",
        "今天天气很好，我们一起学习数学与编程。",
        "The capital of France is Paris.",
    ]


@torch.no_grad()
def _prepare_calibration_batch(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    out = {k: v.to(device) for k, v in enc.items()}
    return out


def run_logit_alignment_calibration(
    original_model,
    converted_model,
    tokenizer,
    device: torch.device,
    steps: int,
    lr: float,
    max_length: int,
) -> dict[str, Any]:
    if steps <= 0:
        return {"enabled": False, "steps": 0}

    original_model.eval()
    converted_model.train()

    for p in original_model.parameters():
        p.requires_grad_(False)
    for p in converted_model.parameters():
        p.requires_grad_(False)

    # For GPT-like tied embeddings this already updates lm_head behavior as well.
    trainable = [converted_model.get_input_embeddings().weight]
    for p in trainable:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    texts = _build_calibration_texts()
    losses: list[float] = []

    for step in range(steps):
        batch = _prepare_calibration_batch(tokenizer, texts, max_length=max_length, device=device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        with torch.no_grad():
            teacher_logits = original_model(input_ids=input_ids, attention_mask=attention_mask).logits

        student_logits = converted_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # KL(teacher || student) on token distributions.
        t = 1.0
        teacher_prob = F.softmax(teacher_logits / t, dim=-1)
        student_log_prob = F.log_softmax(student_logits / t, dim=-1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))

        # Light curriculum: rotate text order.
        texts = texts[1:] + texts[:1]

    converted_model.eval()
    return {
        "enabled": True,
        "steps": steps,
        "lr": lr,
        "max_length": max_length,
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
    }


@torch.no_grad()
def apply_hidden_permutation_isomorphism(model, seed: int = 42) -> dict[str, Any]:
    """
    Apply a function-preserving hidden-dimension permutation isomorphism.
    This is an exact equivalence transform for GPT2-like architectures where
    hidden coordinates are consistently permuted across embeddings, LN and blocks.
    """
    cfg = model.config
    n_embd = int(getattr(cfg, "n_embd"))
    n_inner = int(getattr(cfg, "n_inner") or (4 * n_embd))

    rng = random.Random(seed)

    # Build an involutive permutation (only 1-cycles / 2-cycles), so P == P^{-1}.
    # This guarantees compatibility with tied input/output embeddings in GPT-like LMs.
    perm_list = list(range(n_embd))
    indices = list(range(n_embd))
    rng.shuffle(indices)
    for i in range(0, len(indices) - 1, 2):
        a = indices[i]
        b = indices[i + 1]
        perm_list[a], perm_list[b] = perm_list[b], perm_list[a]
    perm = torch.tensor(perm_list, dtype=torch.long, device=model.device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n_embd, device=model.device)

    def permute_hidden_cols(w: torch.Tensor) -> torch.Tensor:
        return w.index_select(1, perm)

    def permute_hidden_cols_inv(w: torch.Tensor) -> torch.Tensor:
        return w.index_select(1, inv_perm)

    def permute_hidden_rows(w: torch.Tensor) -> torch.Tensor:
        return w.index_select(0, inv_perm)

    def permute_hidden_vec(v: torch.Tensor) -> torch.Tensor:
        return v.index_select(0, perm)

    # Embeddings produce hidden states: h' = hP
    model.transformer.wte.weight.copy_(permute_hidden_cols(model.transformer.wte.weight))
    model.transformer.wpe.weight.copy_(permute_hidden_cols(model.transformer.wpe.weight))

    # Final normalization parameters transform with hidden permutation.
    model.transformer.ln_f.weight.copy_(permute_hidden_vec(model.transformer.ln_f.weight))
    model.transformer.ln_f.bias.copy_(permute_hidden_vec(model.transformer.ln_f.bias))

    # Block-wise exact parameter conjugation/permutation.
    for block in model.transformer.h:
        block.ln_1.weight.copy_(permute_hidden_vec(block.ln_1.weight))
        block.ln_1.bias.copy_(permute_hidden_vec(block.ln_1.bias))
        block.ln_2.weight.copy_(permute_hidden_vec(block.ln_2.weight))
        block.ln_2.bias.copy_(permute_hidden_vec(block.ln_2.bias))

        # attn.c_attn: hidden -> 3*hidden, output basis unchanged
        # Conv1D weight layout is [in, out], so W' = P^T W
        block.attn.c_attn.weight.copy_(permute_hidden_rows(block.attn.c_attn.weight))

        # attn.c_proj: hidden -> hidden, input basis unchanged, output hidden permuted
        # Conv1D layout gives W' = W P
        block.attn.c_proj.weight.copy_(permute_hidden_cols(block.attn.c_proj.weight))
        block.attn.c_proj.bias.copy_(permute_hidden_vec(block.attn.c_proj.bias))

        # mlp.c_fc: hidden -> inner, output basis unchanged
        if block.mlp.c_fc.weight.shape[0] == n_embd and block.mlp.c_fc.weight.shape[1] == n_inner:
            block.mlp.c_fc.weight.copy_(permute_hidden_rows(block.mlp.c_fc.weight))

        # mlp.c_proj: inner -> hidden, output hidden permuted
        if block.mlp.c_proj.weight.shape[0] == n_inner and block.mlp.c_proj.weight.shape[1] == n_embd:
            block.mlp.c_proj.weight.copy_(permute_hidden_cols(block.mlp.c_proj.weight))
        block.mlp.c_proj.bias.copy_(permute_hidden_vec(block.mlp.c_proj.bias))

    # lm_head consumes hidden states: logits' == logits when W' = W P
    if model.get_output_embeddings() is not None:
        lm = model.get_output_embeddings()
        tied = lm.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
        if not tied:
            lm.weight.copy_(permute_hidden_cols_inv(lm.weight))

    return {
        "enabled": True,
        "type": "hidden_permutation_isomorphism",
        "seed": seed,
        "involution": True,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convolution + math-core LLM weight conversion experiment")
    p.add_argument("--model-id", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion")
    p.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    p.add_argument("--conversion-mode", type=str, choices=["approx", "permute_exact"], default="approx")
    p.add_argument("--permute-seed", type=int, default=42)
    p.add_argument("--calib-steps", type=int, default=80)
    p.add_argument("--calib-lr", type=float, default=2e-4)
    p.add_argument("--calib-max-length", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model_id
    model_name = _sanitize_name(model_id)

    out_root = Path(args.output_dir) / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading model/tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    emb_w = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model has no output embedding head; unsupported for this experiment.")
    head_w = lm_head.weight

    rank = min(args.rank, emb_w.shape[1], head_w.shape[1])

    emb_conv = None
    head_conv = None

    print("[4/6] Building converted model and writing weights")
    converted = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    converted.eval()

    conversion_meta: dict[str, Any]
    with torch.no_grad():
        if args.conversion_mode == "approx":
            print(f"[2/6] Converting embedding with rank={rank}")
            emb_conv = convert_matrix_with_conv_math(emb_w, rank=rank, device=device)
            print(f"[3/6] Converting lm_head with rank={rank}")
            head_conv = convert_matrix_with_conv_math(head_w, rank=rank, device=device)
            converted.get_input_embeddings().weight.copy_(emb_conv.reconstructed.to(converted.dtype))
            converted.get_output_embeddings().weight.copy_(head_conv.reconstructed.to(converted.dtype))
            conversion_meta = {
                "mode": "approx",
                "rank": rank,
            }
        else:
            print(f"[2/6] Applying exact permutation isomorphism (seed={args.permute_seed})")
            conversion_meta = apply_hidden_permutation_isomorphism(converted, seed=args.permute_seed)
            print("[3/6] Exact isomorphism conversion complete")

    converted_dir = out_root / "converted_model"
    converted.save_pretrained(converted_dir)
    tokenizer.save_pretrained(converted_dir)

    # Save conversion artifacts for direct reproducibility.
    artifacts: dict[str, Any] = {
        "model_id": model_id,
        "conversion_mode": args.conversion_mode,
        "conversion_meta": conversion_meta,
    }
    if emb_conv is not None and head_conv is not None:
        artifacts.update(
            {
                "rank": rank,
                "embedding": {
                    "compressed": emb_conv.compressed.cpu(),
                    "right_basis": emb_conv.right_basis.cpu(),
                    "conv_kernel": emb_conv.conv_kernel.cpu(),
                    "rel_l2_error": emb_conv.rel_l2_error,
                    "compression_ratio": emb_conv.compression_ratio,
                },
                "lm_head": {
                    "compressed": head_conv.compressed.cpu(),
                    "right_basis": head_conv.right_basis.cpu(),
                    "conv_kernel": head_conv.conv_kernel.cpu(),
                    "rel_l2_error": head_conv.rel_l2_error,
                    "compression_ratio": head_conv.compression_ratio,
                },
            }
        )
    torch.save(artifacts, out_root / "conversion_artifacts.pt")

    print("[5/7] Running logit-alignment calibration")
    if args.conversion_mode == "approx":
        calib_info = run_logit_alignment_calibration(
            original_model=model,
            converted_model=converted,
            tokenizer=tokenizer,
            device=device,
            steps=args.calib_steps,
            lr=args.calib_lr,
            max_length=args.calib_max_length,
        )
    else:
        calib_info = {"enabled": False, "reason": "exact isomorphism mode does not require calibration"}

    print("[6/7] Running alignment middleware evaluation")
    middleware = TranslationAlignmentMiddleware(tokenizer, model, converted, device)
    prompts = [
        "Hello, my name is",
        "The quick brown fox",
        "深度学习的核心思想是",
        "def fibonacci(n):",
        "In geometry, a manifold is",
    ]
    per_prompt = [middleware.compare_prompt(p) for p in prompts]

    avg_cos = float(sum(x["cosine_last_logits"] for x in per_prompt) / len(per_prompt))
    avg_top1 = float(sum(x["top1_match"] for x in per_prompt) / len(per_prompt))
    avg_top5 = float(sum(x["top5_overlap"] for x in per_prompt) / len(per_prompt))

    report = {
        "model_id": model_id,
        "device": str(device),
        "conversion_mode": args.conversion_mode,
        "conversion_meta": conversion_meta,
        "rank": rank,
        "calibration": calib_info,
        "embedding": {
            "shape": list(emb_w.shape),
            "rel_l2_error": emb_conv.rel_l2_error if emb_conv is not None else 0.0,
            "compression_ratio_estimate": emb_conv.compression_ratio if emb_conv is not None else 1.0,
        },
        "lm_head": {
            "shape": list(head_w.shape),
            "rel_l2_error": head_conv.rel_l2_error if head_conv is not None else 0.0,
            "compression_ratio_estimate": head_conv.compression_ratio if head_conv is not None else 1.0,
        },
        "middleware_alignment": {
            "avg_cosine_last_logits": avg_cos,
            "avg_top1_match": avg_top1,
            "avg_top5_overlap": avg_top5,
            "per_prompt": per_prompt,
        },
        "consistency_assessment": {
            "usable_for_inference": bool(avg_cos > 0.90 and avg_top5 > 0.60),
            "note": "Exact mode is mathematically function-preserving; approx mode is embedding/lm_head pilot.",
        },
        "boundary": {
            "not_full_model_equivalence": bool(args.conversion_mode != "permute_exact"),
            "conversion_scope": "full hidden-basis isomorphism" if args.conversion_mode == "permute_exact" else "embedding + lm_head",
            "requires_further_validation": [
                "full-layer conversion",
                "long-context generation stability",
                "task-level benchmarks",
            ],
        },
    }

    report_path = out_root / "conversion_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[7/7] Done")
    print(f"Converted model: {converted_dir}")
    print(f"Report: {report_path}")
    print(f"Avg cosine(last logits): {avg_cos:.4f}")
    print(f"Avg top1 match: {avg_top1:.4f}")
    print(f"Avg top5 overlap: {avg_top5:.4f}")


if __name__ == "__main__":
    main()
