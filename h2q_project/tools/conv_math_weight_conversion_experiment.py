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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convolution + math-core LLM weight conversion experiment")
    p.add_argument("--model-id", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion")
    p.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
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
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    emb_w = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model has no output embedding head; unsupported for this experiment.")
    head_w = lm_head.weight

    rank = min(args.rank, emb_w.shape[1], head_w.shape[1])

    print(f"[2/6] Converting embedding with rank={rank}")
    emb_conv = convert_matrix_with_conv_math(emb_w, rank=rank, device=device)

    print(f"[3/6] Converting lm_head with rank={rank}")
    head_conv = convert_matrix_with_conv_math(head_w, rank=rank, device=device)

    print("[4/6] Building converted model and writing weights")
    converted = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    converted.eval()

    with torch.no_grad():
        converted.get_input_embeddings().weight.copy_(emb_conv.reconstructed.to(converted.dtype))
        converted.get_output_embeddings().weight.copy_(head_conv.reconstructed.to(converted.dtype))

    converted_dir = out_root / "converted_model"
    converted.save_pretrained(converted_dir)
    tokenizer.save_pretrained(converted_dir)

    # Save conversion artifacts for direct reproducibility.
    torch.save(
        {
            "model_id": model_id,
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
        },
        out_root / "conversion_artifacts.pt",
    )

    print("[5/6] Running alignment middleware evaluation")
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
        "rank": rank,
        "embedding": {
            "shape": list(emb_w.shape),
            "rel_l2_error": emb_conv.rel_l2_error,
            "compression_ratio_estimate": emb_conv.compression_ratio,
        },
        "lm_head": {
            "shape": list(head_w.shape),
            "rel_l2_error": head_conv.rel_l2_error,
            "compression_ratio_estimate": head_conv.compression_ratio,
        },
        "middleware_alignment": {
            "avg_cosine_last_logits": avg_cos,
            "avg_top1_match": avg_top1,
            "avg_top5_overlap": avg_top5,
            "per_prompt": per_prompt,
        },
        "consistency_assessment": {
            "usable_for_inference": bool(avg_cos > 0.90 and avg_top5 > 0.60),
            "note": "This reflects a small-model pilot under embedding/lm_head conversion only.",
        },
        "boundary": {
            "not_full_model_equivalence": True,
            "conversion_scope": "embedding + lm_head",
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

    print("[6/6] Done")
    print(f"Converted model: {converted_dir}")
    print(f"Report: {report_path}")
    print(f"Avg cosine(last logits): {avg_cos:.4f}")
    print(f"Avg top1 match: {avg_top1:.4f}")
    print(f"Avg top5 overlap: {avg_top5:.4f}")


if __name__ == "__main__":
    main()
