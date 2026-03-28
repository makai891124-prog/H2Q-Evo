#!/usr/bin/env python3
"""
Public DAS distillation experiment:
1) Single-head Transformer Q/K/V conversion via distillation.
2) Token-table distillation to DAS math structure.
3) Save new DAS structure weight file and readable manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _bytes_for_params(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> float:
    top_a = torch.topk(a, k=k, dim=-1).indices
    top_b = torch.topk(b, k=k, dim=-1).indices
    vals = []
    for i in range(top_a.shape[0]):
        sa = set(top_a[i].tolist())
        sb = set(top_b[i].tolist())
        vals.append(len(sa.intersection(sb)) / k)
    return float(sum(vals) / max(len(vals), 1))


class DASRotorProjector(nn.Module):
    """
    DAS projector with lazy path execution:
    x --(rotor chain)--> h --(rank paths)--> y
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_rotors = num_rotors

        self.basis_in = nn.Parameter(torch.empty(rank, in_dim))
        self.basis_out = nn.Parameter(torch.empty(rank, out_dim))
        self.path_logits = nn.Parameter(torch.full((rank,), 3.5))
        self.rotor_angles = nn.Parameter(torch.zeros(num_rotors))

        i_idx, j_idx = self._build_pair_schedule(in_dim, num_rotors)
        self.register_buffer("rotor_i", i_idx, persistent=False)
        self.register_buffer("rotor_j", j_idx, persistent=False)

        self.reset_parameters()

    @staticmethod
    def _build_pair_schedule(dim: int, num_rotors: int) -> tuple[torch.Tensor, torch.Tensor]:
        i_vals = []
        j_vals = []
        stride = 7 % dim
        if stride == 0:
            stride = 1
        cur = 0
        for t in range(num_rotors):
            i = cur
            j = (cur + 17 + 3 * t) % dim
            if i == j:
                j = (j + 1) % dim
            i_vals.append(i)
            j_vals.append(j)
            cur = (cur + stride + t) % dim
        return torch.tensor(i_vals, dtype=torch.long), torch.tensor(j_vals, dtype=torch.long)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis_in, mean=0.0, std=1.0 / math.sqrt(self.in_dim))
        nn.init.normal_(self.basis_out, mean=0.0, std=1.0 / math.sqrt(max(self.rank, 1)))

    def _apply_rotors(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for t in range(self.num_rotors):
            i = int(self.rotor_i[t].item())
            j = int(self.rotor_j[t].item())
            th = self.rotor_angles[t]
            c = torch.cos(th)
            s = torch.sin(th)
            xi = h[..., i]
            xj = h[..., j]
            yi = c * xi - s * xj
            yj = s * xi + c * xj
            h = h.clone()
            h[..., i] = yi
            h[..., j] = yj
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._apply_rotors(x)
        gates = torch.sigmoid(self.path_logits)
        coeff = h @ self.basis_in.T
        coeff = coeff * gates.unsqueeze(0)
        y = coeff @ self.basis_out
        return y


class DASQKVHead(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.q = DASRotorProjector(in_dim, head_dim, rank, num_rotors)
        self.k = DASRotorProjector(in_dim, head_dim, rank, num_rotors)
        self.v = DASRotorProjector(in_dim, head_dim, rank, num_rotors)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


class DASTokenMapper(nn.Module):
    """Distilled token-table mapper on selected token ids."""

    def __init__(self, hidden_dim: int, token_table_size: int, rank: int, num_rotors: int) -> None:
        super().__init__()
        self.projector = DASRotorProjector(hidden_dim, token_table_size, rank, num_rotors)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden)


@dataclass
class DistillBatch:
    c_attn_input: torch.Tensor
    q_target: torch.Tensor
    k_target: torch.Tensor
    v_target: torch.Tensor
    hidden_final: torch.Tensor
    logits_subset: torch.Tensor


def _build_prompts() -> list[str]:
    return [
        "Hello, my name is",
        "The quick brown fox jumps over the lazy dog.",
        "In geometry, a manifold is a topological space.",
        "Large language models rely on token embeddings and attention mechanisms.",
        "Knowledge distillation can transfer behavior into compact structures.",
        "Deep learning and algebraic geometry can be connected through symmetry.",
        "Engineering validation requires auditable and reproducible metrics.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    ]


def _collect_teacher_batch(
    teacher,
    tokenizer,
    layer_idx: int,
    head_idx: int,
    token_table_size: int,
    device: torch.device,
) -> tuple[DistillBatch, torch.Tensor]:
    texts = _build_prompts()
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    cache: dict[str, torch.Tensor] = {}

    # Family A: GPT2-like fused c_attn projection.
    if hasattr(teacher, "transformer") and hasattr(teacher.transformer, "h"):
        block = teacher.transformer.h[layer_idx]
        c_attn = block.attn.c_attn

        def _hook_cattn(_m, hook_in, hook_out):
            cache["x"] = hook_in[0].detach()
            cache["y"] = hook_out.detach()

        h = c_attn.register_forward_hook(_hook_cattn)
        with torch.no_grad():
            out = teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h.remove()

        x = cache["x"]
        y = cache["y"]
        bsz, seqlen, three_c = y.shape
        c = three_c // 3
        n_head = int(getattr(teacher.config, "n_head", getattr(teacher.config, "num_attention_heads", 1)))
        d_head = c // max(n_head, 1)

        q_all = y[..., :c]
        k_all = y[..., c : 2 * c]
        v_all = y[..., 2 * c :]

        q = q_all.view(bsz, seqlen, n_head, d_head)[..., head_idx, :]
        k = k_all.view(bsz, seqlen, n_head, d_head)[..., head_idx, :]
        v = v_all.view(bsz, seqlen, n_head, d_head)[..., head_idx, :]
    else:
        # Family B: Decoder-only models with explicit q_proj/k_proj/v_proj (Llama/Qwen/SmolLM style).
        model_body = getattr(teacher, "model", None)
        if model_body is None or not hasattr(model_body, "layers"):
            raise RuntimeError("Unsupported model architecture for QKV hook collection")

        layer = model_body.layers[layer_idx]
        self_attn = layer.self_attn
        q_proj = self_attn.q_proj
        k_proj = self_attn.k_proj
        v_proj = self_attn.v_proj

        def _hook_q(_m, hook_in, hook_out):
            cache["x"] = hook_in[0].detach()
            cache["q"] = hook_out.detach()

        def _hook_k(_m, _hook_in, hook_out):
            cache["k"] = hook_out.detach()

        def _hook_v(_m, _hook_in, hook_out):
            cache["v"] = hook_out.detach()

        hq = q_proj.register_forward_hook(_hook_q)
        hk = k_proj.register_forward_hook(_hook_k)
        hv = v_proj.register_forward_hook(_hook_v)
        with torch.no_grad():
            out = teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hq.remove()
        hk.remove()
        hv.remove()

        x = cache["x"]
        q_all = cache["q"]
        k_all = cache["k"]
        v_all = cache["v"]

        bsz, seqlen, c_q = q_all.shape
        n_head = int(getattr(teacher.config, "num_attention_heads", getattr(teacher.config, "n_head", 1)))
        n_kv_head = int(getattr(teacher.config, "num_key_value_heads", n_head))
        d_head = c_q // max(n_head, 1)

        q_head_idx = int(max(0, min(head_idx, n_head - 1)))
        kv_group = max(1, n_head // max(n_kv_head, 1))
        kv_head_idx = int(max(0, min(q_head_idx // kv_group, n_kv_head - 1)))

        q = q_all.view(bsz, seqlen, n_head, d_head)[..., q_head_idx, :]
        k = k_all.view(bsz, seqlen, n_kv_head, d_head)[..., kv_head_idx, :]
        v = v_all.view(bsz, seqlen, n_kv_head, d_head)[..., kv_head_idx, :]

    hidden_final = out.hidden_states[-1].detach()  # [B,T,C]
    logits = out.logits.detach()  # [B,T,V]

    # Build token table by teacher global top magnitude over this calibration set.
    score = logits.abs().mean(dim=(0, 1))
    table_size = min(token_table_size, score.numel())
    token_ids = torch.topk(score, k=table_size, dim=0).indices.detach()
    logits_subset = logits.index_select(dim=-1, index=token_ids)

    batch = DistillBatch(
        c_attn_input=x.reshape(-1, x.shape[-1]),
        q_target=q.reshape(-1, q.shape[-1]),
        k_target=k.reshape(-1, k.shape[-1]),
        v_target=v.reshape(-1, v.shape[-1]),
        hidden_final=hidden_final.reshape(-1, hidden_final.shape[-1]),
        logits_subset=logits_subset.reshape(-1, logits_subset.shape[-1]),
    )
    return batch, token_ids


def distill_qkv(
    model,
    tokenizer,
    layer_idx: int,
    head_idx: int,
    rank: int,
    num_rotors: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[DASQKVHead, dict[str, Any], DistillBatch, torch.Tensor]:
    teacher = model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    batch, token_ids = _collect_teacher_batch(
        teacher,
        tokenizer,
        layer_idx=layer_idx,
        head_idx=head_idx,
        token_table_size=4096,
        device=device,
    )

    student = DASQKVHead(
        in_dim=batch.c_attn_input.shape[-1],
        head_dim=batch.q_target.shape[-1],
        rank=rank,
        num_rotors=num_rotors,
    ).to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    losses = []

    for _ in range(steps):
        q_hat, k_hat, v_hat = student(batch.c_attn_input)
        loss_q = F.mse_loss(q_hat, batch.q_target)
        loss_k = F.mse_loss(k_hat, batch.k_target)
        loss_v = F.mse_loss(v_hat, batch.v_target)
        loss = loss_q + loss_k + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        q_hat, k_hat, v_hat = student(batch.c_attn_input)
        q_cos = F.cosine_similarity(q_hat, batch.q_target, dim=-1).mean().item()
        k_cos = F.cosine_similarity(k_hat, batch.k_target, dim=-1).mean().item()
        v_cos = F.cosine_similarity(v_hat, batch.v_target, dim=-1).mean().item()
        q_mae = (q_hat - batch.q_target).abs().mean().item()
        k_mae = (k_hat - batch.k_target).abs().mean().item()
        v_mae = (v_hat - batch.v_target).abs().mean().item()

    metrics = {
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "q_cosine": float(q_cos),
        "k_cosine": float(k_cos),
        "v_cosine": float(v_cos),
        "q_mae": float(q_mae),
        "k_mae": float(k_mae),
        "v_mae": float(v_mae),
    }
    return student, metrics, batch, token_ids


def distill_token_mapper(
    hidden: torch.Tensor,
    logits_subset: torch.Tensor,
    rank: int,
    num_rotors: int,
    steps: int,
    lr: float,
    temperature: float,
    temperature_end: float,
    topk: int,
    ranking_weight: float,
    mse_weight: float,
    ranking_margin: float,
    hard_neg_k: int,
    hard_neg_weight: float,
    stage_split: float,
    stage1_rank_scale: float,
    device: torch.device,
) -> tuple[DASTokenMapper, dict[str, Any]]:
    mapper = DASTokenMapper(
        hidden_dim=hidden.shape[-1],
        token_table_size=logits_subset.shape[-1],
        rank=rank,
        num_rotors=num_rotors,
    ).to(device)

    opt = torch.optim.AdamW(mapper.parameters(), lr=lr)
    losses = []

    t_start = max(float(temperature), 1e-4)
    t_end = max(float(temperature_end), 1e-4)

    with torch.no_grad():
        mu = logits_subset.mean(dim=0, keepdim=True)
        sigma = logits_subset.std(dim=0, keepdim=True).clamp_min(1e-4)
        teacher_norm = (logits_subset - mu) / sigma

    k = max(1, min(int(topk), teacher_norm.shape[-1] // 2 if teacher_norm.shape[-1] > 1 else 1))
    rank_w = float(max(0.0, ranking_weight))
    mse_w = float(min(max(0.0, mse_weight), 1.0))
    hard_k = max(1, min(int(hard_neg_k), teacher_norm.shape[-1] // 2 if teacher_norm.shape[-1] > 1 else 1))
    hard_w = float(max(0.0, hard_neg_weight))
    kl_w = max(0.0, 1.0 - mse_w - rank_w - hard_w)
    split = float(min(max(0.05, stage_split), 0.95))
    stage1_scale = float(min(max(0.05, stage1_rank_scale), 1.0))

    for step in range(steps):
        alpha = float(step) / float(max(steps - 1, 1))
        cur_t = t_start + (t_end - t_start) * alpha
        in_stage1 = alpha < split

        cur_rank_w = rank_w * (stage1_scale if in_stage1 else 1.0)
        cur_hard_w = hard_w * (stage1_scale if in_stage1 else 1.0)
        cur_margin = float(ranking_margin) * (0.8 if in_stage1 else 1.0)

        pred = mapper(hidden)
        pred_norm = (pred - mu) / sigma

        teacher_prob = F.softmax(teacher_norm / cur_t, dim=-1)
        student_log_prob = F.log_softmax(pred_norm / cur_t, dim=-1)
        loss_kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (cur_t * cur_t)
        loss_mse = F.mse_loss(pred_norm, teacher_norm)

        pos_idx = torch.topk(teacher_norm, k=k, dim=-1).indices
        neg_idx = torch.topk(-teacher_norm, k=k, dim=-1).indices
        pred_pos = pred_norm.gather(dim=-1, index=pos_idx)
        pred_neg = pred_norm.gather(dim=-1, index=neg_idx)
        pair_margin = pred_pos.unsqueeze(-1) - pred_neg.unsqueeze(-2)
        loss_rank = F.relu(cur_margin - pair_margin).mean()

        student_hard_idx = torch.topk(pred_norm.detach(), k=hard_k, dim=-1).indices
        pred_hard_neg = pred_norm.gather(dim=-1, index=student_hard_idx)
        hard_margin = pred_pos.unsqueeze(-1) - pred_hard_neg.unsqueeze(-2)
        loss_hard = F.relu(cur_margin - hard_margin).mean()

        total_w = kl_w + mse_w + cur_rank_w + cur_hard_w
        if total_w <= 1e-8:
            total_w = 1.0
        n_kl = kl_w / total_w
        n_mse = mse_w / total_w
        n_rank = cur_rank_w / total_w
        n_hard = cur_hard_w / total_w

        loss = n_kl * loss_kl + n_mse * loss_mse + n_rank * loss_rank + n_hard * loss_hard
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        pred = mapper(hidden)
        cos = F.cosine_similarity(pred, logits_subset, dim=-1).mean().item()
        mae = (pred - logits_subset).abs().mean().item()
        top1 = (pred.argmax(dim=-1) == logits_subset.argmax(dim=-1)).float().mean().item()
        top5 = _topk_overlap(pred, logits_subset, k=5)

    metrics = {
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "cosine": float(cos),
        "mae": float(mae),
        "top1_match": float(top1),
        "top5_overlap": float(top5),
        "temperature_start": float(t_start),
        "temperature_end": float(t_end),
        "topk": int(k),
        "ranking_weight": float(rank_w),
        "hard_neg_k": int(hard_k),
        "hard_neg_weight": float(hard_w),
        "mse_weight": float(mse_w),
        "kl_weight": float(kl_w),
        "ranking_margin": float(ranking_margin),
        "stage_split": float(split),
        "stage1_rank_scale": float(stage1_scale),
    }
    return mapper, metrics


@torch.inference_mode()
def benchmark_latency(module: nn.Module, x: torch.Tensor, rounds: int = 80, warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        _ = module(x)
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _ = module(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    ts = sorted(times)
    p90_idx = max(0, int(0.9 * len(ts)) - 1)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times)),
        "p50_ms": float(statistics.median(times)),
        "p90_ms": float(ts[p90_idx]),
    }


class DenseTokenMapper(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def save_structure(
    output_pt: Path,
    output_json: Path,
    model_id: str,
    layer_idx: int,
    head_idx: int,
    token_ids: torch.Tensor,
    qkv_student: DASQKVHead,
    token_mapper: DASTokenMapper,
    metrics: dict[str, Any],
) -> None:
    pkg = {
        "model_id": model_id,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "token_ids": token_ids.cpu(),
        "qkv_state_dict": qkv_student.state_dict(),
        "token_mapper_state_dict": token_mapper.state_dict(),
        "qkv_config": {
            "in_dim": qkv_student.q.in_dim,
            "head_dim": qkv_student.q.out_dim,
            "rank": qkv_student.q.rank,
            "num_rotors": qkv_student.q.num_rotors,
        },
        "token_mapper_config": {
            "hidden_dim": token_mapper.projector.in_dim,
            "token_table_size": token_mapper.projector.out_dim,
            "rank": token_mapper.projector.rank,
            "num_rotors": token_mapper.projector.num_rotors,
        },
        "metrics": metrics,
        "format_version": "das_token_structure_v1",
    }
    torch.save(pkg, output_pt)

    manifest = {
        "format_version": "das_token_structure_v1",
        "structure_file": str(output_pt),
        "model_id": model_id,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "token_table_size": int(token_ids.numel()),
        "metrics": metrics,
    }
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAS QKV + token-table distillation experiment")
    p.add_argument("--model-id", type=str, default="distilgpt2")
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--head-idx", type=int, default=0)
    p.add_argument("--qkv-rank", type=int, default=64)
    p.add_argument("--qkv-rotors", type=int, default=12)
    p.add_argument("--qkv-steps", type=int, default=120)
    p.add_argument("--token-table-size", type=int, default=4096)
    p.add_argument("--token-rank", type=int, default=96)
    p.add_argument("--token-rotors", type=int, default=8)
    p.add_argument("--token-steps", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-end", type=float, default=0.55)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--ranking-weight", type=float, default=0.35)
    p.add_argument("--mse-weight", type=float, default=0.15)
    p.add_argument("--ranking-margin", type=float, default=0.20)
    p.add_argument("--hard-neg-k", type=int, default=6)
    p.add_argument("--hard-neg-weight", type=float, default=0.15)
    p.add_argument("--stage-split", type=float, default=0.45)
    p.add_argument("--stage1-rank-scale", type=float, default=0.35)
    p.add_argument("--output-dir", type=str, default="reports/conv_math_conversion")
    p.add_argument("--seed", type=int, default=20260328)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qkv_student, qkv_metrics, batch, token_ids = distill_qkv(
        model=model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        head_idx=args.head_idx,
        rank=args.qkv_rank,
        num_rotors=args.qkv_rotors,
        steps=args.qkv_steps,
        lr=args.lr,
        device=device,
    )

    # Trim token table to requested size.
    token_ids = token_ids[: min(args.token_table_size, token_ids.numel())]
    logits_subset = batch.logits_subset[:, : token_ids.numel()]

    token_mapper, token_metrics = distill_token_mapper(
        hidden=batch.hidden_final,
        logits_subset=logits_subset,
        rank=args.token_rank,
        num_rotors=args.token_rotors,
        steps=args.token_steps,
        lr=args.lr,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        topk=args.topk,
        ranking_weight=args.ranking_weight,
        mse_weight=args.mse_weight,
        ranking_margin=args.ranking_margin,
        hard_neg_k=args.hard_neg_k,
        hard_neg_weight=args.hard_neg_weight,
        stage_split=args.stage_split,
        stage1_rank_scale=args.stage1_rank_scale,
        device=device,
    )

    # Baseline dense mapper for speed/memory comparison on token-table projection.
    dense_mapper = DenseTokenMapper(batch.hidden_final.shape[-1], logits_subset.shape[-1]).to(device)
    with torch.no_grad():
        w = torch.linalg.lstsq(batch.hidden_final, logits_subset).solution
        dense_mapper.linear.weight.copy_(w.T)

    x_bench = batch.hidden_final[: min(256, batch.hidden_final.shape[0])]
    dense_lat = benchmark_latency(dense_mapper, x_bench, rounds=60, warmup=15)
    das_lat = benchmark_latency(token_mapper, x_bench, rounds=60, warmup=15)

    dense_params = _count_parameters(dense_mapper)
    das_params = _count_parameters(token_mapper)

    report = {
        "model_id": args.model_id,
        "device": str(device),
        "layer_idx": args.layer_idx,
        "head_idx": args.head_idx,
        "qkv_distillation": qkv_metrics,
        "token_distillation": token_metrics,
        "token_table_size": int(token_ids.numel()),
        "memory": {
            "dense_params": dense_params,
            "das_params": das_params,
            "param_compression_ratio": float(dense_params / max(das_params, 1)),
            "dense_param_bytes": _bytes_for_params(dense_mapper),
            "das_param_bytes": _bytes_for_params(token_mapper),
        },
        "latency_ms": {
            "dense": dense_lat,
            "das": das_lat,
            "speedup_ratio": float(dense_lat["mean_ms"] / max(das_lat["mean_ms"], 1e-9)),
        },
        "acceptance": {
            "qkv_head_alignment": bool(
                qkv_metrics["q_cosine"] > 0.97 and qkv_metrics["k_cosine"] > 0.97 and qkv_metrics["v_cosine"] > 0.97
            ),
            "token_table_alignment": bool(token_metrics["cosine"] > 0.97 and token_metrics["top5_overlap"] > 0.6),
            "speedup_observed": bool((dense_lat["mean_ms"] / max(das_lat["mean_ms"], 1e-9)) > 1.05),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = _sanitize_model_name(args.model_id)

    report_path = out_dir / f"das_qkv_token_distill_{model_name}_20260328.json"
    struct_pt = out_dir / f"das_token_structure_{model_name}_20260328.pt"
    struct_json = out_dir / f"das_token_structure_manifest_{model_name}_20260328.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_structure(
        output_pt=struct_pt,
        output_json=struct_json,
        model_id=args.model_id,
        layer_idx=args.layer_idx,
        head_idx=args.head_idx,
        token_ids=token_ids,
        qkv_student=qkv_student,
        token_mapper=token_mapper,
        metrics=report,
    )

    print("[DAS] QKV + token-table distillation done")
    print(f"Report: {report_path}")
    print(f"Structure: {struct_pt}")
    print(f"Manifest: {struct_json}")
    print(f"Speedup ratio dense/das: {report['latency_ms']['speedup_ratio']:.4f}")
    print(f"Token cosine: {token_metrics['cosine']:.6f}")


if __name__ == "__main__":
    main()
