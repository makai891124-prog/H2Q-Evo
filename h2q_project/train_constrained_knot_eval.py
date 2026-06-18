import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from h2q.knot_kernel import H2Q_Knot_Kernel
from tools.byte_loader import get_byte_dataloader
from tools.mix_corpus_generator import generate_mix_corpus


def build_large_corpus(base_path: Path, out_path: Path, repeat: int = 4) -> None:
    generate_mix_corpus(str(base_path))
    base_text = base_path.read_text(encoding="utf-8")
    extra_parts: List[str] = []
    for rel in ("README.md", "README_H2Q_AGI.md", "README_EVALUATION_CN.md"):
        p = Path(__file__).resolve().parents[1] / rel
        if p.exists():
            extra_parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    out_text = (base_text + "\n".join(extra_parts)) * repeat
    out_path.write_text(out_text, encoding="utf-8")


def decode_token_bytes(values: List[int]) -> str:
    safe = [max(0, min(255, int(v))) for v in values]
    return bytes(safe).decode("utf-8", "ignore")


def sample_text(model: H2Q_Knot_Kernel, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    model.eval()
    token_ids = list(prompt.encode("utf-8"))[-64:]
    if not token_ids:
        token_ids = [32]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(token_ids[-64:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            if next_token == 256:
                break
            token_ids.append(next_token)
    return decode_token_bytes(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained H2Q knot retraining and evaluation")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--artifact-dir", type=str, default="/tmp/h2q_constrained_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    corpus_small = artifact_dir / "corpus_small.txt"
    corpus_large = artifact_dir / "corpus_large.txt"
    build_large_corpus(corpus_small, corpus_large, repeat=4)

    train_loader = get_byte_dataloader(
        file_path=str(corpus_large),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    data_iter = iter(train_loader)

    model = H2Q_Knot_Kernel(
        max_dim=args.dim,
        vocab_size=257,
        depth=args.depth,
        dropout_p=args.dropout,
        use_layer_norm=True,
        use_spectral_head=True,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    metrics: List[Dict[str, float]] = []
    ce_history: List[float] = []
    model.train()

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        inputs = batch.to(device)
        logits, stability_loss = model(inputs)
        cross_entropy = criterion(logits.reshape(-1, 257), inputs.reshape(-1))
        total_loss = cross_entropy + 0.1 * stability_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        ce_value = float(cross_entropy.detach().cpu())
        ce_history.append(ce_value)
        if step == 1 or step % 100 == 0:
            recent = ce_history[-100:]
            avg_last_100 = sum(recent) / len(recent)
            metrics.append(
                {
                    "step": float(step),
                    "cross_entropy": ce_value,
                    "avg_last100": float(avg_last_100),
                }
            )
            print(f"[constrained] step={step} ce={ce_value:.4f} avg100={avg_last_100:.4f}")

    prompts = ["H2Q架构", "def hello_world():", "The price is"]
    samples = {
        prompt: sample_text(model, prompt, max_new_tokens=80, device=device)
        for prompt in prompts
    }

    result = {
        "config": {
            "dim": args.dim,
            "depth": args.depth,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "grad_clip": args.grad_clip,
            "weight_decay": 0.01,
            "use_layer_norm": True,
            "use_spectral_head": True,
        },
        "corpus_bytes": os.path.getsize(corpus_large),
        "metrics": metrics,
        "final_cross_entropy": ce_history[-1],
        "best_cross_entropy": min(ce_history),
        "avg_last100": sum(ce_history[-100:]) / min(100, len(ce_history)),
        "samples": samples,
    }

    out = artifact_dir / "constrained_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
