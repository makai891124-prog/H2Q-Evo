#!/usr/bin/env python3
"""Accurate learning evaluation with regularized validation on CIFAR-10."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from superior_agi_evolution_system import SuperiorAGIEvolutionSystem


def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(batch_size: int, val_split: float = 0.1, subset_size: int = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    if subset_size:
        subset_size = min(subset_size, len(train_set))
        indices = torch.randperm(len(train_set))[:subset_size].tolist()
        train_set = Subset(train_set, indices)
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


class LearnableProjection(nn.Module):
    """低秩可学习投影 + 温和量化（输出256维0-1流）"""

    def __init__(self, in_dim: int = 3072, proj_dim: int = 256, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, proj_dim, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.up(x)
        return self.act(x)


class ConvProjection(nn.Module):
    """小型Conv+MLP投影（图像32x32）"""

    def __init__(self, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, proj_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    """增强监督头：3层MLP"""

    def __init__(self, in_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _prepare_binary_stream(system: SuperiorAGIEvolutionSystem,
                           projector: nn.Module,
                           images: torch.Tensor,
                           use_projection: bool,
                           hard_binarize: bool,
                           device: torch.device) -> torch.Tensor:
    if use_projection:
        if isinstance(projector, ConvProjection):
            projected = projector(images)
        else:
            flat = images.view(images.size(0), -1)
            projected = projector(flat)
        if hard_binarize:
            projected = (projected > 0.5).float()
        # 应用m24/DAS约束强度（若启用）
        if system.enable_m24_constraints and system.m24_strength > 0:
            constrained = system._apply_m24_constraints(projected, "image")
            projected = (1 - system.m24_strength) * projected + system.m24_strength * constrained
        if system.enable_das_core and system.das_strength > 0:
            constrained = system._apply_das_core(projected, "image")
            projected = (1 - system.das_strength) * projected + system.das_strength * constrained
        return projected.to(device)
    return system._convert_to_binary_stream(images, "image").to(device)


def evaluate(model: SuperiorAGIEvolutionSystem,
             classifier: nn.Module,
             projector: nn.Module,
             loader: DataLoader,
             device: torch.device,
             use_projection: bool,
             hard_binarize: bool) -> Dict[str, float]:
    model.evolution_core.eval()
    classifier.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            binary = _prepare_binary_stream(model, projector, images, use_projection, hard_binarize, device)
            evolved, _, _, _ = model.evolution_core({"image": binary})
            logits = classifier(evolved)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            loss_sum += loss.item() * labels.size(0)

    return {
        "accuracy": correct / max(1, total),
        "loss": loss_sum / max(1, total)
    }


def train(args):
    seed_all(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    system = SuperiorAGIEvolutionSystem(max_memory_gb=args.max_memory_gb, device="mps")
    system.raise_on_error = True
    if args.disable_m24:
        system.enable_m24_constraints = False
    else:
        system.m24_strength = args.m24_strength
    if args.disable_das:
        system.enable_das_core = False
    else:
        system.das_strength = args.das_strength

    if args.proj_type == "conv":
        projector = ConvProjection().to(device)
    else:
        projector = LearnableProjection(rank=args.proj_rank).to(device)
    classifier = MLPClassifier(in_dim=256, num_classes=10).to(device)

    params = list(system.evolution_core.parameters()) + list(projector.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    train_loader, val_loader = build_dataloaders(args.batch_size, args.val_split, args.subset_size)
    best_val = 0.0
    patience = 0
    history = []

    out_dir = Path("benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"learning_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def save_report():
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "CIFAR10",
            "train_split": 1 - args.val_split,
            "val_split": args.val_split,
            "regularization": {
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "dropout": "model default",
                "early_stopping": args.early_stopping
            },
            "best_val_accuracy": best_val,
            "history": history
        }
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        for epoch in range(args.epochs):
            system.evolution_core.train()
            classifier.train()

            epoch_loss = 0.0
            seen = 0
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                binary = _prepare_binary_stream(system, projector, images, args.use_projection, args.hard_binarize, device)
                evolved, _, _, _ = system.evolution_core({"image": binary})
                logits = classifier(evolved)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

                if args.max_batches and step + 1 >= args.max_batches:
                    break

            print(f"Epoch {epoch + 1}: train_loss={epoch_loss / max(1, seen):.4f}")

            train_metrics = {
                "loss": epoch_loss / max(1, seen)
            }
            val_metrics = evaluate(system, classifier, projector, val_loader, device, args.use_projection, args.hard_binarize)
            print(f"Epoch {epoch + 1}: val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"]
            })

            if val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stopping:
                    break

            save_report()
    finally:
        save_report()
        print(f"✅ 学习评估结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--use-projection", action="store_true", default=True)
    parser.add_argument("--hard-binarize", action="store_true", default=False)
    parser.add_argument("--proj-rank", type=int, default=128)
    parser.add_argument("--proj-type", type=str, default="conv", choices=["conv", "linear"])
    parser.add_argument("--disable-m24", action="store_true")
    parser.add_argument("--disable-das", action="store_true")
    parser.add_argument("--m24-strength", type=float, default=1.0)
    parser.add_argument("--das-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory-gb", type=float, default=16.0)
    args = parser.parse_args()

    train(args)
