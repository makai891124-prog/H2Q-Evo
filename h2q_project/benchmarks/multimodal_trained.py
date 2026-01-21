"""Trained Multimodal Alignment Benchmark.

Trains the H2Q aligner to properly learn vision-text alignment,
then compares with baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimodal_alignment import (
    H2QMultimodalAligner, BaselineConcatAligner,
    generate_matched_pairs, generate_unmatched_pairs,
)


@dataclass
class TrainedAlignmentResult:
    model_name: str
    train_loss: float
    val_matched_score: float
    val_unmatched_score: float
    discrimination_gap: float
    accuracy: float  # Binary classification: matched vs unmatched
    berry_coherence: float


def create_alignment_dataset(
    num_samples: int,
    vision_dim: int,
    text_dim: int,
) -> Tuple[TensorDataset, TensorDataset]:
    """Create training and validation datasets for alignment task."""
    
    # Training data
    train_matched_v, train_matched_t = generate_matched_pairs(num_samples // 2, vision_dim, text_dim)
    train_unmatched_v, train_unmatched_t = generate_unmatched_pairs(num_samples // 2, vision_dim, text_dim)
    
    train_v = torch.cat([train_matched_v, train_unmatched_v], dim=0)
    train_t = torch.cat([train_matched_t, train_unmatched_t], dim=0)
    train_labels = torch.cat([
        torch.ones(num_samples // 2),
        torch.zeros(num_samples // 2),
    ])
    
    # Shuffle
    perm = torch.randperm(num_samples)
    train_v, train_t, train_labels = train_v[perm], train_t[perm], train_labels[perm]
    
    train_dataset = TensorDataset(train_v, train_t, train_labels)
    
    # Validation data
    val_size = num_samples // 5
    val_matched_v, val_matched_t = generate_matched_pairs(val_size // 2, vision_dim, text_dim)
    val_unmatched_v, val_unmatched_t = generate_unmatched_pairs(val_size // 2, vision_dim, text_dim)
    
    val_v = torch.cat([val_matched_v, val_unmatched_v], dim=0)
    val_t = torch.cat([val_matched_t, val_unmatched_t], dim=0)
    val_labels = torch.cat([
        torch.ones(val_size // 2),
        torch.zeros(val_size // 2),
    ])
    
    val_dataset = TensorDataset(val_v, val_t, val_labels)
    
    return train_dataset, val_dataset


class AlignmentClassifier(nn.Module):
    """Wrapper that adds classification head to alignment model."""
    
    def __init__(self, aligner: nn.Module, hidden_dim: int):
        super().__init__()
        self.aligner = aligner
        
        # Classification head for matched/unmatched
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = self.aligner(vision, text)
        
        # Get quaternion scalar parts as features
        v_q = result.get("vision_quaternion")
        t_q = result.get("text_quaternion")
        
        if v_q is not None and t_q is not None:
            # Use scalar part (w) of quaternions as features
            v_feat = v_q[:, 0]  # (B, D)
            t_feat = t_q[:, 0]  # (B, D)
            combined = torch.cat([v_feat, t_feat], dim=1)
        else:
            # Fallback for baseline
            combined = torch.cat([vision[:, :64], text[:, :64]], dim=1)
            combined = F.pad(combined, (0, self.classifier[0].in_features - combined.shape[1]))
        
        logits = self.classifier(combined).squeeze(-1)
        result["logits"] = logits
        result["pred"] = torch.sigmoid(logits)
        
        return result


def train_aligner(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> Tuple[nn.Module, float]:
    """Train alignment classifier."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for v, t, labels in train_loader:
            v, t, labels = v.to(device), t.to(device), labels.to(device)
            
            optimizer.zero_grad()
            out = model(v, t)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v, t, labels in val_loader:
                v, t, labels = v.to(device), t.to(device), labels.to(device)
                out = model(v, t)
                val_loss += criterion(out["logits"], labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    return model, best_val_loss


def evaluate_aligner(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> TrainedAlignmentResult:
    """Evaluate trained alignment model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    matched_scores = []
    unmatched_scores = []
    berry_coherences = []
    
    with torch.no_grad():
        for v, t, labels in val_loader:
            v, t, labels = v.to(device), t.to(device), labels.to(device)
            out = model(v, t)
            
            preds = (out["pred"] > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # Separate matched/unmatched scores
            alignment = out["alignment_score"]
            for i, label in enumerate(labels):
                if label.item() > 0.5:
                    matched_scores.append(alignment[i].item())
                else:
                    unmatched_scores.append(alignment[i].item())
            
            # Berry coherence (H2Q specific)
            if "berry_phase_coherence" in out:
                berry_coherences.extend(out["berry_phase_coherence"].cpu().tolist())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = (all_preds == all_labels).float().mean().item() * 100
    matched_mean = np.mean(matched_scores) if matched_scores else 0
    unmatched_mean = np.mean(unmatched_scores) if unmatched_scores else 0
    berry_mean = np.mean(berry_coherences) if berry_coherences else 0
    
    return TrainedAlignmentResult(
        model_name=model_name,
        train_loss=0.0,  # Will be filled by caller
        val_matched_score=matched_mean,
        val_unmatched_score=unmatched_mean,
        discrimination_gap=matched_mean - unmatched_mean,
        accuracy=accuracy,
        berry_coherence=berry_mean,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vision-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[H2Q] Trained Multimodal Alignment Benchmark")
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}, Epochs: {args.epochs}")
    
    # Create dataset
    train_dataset, val_dataset = create_alignment_dataset(
        args.num_samples, args.vision_dim, args.text_dim
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    results = []
    
    # 1. H2Q Berry Phase Aligner
    print("\n[Training H2Q-BerryPhase Aligner]")
    h2q_base = H2QMultimodalAligner(args.vision_dim, args.text_dim, args.hidden_dim)
    h2q_model = AlignmentClassifier(h2q_base, args.hidden_dim)
    h2q_model, h2q_loss = train_aligner(h2q_model, train_loader, val_loader, device, args.epochs)
    h2q_result = evaluate_aligner(h2q_model, val_loader, device, "H2Q-BerryPhase")
    h2q_result = TrainedAlignmentResult(
        model_name=h2q_result.model_name,
        train_loss=h2q_loss,
        val_matched_score=h2q_result.val_matched_score,
        val_unmatched_score=h2q_result.val_unmatched_score,
        discrimination_gap=h2q_result.discrimination_gap,
        accuracy=h2q_result.accuracy,
        berry_coherence=h2q_result.berry_coherence,
    )
    results.append(h2q_result)
    
    # 2. Baseline
    print("\n[Training Baseline-Concat Aligner]")
    base_base = BaselineConcatAligner(args.vision_dim, args.text_dim, args.hidden_dim)
    base_model = AlignmentClassifier(base_base, args.hidden_dim)
    base_model, base_loss = train_aligner(base_model, train_loader, val_loader, device, args.epochs)
    base_result = evaluate_aligner(base_model, val_loader, device, "Baseline-Concat")
    base_result = TrainedAlignmentResult(
        model_name=base_result.model_name,
        train_loss=base_loss,
        val_matched_score=base_result.val_matched_score,
        val_unmatched_score=base_result.val_unmatched_score,
        discrimination_gap=base_result.discrimination_gap,
        accuracy=base_result.accuracy,
        berry_coherence=base_result.berry_coherence,
    )
    results.append(base_result)
    
    # Print results
    print("\n" + "="*80)
    print("TRAINED MULTIMODAL ALIGNMENT RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Matched':<12} {'Unmatched':<12} {'Gap':<12} {'Berry':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r.model_name:<20} {r.accuracy:.2f}%{'':<5} "
              f"{r.val_matched_score:.4f}{'':<6} "
              f"{r.val_unmatched_score:.4f}{'':<6} "
              f"{r.discrimination_gap:+.4f}{'':<5} "
              f"{r.berry_coherence:.4f}")
    
    print("="*80)
    
    # Determine winner
    h2q_acc = results[0].accuracy
    base_acc = results[1].accuracy
    
    if h2q_acc > base_acc:
        print(f"\n✓ H2Q-BerryPhase WINS with {h2q_acc - base_acc:.2f}% higher accuracy")
    elif base_acc > h2q_acc:
        print(f"\n✓ Baseline-Concat wins with {base_acc - h2q_acc:.2f}% higher accuracy")
    else:
        print("\n= Models perform equally")
    
    print("\nKey observation: H2Q provides Berry phase coherence metric for")
    print("interpretable alignment quality assessment (not available in baseline)")
    
    return results


if __name__ == "__main__":
    main()
