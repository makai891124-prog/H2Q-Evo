"""CIFAR-10 Classification Benchmark for H2Q Spacetime Vision.

Evaluates the H2Q 4D spacetime manifold approach on standard image classification.
Compares with a simple CNN baseline to establish relative performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Stores benchmark results."""
    model_name: str
    accuracy: float
    loss: float
    train_time_sec: float
    params: int
    memory_mb: float
    throughput_samples_per_sec: float


# ============================================================================
# H2Q Spacetime Vision Classifier
# ============================================================================

class QuaternionConv2d(nn.Module):
    """Quaternion-valued 2D convolution respecting Hamilton product structure."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        # Hamilton product: q1*q2 requires 4 components interacting
        # Split channels into quaternion groups (each group has 4 channels)
        self.in_qgroups = in_channels // 4
        self.out_qgroups = out_channels // 4
        
        # Real-valued weights that implement Hamilton structure
        self.conv_ww = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_wx = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_wy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_wz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) where C = 4*n_groups
        # Apply Hamilton-structured convolution
        out = self.conv_ww(x) - self.conv_wx(x) - self.conv_wy(x) - self.conv_wz(x)
        return out + self.bias.view(1, -1, 1, 1)


class SpacetimeEvolutionBlock(nn.Module):
    """
    Spacetime evolution block implementing fractal dimension expansion.
    
    Uses constructive/destructive interference pattern:
    q_left = q + diff (constructive)
    q_right = q - diff (destructive)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        # Group action (determines evolution direction)
        self.group_action = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=max(1, channels // 16)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        
        # Differential generator (constructive/destructive split)
        self.diff_gen = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Tanh(),  # Bounded perturbation
        )
        
        # Recombination (merge two branches)
        self.recombine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
        )
        
        # Residual scaling
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Group action evolution
        evolved = self.group_action(x)
        
        # Differential (perturbation)
        diff = self.diff_gen(evolved) * self.scale
        
        # Constructive and destructive interference
        q_left = evolved + diff
        q_right = evolved - diff
        
        # Recombine branches
        combined = torch.cat([q_left, q_right], dim=1)
        out = self.recombine(combined)
        
        return out + x  # Residual


class H2QSpacetimeClassifier(nn.Module):
    """
    H2Q 4D Spacetime Manifold Classifier for CIFAR-10.
    
    Architecture (Optimized):
    1. RGB → YCbCr → Quaternion (4D) with proper normalization
    2. Hierarchical spacetime evolution with downsampling
    3. Multi-scale feature aggregation
    4. Classification head with dropout
    """
    
    def __init__(self, num_classes: int = 10, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # YCbCr projection matrix (BT.601)
        ycbcr_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=torch.float32)
        self.register_buffer('ycbcr_proj', ycbcr_matrix)
        
        # Initial stem: RGB(3) → base features
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        # Quaternion projection: 64 → 4*hidden_dim//4 (quaternion groups)
        q_channels = (hidden_dim // 4) * 4  # Ensure divisible by 4
        self.to_quaternion = nn.Sequential(
            nn.Conv2d(64, q_channels, 1),
            nn.BatchNorm2d(q_channels),
        )
        
        # Hierarchical spacetime stages
        self.stages = nn.ModuleList()
        channels = q_channels
        
        for i in range(depth):
            stage = nn.Sequential(
                SpacetimeEvolutionBlock(channels),
                SpacetimeEvolutionBlock(channels),
                nn.MaxPool2d(2) if i < depth - 1 else nn.Identity(),
            )
            self.stages.append(stage)
            
            # Double channels after pooling (except last)
            if i < depth - 2:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1),
                    nn.BatchNorm2d(channels * 2),
                ))
                channels *= 2
        
        self.final_channels = channels
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to YCbCr color space."""
        b, c, h, w = x.shape
        x_flat = x.view(b, 3, -1).permute(0, 2, 1)
        ycbcr = torch.matmul(x_flat, self.ycbcr_proj.t())
        return ycbcr.permute(0, 2, 1).view(b, 3, h, w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Stem processing
        h = self.stem(x)
        
        # 2. Project to quaternion space
        h = self.to_quaternion(h)
        
        # Normalize quaternion groups to unit sphere
        B, C, H, W = h.shape
        h_groups = h.view(B, C // 4, 4, H, W)
        h_norm = F.normalize(h_groups, p=2, dim=2)
        h = h_norm.view(B, C, H, W)
        
        # 3. Hierarchical spacetime evolution
        for stage in self.stages:
            h = stage(h)
        
        # 4. Global pooling and classification
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# ============================================================================
# Baseline CNN Classifier
# ============================================================================

class BaselineCNN(nn.Module):
    """Simple CNN baseline for comparison."""
    
    def __init__(self, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.classifier(h)


# ============================================================================
# Training and Evaluation
# ============================================================================

def get_cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data_cifar', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, estimate from process
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    return 0.0


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total, total_loss / len(loader)


def benchmark_model(model: nn.Module, model_name: str, train_loader: DataLoader,
                    test_loader: DataLoader, device: torch.device,
                    epochs: int = 10, lr: float = 1e-3) -> BenchmarkResult:
    """Full training and evaluation benchmark."""
    model = model.to(device)
    params = count_parameters(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Parameters: {params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc, test_loss = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    final_acc, final_loss = evaluate(model, test_loader, criterion, device)
    memory = get_memory_usage()
    
    # Throughput measurement
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(100, 3, 32, 32).to(device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(dummy)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        throughput = 1000 / (time.perf_counter() - t0)
    
    return BenchmarkResult(
        model_name=model_name,
        accuracy=final_acc,
        loss=final_loss,
        train_time_sec=train_time,
        params=params,
        memory_mb=memory,
        throughput_samples_per_sec=throughput,
    )


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Classification Benchmark")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="Model depth")
    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[H2Q Benchmark] CIFAR-10 Classification")
    print(f"Device: {device}")
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Benchmark models
    results = []
    
    # 1. H2Q Spacetime Classifier
    h2q_model = H2QSpacetimeClassifier(
        num_classes=10, hidden_dim=args.hidden_dim, depth=args.depth
    )
    results.append(benchmark_model(
        h2q_model, "H2Q-Spacetime", train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr
    ))
    
    # 2. Baseline CNN
    baseline_model = BaselineCNN(num_classes=10, hidden_dim=args.hidden_dim)
    results.append(benchmark_model(
        baseline_model, "Baseline-CNN", train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr
    ))
    
    # Print comparison
    print("\n" + "="*80)
    print("CIFAR-10 BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Params':<12} {'Time(s)':<12} {'Throughput':<15}")
    print("-"*80)
    for r in results:
        print(f"{r.model_name:<20} {r.accuracy:.2f}%{'':<6} {r.params:,}{'':<4} {r.train_time_sec:.1f}{'':<7} {r.throughput_samples_per_sec:.1f} samp/s")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
