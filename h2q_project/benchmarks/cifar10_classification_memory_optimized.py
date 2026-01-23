#!/usr/bin/env python3
"""
CIFAR-10 Classification Benchmark - Memory Optimized Version
内存优化版本: 适用于16GB RAM的M4 Mac

主要优化策略:
1. 减小batch size: 128 → 16
2. 梯度累积: accumulation_steps = 8 (有效batch = 16 * 8 = 128)
3. 流式数据加载: 减少num_workers以降低多进程开销
4. 内存清理: 每个step后清理缓存
5. 混合精度训练: 使用float16减少内存占用 (可选)
6. 内存监控: 实时显示内存使用
"""

import sys
import os
import time
import argparse
import tracemalloc
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 导入模型定义
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from h2q_project.benchmarks.cifar10_classification import (
        H2QSpacetimeClassifier,
        BaselineCNN,
        BenchmarkResult
    )
except ImportError:
    # 如果无法导入，则在本文件中定义简化版本
    from h2q_project.core.models import H2QSpacetimeClassifier, BaselineCNN
    
    @dataclass
    class BenchmarkResult:
        model_name: str
        accuracy: float
        loss: float
        train_time_sec: float
        params: int
        memory_mb: float
        throughput_samples_per_sec: float


def get_memory_usage() -> float:
    """获取当前内存使用量 (MB)"""
    if torch.backends.mps.is_available():
        # MPS backend - 使用系统内存
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
    elif torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def clear_cache(device: torch.device):
    """清理内存缓存"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def get_cifar10_loaders_memory_optimized(
    batch_size: int = 16,  # 从128降到16
    data_root: str = './data_cifar'
) -> Tuple[DataLoader, DataLoader]:
    """
    获取CIFAR-10数据加载器 (内存优化版本)
    
    优化点:
    - 减小batch_size: 16 (原128)
    - 减少num_workers: 0 (原2) - 避免多进程内存开销
    - pin_memory: False (避免额外内存占用)
    """
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
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    
    # 内存优化的DataLoader配置
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,      # 避免多进程开销
        pin_memory=False,   # 避免额外内存占用
        drop_last=True      # 避免最后一个batch大小不一致
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,  # 测试时可以用更大batch
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader


def train_epoch_memory_optimized(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int = 8,  # 梯度累积步数
    verbose: bool = True
) -> float:
    """
    训练一个epoch (内存优化版本)
    
    优化点:
    - 梯度累积: accumulation_steps = 8 (有效batch = 16 * 8 = 128)
    - 内存清理: 每个step后删除中间变量
    - 定期缓存清理: 每100个step清理一次
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(loader):
        # 数据传输
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 梯度累积: 损失除以累积步数
        loss = loss / accumulation_steps
        loss.backward()
        
        running_loss += loss.item() * accumulation_steps
        
        # 梯度累积达到指定步数后更新参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if verbose and (i + 1) % (accumulation_steps * 10) == 0:
                mem_mb = get_memory_usage()
                print(f"  Step {i+1}/{len(loader)} | Loss: {running_loss/(i+1):.4f} | Mem: {mem_mb:.1f}MB", end='\r')
        
        # 清理中间变量
        del inputs, targets, outputs, loss
        
        # 定期清理缓存
        if (i + 1) % 100 == 0:
            clear_cache(device)
    
    # 处理最后一个不完整的累积batch
    if (len(loader) % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if verbose:
        print()  # 换行
    
    return running_loss / len(loader)


def evaluate_memory_optimized(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """评估模型 (内存优化版本)"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 清理中间变量
            del inputs, targets, outputs, loss
        
        # 清理缓存
        clear_cache(device)
    
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    
    return accuracy, avg_loss


def benchmark_model_memory_optimized(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    accumulation_steps: int = 8
) -> BenchmarkResult:
    """
    模型benchmark (内存优化版本)
    
    优化点:
    - 使用梯度累积训练
    - 定期内存监控
    - 每个epoch结束清理缓存
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name} (Memory Optimized)")
    print(f"Parameters: {params:,}")
    print(f"Device: {device}")
    print(f"Batch size: {train_loader.batch_size} × Accumulation: {accumulation_steps} = Effective {train_loader.batch_size * accumulation_steps}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss = train_epoch_memory_optimized(
            model, train_loader, optimizer, criterion, device,
            accumulation_steps=accumulation_steps, verbose=True
        )
        scheduler.step()
        
        # 评估
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            acc, test_loss = evaluate_memory_optimized(model, test_loader, criterion, device)
            mem_mb = get_memory_usage()
            print(f"  Train Loss: {train_loss:.4f} | Test Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Memory: {mem_mb:.1f}MB")
        
        # 每个epoch结束清理缓存
        clear_cache(device)
    
    train_time = time.time() - start_time
    
    # 最终评估
    final_acc, final_loss = evaluate_memory_optimized(model, test_loader, criterion, device)
    memory = get_memory_usage()
    
    # 吞吐量测量
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(100, 3, 32, 32).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
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
    parser = argparse.ArgumentParser(description="CIFAR-10 Classification Benchmark (Memory Optimized)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10 for memory efficiency)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16 for 16GB RAM)")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch=128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="Model depth")
    parser.add_argument("--model", type=str, default="both", choices=["h2q", "baseline", "both"], help="Which model to benchmark")
    args = parser.parse_args()
    
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n{'='*80}")
    print(f"H2Q CIFAR-10 Classification Benchmark (Memory Optimized)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*80}")
    
    # 启动内存跟踪
    tracemalloc.start()
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.1f}MB\n")
    
    # 加载数据
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders_memory_optimized(args.batch_size)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Benchmark模型
    results = []
    
    # 1. H2Q Spacetime Classifier
    if args.model in ["h2q", "both"]:
        h2q_model = H2QSpacetimeClassifier(
            num_classes=10, hidden_dim=args.hidden_dim, depth=args.depth
        )
        results.append(benchmark_model_memory_optimized(
            h2q_model, "H2Q-Spacetime", train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, accumulation_steps=args.accumulation_steps
        ))
        
        # 清理模型释放内存
        del h2q_model
        clear_cache(device)
    
    # 2. Baseline CNN
    if args.model in ["baseline", "both"]:
        baseline_model = BaselineCNN(num_classes=10, hidden_dim=args.hidden_dim)
        results.append(benchmark_model_memory_optimized(
            baseline_model, "Baseline-CNN", train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, accumulation_steps=args.accumulation_steps
        ))
        
        del baseline_model
        clear_cache(device)
    
    # 打印对比结果
    print("\n" + "="*80)
    print("CIFAR-10 BENCHMARK RESULTS (Memory Optimized)")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Params':<12} {'Time(s)':<12} {'Memory(MB)':<15}")
    print("-"*80)
    for r in results:
        print(f"{r.model_name:<20} {r.accuracy:.2f}%{'':<6} {r.params:,}{'':<4} {r.train_time_sec:.1f}{'':<7} {r.memory_mb:.1f}")
    print("="*80)
    
    # 内存统计
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nMemory Statistics:")
    print(f"  Initial: {initial_mem:.1f}MB")
    print(f"  Current: {current/1024/1024:.1f}MB (Python objects)")
    print(f"  Peak: {peak/1024/1024:.1f}MB (Python objects)")
    print(f"  Final system: {get_memory_usage():.1f}MB")
    
    return results


if __name__ == "__main__":
    main()
