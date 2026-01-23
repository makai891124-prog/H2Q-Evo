#!/usr/bin/env python3
"""
性能宣称审计验证脚本
Audit script for verifying README performance claims
"""

import torch
import torch.nn as nn
import time
import psutil
import tracemalloc
import numpy as np
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("H2Q-Evo 性能宣称审计验证 (Performance Claims Audit)")
print("="*80)
print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"设备: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
print("="*80)

results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'claims': {}
}

# ============================================================================
# 宣称1: 706K tok/s 训练吞吐
# Claim 1: 706K tokens/sec training throughput
# ============================================================================
print("\n[宣称1] 训练吞吐: 706K tokens/sec")
print("-"*80)

def test_training_throughput():
    """测试训练吞吐量"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # 简单Transformer-like模型
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=50000, dim=256, seq_len=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
            self.head = nn.Linear(dim, vocab_size)
            
        def forward(self, x):
            h = self.embed(x)
            h = self.transformer(h)
            return self.head(h)
    
    batch_size = 64
    seq_len = 64
    vocab_size = 50000
    
    model = SimpleModel(vocab_size, dim=256, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 预热
    for _ in range(5):
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 正式测试
    iterations = 100
    torch.mps.synchronize() if hasattr(torch.backends, 'mps') else None
    
    start = time.perf_counter()
    for _ in range(iterations):
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.mps.synchronize() if hasattr(torch.backends, 'mps') else None
    elapsed = time.perf_counter() - start
    
    total_tokens = batch_size * seq_len * iterations
    throughput = total_tokens / elapsed
    
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  迭代次数: {iterations}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  实际吞吐: {throughput:.0f} tokens/sec")
    print(f"  宣称吞吐: 706,000 tokens/sec")
    print(f"  达成率: {(throughput/706000)*100:.1f}%")
    
    if throughput < 706000:
        print(f"  ❌ 未达到宣称值 (差距: {706000-throughput:.0f} tokens/sec)")
    else:
        print(f"  ✅ 达到宣称值")
    
    return {
        'claimed': 706000,
        'actual': float(throughput),
        'achievement_rate': float((throughput/706000)*100),
        'verified': 'yes' if throughput >= 706000 else 'no'
    }

try:
    results['claims']['training_throughput'] = test_training_throughput()
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    results['claims']['training_throughput'] = {'error': str(e)}

# ============================================================================
# 宣称2: 23.68μs 推理延迟
# Claim 2: 23.68 microseconds inference latency
# ============================================================================
print("\n[宣称2] 推理延迟: 23.68μs (per token)")
print("-"*80)

def test_inference_latency():
    """测试单token推理延迟"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # 轻量级推理模型
    class LightweightModel(nn.Module):
        def __init__(self, vocab_size=50000, dim=256):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, vocab_size)
            
        def forward(self, x):
            h = self.embed(x)  # (B, 1, dim)
            h = torch.relu(self.fc1(h))
            return self.fc2(h)
    
    model = LightweightModel().to(device)
    model.eval()
    
    # 单token输入
    x = torch.randint(0, 50000, (1, 1)).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    # 正式测试
    iterations = 1000
    torch.mps.synchronize() if hasattr(torch.backends, 'mps') else None
    
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            torch.mps.synchronize() if hasattr(torch.backends, 'mps') else None
            latencies.append((time.perf_counter() - start) * 1e6)  # 转换为微秒
    
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"  迭代次数: {iterations}")
    print(f"  平均延迟: {avg_latency:.2f}μs")
    print(f"  P50延迟: {p50_latency:.2f}μs")
    print(f"  P99延迟: {p99_latency:.2f}μs")
    print(f"  宣称延迟: 23.68μs")
    
    if avg_latency > 23.68:
        print(f"  ❌ 超出宣称值 (慢了: {avg_latency-23.68:.2f}μs)")
    else:
        print(f"  ✅ 优于宣称值")
    
    return {
        'claimed': 23.68,
        'actual_mean': float(avg_latency),
        'actual_p50': float(p50_latency),
        'actual_p99': float(p99_latency),
        'verified': 'yes' if avg_latency <= 23.68 else 'no'
    }

try:
    results['claims']['inference_latency'] = test_inference_latency()
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    results['claims']['inference_latency'] = {'error': str(e)}

# ============================================================================
# 宣称3: 0.7MB 峰值内存
# Claim 3: 0.7MB peak memory
# ============================================================================
print("\n[宣称3] 峰值内存: 0.7MB")
print("-"*80)

def test_peak_memory():
    """测试峰值内存使用"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # 启动内存追踪
    tracemalloc.start()
    
    # 创建轻量级模型
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = TinyModel().to(device)
    
    # 运行推理
    x = torch.randn(1, 128).to(device)
    with torch.no_grad():
        _ = model(x)
    
    # 获取内存统计
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 获取进程内存
    process = psutil.Process()
    process_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    peak_mb = peak / (1024 * 1024)
    
    print(f"  Python对象峰值: {peak_mb:.2f}MB")
    print(f"  进程总内存: {process_mem:.2f}MB")
    print(f"  宣称峰值: 0.7MB")
    
    if peak_mb > 0.7:
        print(f"  ❌ 超出宣称值 (多用了: {peak_mb-0.7:.2f}MB)")
        print(f"  注: 0.7MB极小,实际Python运行时就需约10-50MB基础开销")
    else:
        print(f"  ✅ 符合宣称值")
    
    return {
        'claimed': 0.7,
        'actual_peak_mb': float(peak_mb),
        'process_memory_mb': float(process_mem),
        'verified': 'yes' if peak_mb <= 0.7 else 'no',
        'note': '0.7MB仅为模型参数,不含Python运行时基础内存'
    }

try:
    results['claims']['peak_memory'] = test_peak_memory()
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    results['claims']['peak_memory'] = {'error': str(e)}

# ============================================================================
# 宣称4: CIFAR-10 88.78% 准确率
# Claim 4: CIFAR-10 88.78% accuracy
# ============================================================================
print("\n[宣称4] CIFAR-10准确率: 88.78%")
print("-"*80)
print("  ⚠️  完整训练需要较长时间(约30分钟-2小时)")
print("  提示: 可单独运行 benchmarks/cifar10_classification.py --epochs 10")
print("  当前: 仅检查脚本是否存在")

cifar_script = Path(__file__).parent / "benchmarks" / "cifar10_classification.py"
if cifar_script.exists():
    print(f"  ✅ 训练脚本存在: {cifar_script}")
    print("  📝 需手动运行验证:")
    print("      PYTHONPATH=. python3 h2q_project/benchmarks/cifar10_classification.py --epochs 10")
    results['claims']['cifar10_accuracy'] = {
        'claimed': 88.78,
        'script_exists': 'yes',  # 改为字符串
        'verified': 'no',  # 改为字符串
        'note': '需手动运行完整训练验证 (约1-2小时)',
        'command': 'PYTHONPATH=. python3 h2q_project/benchmarks/cifar10_classification.py --epochs 10'
    }
else:
    print(f"  ❌ 训练脚本不存在")
    results['claims']['cifar10_accuracy'] = {
        'claimed': 88.78,
        'script_exists': 'no',  # 改为字符串
        'verified': 'no'  # 改为字符串
    }

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("审计总结 (Audit Summary)")
print("="*80)

verified_count = sum(1 for claim in results['claims'].values() 
                     if isinstance(claim, dict) and claim.get('verified', 'no') == 'yes')
total_testable = sum(1 for claim in results['claims'].values() 
                     if isinstance(claim, dict) and 'verified' in claim)

print(f"\n通过验证: {verified_count}/{total_testable}")

for name, data in results['claims'].items():
    if isinstance(data, dict) and 'verified' in data:
        status = "✅ 验证通过" if data['verified'] == 'yes' else "❌ 未通过验证"
        print(f"  {name}: {status}")
        if 'actual' in data and 'claimed' in data:
            print(f"    宣称: {data['claimed']}, 实测: {data['actual']:.2f}")

# 保存结果
output_file = Path(__file__).parent.parent / "performance_audit_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n详细结果已保存: {output_file}")
print("="*80)
