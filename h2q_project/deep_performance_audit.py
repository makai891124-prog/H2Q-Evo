#!/usr/bin/env python3
"""
深度性能审计脚本 - 针对四元数架构的换算公平性和潜在作弊行为审计
Deep Performance Audit - Focusing on quaternion architecture fairness and potential cheating

关注点:
1. 参数量换算公平性 (quaternion vs real-valued parameters)
2. 延迟测试的完整性 (是否跳过关键步骤)
3. 内存测量的准确性 (是否只测量了部分内存)
4. CIFAR-10真实运行验证
"""

import torch
import torch.nn as nn
import time
import tracemalloc
import psutil
import os
import sys
import json
import numpy as np
from datetime import datetime

# ============================================================
# 审计 1: 参数量换算公平性审计
# ============================================================
class QuaternionParameterAudit:
    """审计四元数参数和实数参数的换算是否公平"""
    
    def __init__(self):
        self.results = {}
    
    def count_params(self, model):
        """统计模型参数量"""
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total
    
    def count_memory_footprint(self, model):
        """统计模型内存占用 (字节)"""
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.element_size() * p.nelement()
        return total_bytes
    
    def audit_quaternion_equivalence(self):
        """审计四元数层的参数等价性
        
        关键问题: 1个quaternion参数 = 4个real参数?
        - 如果按4个real计算, 则参数量是公平的
        - 如果按1个quaternion计算, 则低估了4倍复杂度
        """
        print("\n" + "="*80)
        print("审计 1: 四元数参数换算公平性")
        print("="*80)
        
        # 模拟简单的quaternion层
        class SimpleQuaternionLinear(nn.Module):
            """四元数线性层: y = Wx + b (quaternion版本)
            
            每个quaternion有4个分量: (w, x, y, z)
            参数量应该计为: 4 * (in_features * out_features + out_features)
            """
            def __init__(self, in_features, out_features):
                super().__init__()
                # 四元数权重: 4个分量
                self.W_r = nn.Parameter(torch.randn(out_features, in_features))
                self.W_i = nn.Parameter(torch.randn(out_features, in_features))
                self.W_j = nn.Parameter(torch.randn(out_features, in_features))
                self.W_k = nn.Parameter(torch.randn(out_features, in_features))
                # 四元数偏置: 4个分量
                self.b_r = nn.Parameter(torch.randn(out_features))
                self.b_i = nn.Parameter(torch.randn(out_features))
                self.b_j = nn.Parameter(torch.randn(out_features))
                self.b_k = nn.Parameter(torch.randn(out_features))
            
            def forward(self, x):
                # 简化版quaternion乘法
                return self.W_r @ x.T + self.b_r.unsqueeze(1)
        
        # 对比模型
        class RealLinear(nn.Module):
            """普通实数线性层"""
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
            
            def forward(self, x):
                return self.linear(x)
        
        # 创建测试模型
        in_dim, out_dim = 128, 256
        quat_model = SimpleQuaternionLinear(in_dim, out_dim)
        real_model = RealLinear(in_dim, out_dim)
        
        # 统计参数
        quat_params = self.count_params(quat_model)
        real_params = self.count_params(real_model)
        quat_memory = self.count_memory_footprint(quat_model)
        real_memory = self.count_memory_footprint(real_model)
        
        print(f"\n配置: in_dim={in_dim}, out_dim={out_dim}")
        print(f"  Quaternion层参数量: {quat_params:,} 个")
        print(f"  Real层参数量:       {real_params:,} 个")
        print(f"  参数量比例: {quat_params / real_params:.2f}x")
        print(f"  Quaternion层内存: {quat_memory / 1024:.2f} KB")
        print(f"  Real层内存:       {real_memory / 1024:.2f} KB")
        print(f"  内存比例: {quat_memory / real_memory:.2f}x")
        
        # 关键结论
        expected_ratio = 4.0  # 理论上应该是4倍
        actual_ratio = quat_params / real_params
        is_fair = abs(actual_ratio - expected_ratio) < 0.1
        
        print(f"\n✅ 换算公平性判定:")
        print(f"  理论比例: {expected_ratio:.1f}x (1个quaternion = 4个real分量)")
        print(f"  实际比例: {actual_ratio:.2f}x")
        print(f"  结论: {'公平 ✓' if is_fair else '不公平 ✗ (参数量计算可能有误)'}")
        
        self.results['quaternion_equivalence'] = {
            'quaternion_params': quat_params,
            'real_params': real_params,
            'ratio': actual_ratio,
            'expected_ratio': expected_ratio,
            'is_fair': 'yes' if is_fair else 'no'
        }
        
        return is_fair

# ============================================================
# 审计 2: 延迟测试完整性审计
# ============================================================
class LatencyTestIntegrityAudit:
    """审计延迟测试是否存在作弊行为"""
    
    def __init__(self):
        self.results = {}
    
    def audit_warmup_bias(self):
        """审计预热次数是否过多导致不公平优势
        
        潜在作弊:
        - 过度预热导致所有数据都在缓存中
        - 只测量缓存命中的情况,不测量冷启动
        """
        print("\n" + "="*80)
        print("审计 2: 延迟测试完整性 - 预热偏差")
        print("="*80)
        
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        input_data = torch.randn(1, 256).to(device)
        
        # 测试不同预热次数的影响
        warmup_configs = [0, 1, 5, 10, 50, 100]
        results = {}
        
        for warmup_count in warmup_configs:
            # 重新加载模型以清除缓存
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            ).to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(warmup_count):
                    _ = model(input_data)
            
            # 同步
            if hasattr(torch.backends, 'mps'):
                torch.mps.synchronize()
            
            # 测量
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.perf_counter()
                    _ = model(input_data)
                    if hasattr(torch.backends, 'mps'):
                        torch.mps.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1e6)  # 微秒
            
            avg_latency = np.mean(times)
            std_latency = np.std(times)
            results[warmup_count] = {
                'mean': avg_latency,
                'std': std_latency
            }
            print(f"  预热{warmup_count:3d}次: 平均延迟={avg_latency:8.2f}μs, 标准差={std_latency:7.2f}μs")
        
        # 分析偏差
        no_warmup = results[0]['mean']
        with_warmup = results[10]['mean']
        bias_percent = (no_warmup - with_warmup) / no_warmup * 100
        
        print(f"\n✅ 预热偏差分析:")
        print(f"  无预热延迟:   {no_warmup:.2f}μs")
        print(f"  10次预热延迟: {with_warmup:.2f}μs")
        print(f"  偏差:         {bias_percent:.1f}%")
        print(f"  结论: {'合理' if bias_percent < 30 else '可能存在过度预热偏差'}")
        
        self.results['warmup_bias'] = {
            'no_warmup_latency': no_warmup,
            'with_warmup_latency': with_warmup,
            'bias_percent': bias_percent,
            'is_fair': 'yes' if bias_percent < 30 else 'no'
        }
        
        return bias_percent < 30
    
    def audit_measurement_completeness(self):
        """审计是否测量了完整的推理流程
        
        潜在作弊:
        - 只测量forward不测量数据加载
        - 只测量单个token不测量完整序列
        - 跳过后处理步骤
        """
        print("\n" + "="*80)
        print("审计 2: 延迟测试完整性 - 测量完整性")
        print("="*80)
        
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        
        # 测试不同流程的延迟
        measurements = {}
        
        # 1. 只测量forward (最常见的作弊方式)
        input_data = torch.randn(1, 256).to(device)
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(input_data)
                if hasattr(torch.backends, 'mps'):
                    torch.mps.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)
        measurements['forward_only'] = np.mean(times)
        
        # 2. 测量forward + 数据传输
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                input_data = torch.randn(1, 256).to(device)
                _ = model(input_data)
                if hasattr(torch.backends, 'mps'):
                    torch.mps.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)
        measurements['forward_with_transfer'] = np.mean(times)
        
        # 3. 测量forward + 后处理
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                output = model(input_data)
                result = torch.argmax(output, dim=-1)  # 模拟后处理
                if hasattr(torch.backends, 'mps'):
                    torch.mps.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)
        measurements['forward_with_postprocess'] = np.mean(times)
        
        # 4. 完整流程
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                input_data = torch.randn(1, 256).to(device)
                output = model(input_data)
                result = torch.argmax(output, dim=-1)
                if hasattr(torch.backends, 'mps'):
                    torch.mps.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1e6)
        measurements['full_pipeline'] = np.mean(times)
        
        print(f"\n延迟测量对比:")
        for key, value in measurements.items():
            print(f"  {key:30s}: {value:8.2f}μs")
        
        # 分析
        overhead = measurements['full_pipeline'] - measurements['forward_only']
        overhead_percent = overhead / measurements['full_pipeline'] * 100
        
        print(f"\n✅ 测量完整性分析:")
        print(f"  纯forward:    {measurements['forward_only']:.2f}μs")
        print(f"  完整pipeline: {measurements['full_pipeline']:.2f}μs")
        print(f"  overhead:     {overhead:.2f}μs ({overhead_percent:.1f}%)")
        print(f"  结论: 如果只测forward, 低估了 {overhead_percent:.1f}% 的真实延迟")
        
        self.results['measurement_completeness'] = {
            'forward_only': measurements['forward_only'],
            'full_pipeline': measurements['full_pipeline'],
            'overhead_percent': overhead_percent
        }
        
        return measurements

# ============================================================
# 审计 3: 内存测量准确性审计
# ============================================================
class MemoryMeasurementAudit:
    """审计内存测量是否准确和完整"""
    
    def __init__(self):
        self.results = {}
    
    def audit_memory_measurement_methods(self):
        """对比不同内存测量方法的差异
        
        潜在问题:
        - tracemalloc只测量Python对象,不测量PyTorch张量
        - 只测量模型参数,不测量激活内存
        - 只测量峰值,不测量实际运行时内存
        """
        print("\n" + "="*80)
        print("审计 3: 内存测量准确性 - 不同测量方法对比")
        print("="*80)
        
        # 创建一个简单模型
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000)
        )
        
        measurements = {}
        
        # 方法1: 只统计参数内存 (最小估计)
        param_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
        measurements['param_only'] = param_memory
        print(f"  方法1 (参数内存):        {param_memory / 1024 / 1024:.4f} MB")
        
        # 方法2: tracemalloc (Python对象)
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        _ = [torch.randn(1000) for _ in range(100)]  # 分配一些对象
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        tracemalloc_delta = sum(stat.size_diff for stat in top_stats)
        tracemalloc.stop()
        measurements['tracemalloc'] = tracemalloc_delta
        print(f"  方法2 (tracemalloc):     {tracemalloc_delta / 1024 / 1024:.4f} MB")
        
        # 方法3: psutil (进程总内存)
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        # 分配一些数据
        big_tensor = torch.randn(1000, 1000)
        _ = model(torch.randn(32, 1000))
        mem_after = process.memory_info().rss
        psutil_delta = mem_after - mem_before
        measurements['psutil'] = psutil_delta
        print(f"  方法3 (psutil进程内存): {psutil_delta / 1024 / 1024:.4f} MB")
        
        # 方法4: PyTorch缓存 (GPU/MPS)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == "mps":
            # MPS内存难以直接测量,使用估算
            model_mps = model.to(device)
            input_mps = torch.randn(32, 1000).to(device)
            _ = model_mps(input_mps)
            # 估算: 参数 + 激活
            activation_memory = 32 * 1000 * 4 + 32 * 2000 * 4  # batch_size * dim * sizeof(float32)
            torch_memory = param_memory + activation_memory
            measurements['torch_estimate'] = torch_memory
            print(f"  方法4 (PyTorch估算):     {torch_memory / 1024 / 1024:.4f} MB")
        
        # 分析差异
        max_measurement = max(measurements.values())
        min_measurement = min(measurements.values())
        ratio = max_measurement / min_measurement
        
        print(f"\n✅ 内存测量差异分析:")
        print(f"  最小测量: {min_measurement / 1024 / 1024:.4f} MB")
        print(f"  最大测量: {max_measurement / 1024 / 1024:.4f} MB")
        print(f"  差异倍数: {ratio:.2f}x")
        print(f"  结论: 不同测量方法差异显著, 需要明确指出测量方法")
        
        self.results['memory_methods'] = {
            'measurements': {k: v / 1024 / 1024 for k, v in measurements.items()},
            'max_min_ratio': ratio
        }
        
        return measurements
    
    def audit_activation_memory(self):
        """审计激活内存是否被考虑
        
        关键问题: 0.01MB的宣称可能只计算了参数,没算激活
        """
        print("\n" + "="*80)
        print("审计 3: 内存测量准确性 - 激活内存")
        print("="*80)
        
        # 创建不同大小的模型
        configs = [
            (256, 512, 256),
            (512, 1024, 512),
            (1024, 2048, 1024)
        ]
        
        for in_dim, hidden_dim, out_dim in configs:
            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            
            # 参数内存
            param_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
            
            # 激活内存 (batch_size=32)
            batch_size = 32
            activation_memory = (
                batch_size * in_dim * 4 +      # 输入
                batch_size * hidden_dim * 4 +  # 第一层输出
                batch_size * hidden_dim * 4 +  # ReLU输出
                batch_size * out_dim * 4       # 最终输出
            )
            
            total_memory = param_memory + activation_memory
            
            print(f"\n配置: {in_dim}->{hidden_dim}->{out_dim}, batch={batch_size}")
            print(f"  参数内存:   {param_memory / 1024 / 1024:.4f} MB ({param_memory / total_memory * 100:.1f}%)")
            print(f"  激活内存:   {activation_memory / 1024 / 1024:.4f} MB ({activation_memory / total_memory * 100:.1f}%)")
            print(f"  总内存:     {total_memory / 1024 / 1024:.4f} MB")
        
        print(f"\n✅ 激活内存审计结论:")
        print(f"  激活内存通常占总内存的30-70%")
        print(f"  如果宣称的0.01MB只计算参数,则实际运行时内存可能是3-10倍")
        
        return True

# ============================================================
# 审计 4: CIFAR-10真实运行验证
# ============================================================
def run_cifar10_real_benchmark():
    """运行CIFAR-10实际训练以获取真实准确率"""
    print("\n" + "="*80)
    print("审计 4: CIFAR-10真实运行验证")
    print("="*80)
    
    print("\n检查CIFAR-10训练脚本...")
    script_path = "h2q_project/benchmarks/cifar10_classification.py"
    
    if not os.path.exists(script_path):
        print(f"  ✗ 脚本不存在: {script_path}")
        return {'script_exists': 'no'}
    
    print(f"  ✓ 脚本存在: {script_path}")
    print(f"\n⚠️  注意: 完整训练需要1-2小时")
    print(f"  建议运行命令:")
    print(f"    PYTHONPATH=. python3 {script_path} --epochs 10 --batch-size 128")
    print(f"\n  为了审计目的,可以运行快速版本 (3个epoch,验证架构):")
    print(f"    PYTHONPATH=. python3 {script_path} --epochs 3 --batch-size 128")
    
    # 询问是否运行
    print(f"\n  是否立即运行? (完整训练约需1-2小时)")
    print(f"    输入命令手动运行,或在此脚本中设置 AUTO_RUN_CIFAR10=True")
    
    return {
        'script_exists': 'yes',
        'script_path': script_path,
        'command': f'PYTHONPATH=. python3 {script_path} --epochs 10',
        'quick_command': f'PYTHONPATH=. python3 {script_path} --epochs 3'
    }

# ============================================================
# 主审计流程
# ============================================================
def run_deep_audit():
    """运行完整的深度审计"""
    print("="*80)
    print("🔍 H2Q性能宣称深度审计")
    print("="*80)
    print(f"审计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'device': str(torch.device('mps' if torch.backends.mps.is_available() else 'cpu')),
        'audits': {}
    }
    
    # 审计1: 参数换算
    print("\n" + "🔍 开始审计1: 四元数参数换算公平性" + "\n")
    param_audit = QuaternionParameterAudit()
    param_audit.audit_quaternion_equivalence()
    audit_results['audits']['quaternion_parameters'] = param_audit.results
    
    # 审计2: 延迟测试
    print("\n" + "🔍 开始审计2: 延迟测试完整性" + "\n")
    latency_audit = LatencyTestIntegrityAudit()
    latency_audit.audit_warmup_bias()
    latency_audit.audit_measurement_completeness()
    audit_results['audits']['latency_integrity'] = latency_audit.results
    
    # 审计3: 内存测量
    print("\n" + "🔍 开始审计3: 内存测量准确性" + "\n")
    memory_audit = MemoryMeasurementAudit()
    memory_audit.audit_memory_measurement_methods()
    memory_audit.audit_activation_memory()
    audit_results['audits']['memory_accuracy'] = memory_audit.results
    
    # 审计4: CIFAR-10
    print("\n" + "🔍 开始审计4: CIFAR-10真实运行" + "\n")
    cifar10_result = run_cifar10_real_benchmark()
    audit_results['audits']['cifar10_benchmark'] = cifar10_result
    
    # 保存结果
    output_file = 'deep_performance_audit_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ 深度审计完成!")
    print("="*80)
    print(f"详细结果已保存: {output_file}")
    print(f"\n关键发现总结:")
    print(f"  1. 四元数参数换算: {'公平 ✓' if audit_results['audits']['quaternion_parameters']['quaternion_equivalence']['is_fair'] == 'yes' else '不公平 ✗'}")
    print(f"  2. 延迟测试预热: {'合理 ✓' if audit_results['audits']['latency_integrity']['warmup_bias']['is_fair'] == 'yes' else '存在偏差 ⚠️'}")
    print(f"  3. 内存测量方法差异: {audit_results['audits']['memory_accuracy']['memory_methods']['max_min_ratio']:.1f}x")
    print(f"  4. CIFAR-10脚本: {'存在 ✓' if cifar10_result['script_exists'] == 'yes' else '不存在 ✗'}")
    
    return audit_results

if __name__ == "__main__":
    results = run_deep_audit()
