#!/usr/bin/env python3
"""
H2Q-Evo 项目代码真实性审计脚本

验证：
1. 代码是否有硬编码的基准测试结果
2. DeepSeek模型是否真实启动
3. 结晶化压缩的真实性能
4. 内存优化声明的真实性
"""

import torch
import torch.nn as nn
import json
import os
import time
import psutil
import hashlib
from typing import Dict, Any, List
import numpy as np


def audit_hardcoded_results():
    """审计是否有硬编码的基准测试结果"""
    print("🔍 审计1: 检查硬编码基准测试结果")
    print("=" * 50)

    suspicious_files = [
        'deepseek_memory_safe_benchmark_results.json',
        'benchmark_results.json',
        'benchmark_results_v2.json'
    ]

    issues = []

    for file in suspicious_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查可疑模式
                if isinstance(data, dict):
                    for category, tests in data.items():
                        if isinstance(tests, list):
                            for test in tests:
                                if isinstance(test, dict):
                                    # 检查response_time是否不真实（太快）
                                    if 'response_time' in test:
                                        rt = test['response_time']
                                        if rt < 0.001:  # 小于1ms
                                            issues.append(f"{file}: {test.get('test_name', 'unknown')} 响应时间可疑: {rt}秒")

                                    # 检查memory_used是否固定值
                                    if 'memory_used' in test and test['memory_used'] == 50:
                                        issues.append(f"{file}: {test.get('test_name', 'unknown')} 内存使用固定为50MB")

                                    # 检查quality_score是否可疑
                                    if 'quality_score' in test and test['quality_score'] == 0.0:
                                        issues.append(f"{file}: {test.get('test_name', 'unknown')} 质量评分始终为0")

            except Exception as e:
                issues.append(f"{file}: JSON解析错误 - {e}")

    if issues:
        print("❌ 发现可疑硬编码模式:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 未发现明显的硬编码模式")

    return issues


def audit_deepseek_loading():
    """审计DeepSeek模型是否真实启动"""
    print("\n🔍 审计2: 检查DeepSeek模型真实性")
    print("=" * 50)

    issues = []

    # 检查是否有真实的模型文件
    model_dirs = ['models/', 'crystallized_models/']
    deepseek_files = []

    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if 'deepseek' in file.lower():
                        deepseek_files.append(os.path.join(root, file))

    if not deepseek_files:
        issues.append("未找到任何DeepSeek相关的模型文件")
    else:
        print(f"找到 {len(deepseek_files)} 个DeepSeek相关文件:")
        for f in deepseek_files:
            size = os.path.getsize(f) / (1024**3)  # GB
            print(f"     {f}: {size:.3f} GB")

    # 检查ollama桥接是否能真实连接
    try:
        from ollama_bridge import OllamaBridge, OllamaConfig
        config = OllamaConfig()
        bridge = OllamaBridge(config)

        if not bridge.check_ollama_status():
            issues.append("Ollama服务未运行，无法加载真实模型")
        else:
            available_models = bridge.list_available_models()
            deepseek_models = [m for m in available_models if 'deepseek' in m.lower()]

            if not deepseek_models:
                issues.append("Ollama中未找到DeepSeek模型")
            else:
                print(f"Ollama中有 {len(deepseek_models)} 个DeepSeek模型: {deepseek_models}")

    except Exception as e:
        issues.append(f"Ollama桥接测试失败: {e}")

    if issues:
        print("❌ DeepSeek加载问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ DeepSeek模型加载验证通过")

    return issues


def audit_crystallization_performance():
    """审计结晶化压缩的真实性能"""
    print("\n🔍 审计3: 验证结晶化压缩性能")
    print("=" * 50)

    issues = []

    try:
        from model_crystallization_engine import ModelCrystallizationEngine, CrystallizationConfig

        # 创建测试模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(768, 768) for _ in range(12)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = TestModel()
        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

        print("测试模型统计:")
        print(f"   参数数量: {original_params:,}")
        print(f"   模型大小: {original_size:.2f} MB")

        # 初始化结晶化引擎
        config = CrystallizationConfig(
            target_compression_ratio=8.0,
            max_memory_mb=1024
        )
        engine = ModelCrystallizationEngine(config)

        # 执行结晶化
        start_time = time.time()
        report = engine.crystallize_model(model, "test_model")
        crystallization_time = time.time() - start_time

        print("结晶化结果:")
        print(f"   压缩率: {report.get('compression_ratio', 1.0):.1f}x")
        print(f"   质量分数: {report.get('quality_score', 0.0):.3f}")
        print(f"   压缩时间: {crystallization_time:.2f} 秒")
        print(f"   内存使用: {report.get('memory_usage_mb', 0):.2f} MB")

        # 验证压缩率是否合理
        actual_ratio = report.get('compression_ratio', 1.0)
        if actual_ratio < 2.0:
            issues.append(f"压缩率过低: {actual_ratio:.1f}x (期望>8x)")

        # 验证质量保持
        quality = report.get('quality_score', 0.0)
        if quality < 0.8:
            issues.append(f"质量分数过低: {quality:.3f} (期望>0.9)")

        # 验证内存效率
        memory_mb = report.get('memory_usage_mb', 0)
        if memory_mb > 500:  # 500MB
            issues.append(f"内存使用过高: {memory_mb:.1f}MB")

    except Exception as e:
        issues.append(f"结晶化测试失败: {e}")

    if issues:
        print("❌ 结晶化性能问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 结晶化性能验证通过")

    return issues


def audit_memory_optimization():
    """审计内存优化声明的真实性"""
    print("\n🔍 审计4: 验证内存优化声明")
    print("=" * 50)

    issues = []

    # 检查内存使用情况
    memory = psutil.virtual_memory()
    print("当前系统内存状态:")
    print(f"   总内存: {memory.total / (1024**3):.2f} GB")
    print(f"   可用内存: {memory.available / (1024**3):.2f} GB")
    print(f"   使用率: {memory.percent:.1f}%")
    # 检查是否有内存监控
    try:
        from memory_safe_startup import MemorySafeStartupSystem, MemorySafeConfig

        config = MemorySafeConfig(max_memory_mb=2048)  # 2GB限制
        system = MemorySafeStartupSystem(config)

        if system.start_safe_startup():
            print("✅ 内存安全系统启动成功")

            # 检查内存预算
            budget = system.get_memory_budget()
            print("内存预算分配:")
            for key, value in budget.items():
                print(f"   {key}: {value:.1f} MB")

            # 验证预算合理性
            current_usage = budget.get("current_usage", 0)
            budget_limit = budget.get("budget_limit", 0)
            if current_usage > budget_limit:
                issues.append(f"预算超限: {current_usage:.1f}MB > {budget_limit:.1f}MB")

        else:
            issues.append("内存安全系统启动失败")

    except Exception as e:
        issues.append(f"内存系统测试失败: {e}")

    if issues:
        print("❌ 内存优化问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 内存优化验证通过")

    return issues


def run_real_benchmark():
    """运行真实的基准测试"""
    print("\n🔍 审计5: 运行真实基准测试")
    print("=" * 50)

    results = {}

    try:
        # 创建简单的测试模型
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

        # 测试推理时间
        model.eval()
        test_input = torch.randn(1, 100)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)

        # 实际测试
        start_time = time.time()
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        avg_time = (time.time() - start_time) / num_runs

        results['inference_time'] = avg_time
        results['model_params'] = sum(p.numel() for p in model.parameters())

        print("真实基准测试结果:")
        print(f"   平均推理时间: {avg_time:.6f} 秒")
        print(f"   模型参数: {results['model_params']:,}")

        # 与宣称的性能比较
        claimed_time = 0.001  # 假设的宣称时间
        if avg_time > claimed_time * 10:  # 10倍差距
            print(f"⚠️ 实际推理时间 ({avg_time:.6f}s) 远高于宣称水平")

    except Exception as e:
        print(f"❌ 真实基准测试失败: {e}")
        results['error'] = str(e)

    return results


def generate_audit_report(all_issues, benchmark_results):
    """生成审计报告"""
    print("\n📊 审计报告总结")
    print("=" * 50)

    total_issues = sum(len(issues) for issues in all_issues.values())

    if total_issues == 0:
        print("🎉 审计通过！未发现严重问题")
        print("   代码实现真实，性能数据可信")
    else:
        print(f"⚠️ 发现 {total_issues} 个潜在问题:")
        for category, issues in all_issues.items():
            if issues:
                print(f"   {category}: {len(issues)} 个问题")

    # 保存详细报告
    report = {
        'audit_timestamp': time.time(),
        'total_issues': total_issues,
        'issues_by_category': all_issues,
        'benchmark_results': benchmark_results,
        'system_info': {
            'python_version': os.sys.version,
            'torch_version': torch.__version__,
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    }

    with open('code_authenticity_audit_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("详细审计报告已保存到: code_authenticity_audit_report.json")


def main():
    """主审计函数"""
    print("🚀 H2Q-Evo 项目代码真实性审计")
    print("=" * 60)

    all_issues = {}

    # 1. 检查硬编码结果
    all_issues['hardcoded_results'] = audit_hardcoded_results()

    # 2. 检查DeepSeek真实性
    all_issues['deepseek_loading'] = audit_deepseek_loading()

    # 3. 验证结晶化性能
    all_issues['crystallization'] = audit_crystallization_performance()

    # 4. 验证内存优化
    all_issues['memory_optimization'] = audit_memory_optimization()

    # 5. 运行真实基准测试
    benchmark_results = run_real_benchmark()

    # 生成报告
    generate_audit_report(all_issues, benchmark_results)

    print("\n✨ 审计完成！")


if __name__ == "__main__":
    main()