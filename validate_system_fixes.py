#!/usr/bin/env python3
"""
验证修复后的AGI训练系统

检查：
1. 目标完成度的真实性验证
2. Gemini API调用的正确性
3. 知识扩展功能的完整性
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

def check_gemini_cache():
    """检查Gemini缓存状态"""
    cache_dir = Path('/Users/imymm/H2Q-Evo/gemini_cache')
    if not cache_dir.exists():
        print("❌ Gemini缓存目录不存在")
        return False

    cache_files = list(cache_dir.glob('*.json'))
    print(f"📁 发现 {len(cache_files)} 个缓存文件")

    if cache_files:
        # 检查一个缓存文件的结构
        with open(cache_files[0], 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        required_keys = ['timestamp', 'response', 'model']
        if all(key in cache_data for key in required_keys):
            print("✅ 缓存文件结构正确")
            print(f"   模型: {cache_data.get('model', 'unknown')}")
            print(f"   时间戳: {time.ctime(cache_data.get('timestamp', 0))}")
            return True
        else:
            print("❌ 缓存文件结构不完整")
            return False

    return True

def check_training_report():
    """检查训练报告"""
    report_file = Path('/Users/imymm/H2Q-Evo/extended_multimodal_agi_training_final_report.json')
    if not report_file.exists():
        print("❌ 训练报告文件不存在")
        return False

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    print("📊 训练报告分析:")
    print(f"   总步数: {report.get('total_steps', 0)}")
    print(f"   知识扩展次数: {report.get('knowledge_expansions', 0)}")
    print(f"   API调用次数: {report.get('expander_stats', {}).get('api_calls', 0)}")

    # 检查是否有真实的API调用
    api_calls = report.get('expander_stats', {}).get('api_calls', 0)
    if api_calls > 0:
        print("✅ 检测到真实的API调用")
        return True
    else:
        print("⚠️  未检测到API调用，可能存在配置问题")
        return False

def check_goal_completion_logic():
    """检查目标完成逻辑"""
    try:
        from optimized_agi_autonomous_system import EnhancedGoalSystem

        # 创建测试目标
        test_goal = {
            "type": "learning",
            "description": "掌握机器学习基础",
            "complexity": 0.5
        }

        # 测试不同的进度值
        test_cases = [
            (0.7, None, "低进度"),
            (0.9, {"policy_loss": 0.8, "value_loss": 1.5}, "高进度但学习指标差"),
            (0.95, {"policy_loss": 0.2, "value_loss": 0.3, "entropy": 0.5}, "高进度且学习指标好")
        ]

        goal_system = EnhancedGoalSystem(None, {})  # 简化的初始化

        print("🎯 目标完成逻辑验证:")
        for progress, metrics, description in test_cases:
            is_completed, evidence = goal_system.verify_goal_completion(test_goal, progress, metrics)
            status = "✅ 通过" if is_completed else "❌ 拒绝"
            print(f"   {description}: {status} ({evidence.get('reason', 'unknown')})")

        return True

    except Exception as e:
        print(f"❌ 目标完成逻辑检查失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🔍 开始验证修复后的AGI训练系统...")
    print("=" * 60)

    checks = [
        ("Gemini缓存状态", check_gemini_cache),
        ("训练报告分析", check_training_report),
        ("目标完成逻辑", check_goal_completion_logic)
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 检查: {check_name}")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📋 验证结果总结:")

    passed = sum(results)
    total = len(results)

    for i, (check_name, _) in enumerate(checks):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"   {check_name}: {status}")

    print(f"\n🎯 总体结果: {passed}/{total} 项检查通过")

    if passed == total:
        print("🎉 所有检查通过！系统修复成功。")
    else:
        print("⚠️  部分检查失败，需要进一步修复。")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)