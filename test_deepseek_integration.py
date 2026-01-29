#!/usr/bin/env python3
"""
测试DeepSeek本地集成和成本跟踪功能
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from evolution_system import H2QNexus

async def test_deepseek_integration():
    """测试DeepSeek集成和成本跟踪"""
    print("🧪 测试DeepSeek本地集成和成本跟踪...")

    # 初始化系统
    nexus = H2QNexus()

    # 检查DeepSeek集成状态
    if nexus.deepseek_integration is not None:
        print("✅ DeepSeek本地集成已初始化")
    else:
        print("❌ DeepSeek本地集成未初始化")

    # 测试成本统计
    initial_stats = nexus.get_cost_stats()
    print(f"📊 初始成本统计: {initial_stats}")

    # 测试推理（这会触发DeepSeek或API）
    try:
        prompt = "Hello, can you tell me about AGI evolution?"
        result = await nexus.api_inference(prompt)
        print(f"🤖 推理结果: {result[:100]}...")

        # 检查成本统计更新
        final_stats = nexus.get_cost_stats()
        print(f"📊 最终成本统计: {final_stats}")

        if final_stats['cost_savings'] > initial_stats['cost_savings']:
            print("💰 成功记录成本节省！")
        elif final_stats['api_costs'] > initial_stats['api_costs']:
            print("💸 记录了API使用成本")

    except Exception as e:
        print(f"❌ 推理测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepseek_integration())