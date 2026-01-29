#!/usr/bin/env python3
"""
测试知识扩展API调用 - 运行超过30步的训练
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

from extended_multimodal_agi_training import ExtendedMultimodalAGITrainer

async def test_api_calls():
    """测试API调用是否在30步后触发"""
    print("🧪 测试知识扩展API调用触发...")

    # 创建训练器
    trainer = ExtendedMultimodalAGITrainer()

    # 初始化系统
    trainer.initialize_system()

    print("📊 初始统计:")
    initial_stats = trainer.knowledge_expander.get_stats()
    print(f"   API调用: {initial_stats['api_calls']}")

    # 运行35步训练
    print("🏃 运行35步训练...")
    for step in range(35):
        if trainer.agi_system:
            trainer.agi_system.step()

        # 执行知识扩展（同步方式）
        trainer._perform_knowledge_expansion_sync(step)

        if step % 10 == 0:
            print(f"   步骤 {step}: API调用 = {trainer.knowledge_expander.get_stats()['api_calls']}")

    # 检查最终统计
    final_stats = trainer.knowledge_expander.get_stats()
    print("📊 最终统计:")
    print(f"   API调用: {final_stats['api_calls']}")
    print(f"   缓存命中: {final_stats['cache_hits']}")
    print(f"   缓存未命中: {final_stats['cache_misses']}")
    print(f"   错误: {final_stats['errors']}")

    # 检查缓存文件变化
    import os
    cache_dir = '/Users/imymm/H2Q-Evo/gemini_cache'
    if os.path.exists(cache_dir):
        final_cache_files = len(os.listdir(cache_dir))
        print(f"   缓存文件数量: {final_cache_files}")

    # 验证结果
    api_calls_made = final_stats['api_calls'] > initial_stats['api_calls']
    print(f"🎯 测试结果: {'通过' if api_calls_made else '失败'}")
    print(f"   API调用是否增加: {api_calls_made}")

    return api_calls_made

if __name__ == "__main__":
    result = asyncio.run(test_api_calls())