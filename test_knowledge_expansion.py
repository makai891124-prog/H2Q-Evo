#!/usr/bin/env python3
"""
测试知识扩展功能
"""

import sys
import asyncio
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

from extended_multimodal_agi_training import ExtendedMultimodalAGITrainer

async def test_knowledge_expansion():
    """测试知识扩展功能"""
    print("🧪 测试知识扩展功能...")

    trainer = ExtendedMultimodalAGITrainer()

    # 初始化系统
    trainer.initialize_system()

    # 手动调用知识扩展
    try:
        print("🔍 检查知识扩展条件...")
        print(f"   当前步数: 0")
        print(f"   上次扩展步数: {trainer.last_expansion_step}")
        print(f"   扩展间隔: {trainer.expansion_interval}")

        should_expand = 0 - trainer.last_expansion_step >= trainer.expansion_interval
        print(f"   应该扩展: {should_expand}")

        await trainer._perform_async_knowledge_expansion(0)
        print("✅ 知识扩展执行成功")

        # 检查统计信息
        stats = trainer.knowledge_expander.get_stats()
        print(f"📊 扩展器统计: {stats}")

        # 检查是否有新的缓存文件
        import os
        cache_dir = '/Users/imymm/H2Q-Evo/gemini_cache'
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            print(f"📁 缓存文件数量: {len(cache_files)}")

        return True
    except Exception as e:
        print(f"❌ 知识扩展执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_knowledge_expansion())
    print(f"🎯 测试结果: {'通过' if success else '失败'}")
    sys.exit(0 if success else 1)