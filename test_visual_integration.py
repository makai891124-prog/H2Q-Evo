#!/usr/bin/env python3
"""
测试新的视觉数据集成和处理功能
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

from extended_multimodal_agi_training import VisualDataLoader, AdvancedVisualProcessor

async def test_visual_data_integration():
    """测试视觉数据集成"""
    print("🧪 测试视觉数据集成功能...")

    # 测试视觉数据加载器
    print("\n1. 测试视觉数据加载器...")
    visual_loader = VisualDataLoader(batch_size=2)

    print(f"   可用数据集: {visual_loader.available_datasets}")

    # 加载图像批次
    try:
        image_batch = visual_loader.load_image_batch()
        print(f"   图像批次形状: {image_batch.shape}")
        print(f"   图像数据类型: {image_batch.dtype}")
        print(f"   图像值范围: [{image_batch.min():.3f}, {image_batch.max():.3f}]")
    except Exception as e:
        print(f"   ❌ 图像加载失败: {e}")

    # 加载视频批次
    try:
        video_batch = visual_loader.load_video_batch()
        print(f"   视频批次形状: {video_batch.shape}")
        print(f"   视频数据类型: {video_batch.dtype}")
        print(f"   视频值范围: [{video_batch.min():.3f}, {video_batch.max():.3f}]")
    except Exception as e:
        print(f"   ❌ 视频加载失败: {e}")

    # 获取描述
    captions = visual_loader.get_visual_captions(2)
    print(f"   生成的描述: {captions}")

    # 测试高级视觉处理器
    print("\n2. 测试高级视觉处理器...")
    device = 'cpu'  # 使用CPU避免MPS兼容性问题
    visual_processor = AdvancedVisualProcessor(device=device)

    # 测试图像分析
    try:
        if 'image_batch' in locals():
            print("   分析图像...")
            image_analysis = visual_processor.analyze_image_comprehensive(image_batch)
            print(f"   图像特征维度: {image_analysis['features'].shape}")
            print(f"   物体检测: {'objects' in image_analysis}")
            print(f"   场景理解: {'scene' in image_analysis}")
            print(f"   质量评分: {image_analysis.get('quality_score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 图像分析失败: {e}")

    # 测试视频分析
    try:
        if 'video_batch' in locals():
            print("   分析视频...")
            video_analysis = visual_processor.analyze_video_comprehensive(video_batch)
            print(f"   视频特征维度: {video_analysis['features'].shape}")
            print(f"   动作识别: {'actions' in video_analysis}")
            print(f"   运动模式: {'motion_patterns' in video_analysis}")
            print(f"   时间一致性: {video_analysis.get('temporal_consistency', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 视频分析失败: {e}")

    print("\n✅ 视觉数据集成测试完成")

async def test_learning_engine():
    """测试优化后的学习引擎"""
    print("\n3. 测试优化学习引擎...")

    try:
        from extended_multimodal_agi_training import (
            UnifiedBinaryFlowPerceptionCore,
            OptimizedHybridLearningEngine,
            AdvancedVisualProcessor
        )

        # 创建组件
        perception_core = UnifiedBinaryFlowPerceptionCore(dim=256, num_modalities=6)
        visual_processor = AdvancedVisualProcessor(device='cpu')
        learning_engine = OptimizedHybridLearningEngine(perception_core, visual_processor)

        # 启动预取
        await learning_engine.start_prefetch()

        # 测试学习批次生成
        print("   生成学习批次...")
        for step in range(3):
            batch = await learning_engine.get_learning_batch(step)
            print(f"   步骤 {step}: {batch['type']} - 模态: {list(batch.get('data', {}).keys())}")

        # 获取性能报告
        performance = learning_engine.get_performance_report()
        print(f"   学习效率: {performance['performance_metrics']['learning_efficiency']:.2%}")
        print(f"   模态平衡: {performance['performance_metrics']['modality_balance']:.2%}")

        # 停止预取
        await learning_engine.stop_prefetch()

        print("   ✅ 学习引擎测试通过")

    except Exception as e:
        print(f"   ❌ 学习引擎测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主测试函数"""
    print("🚀 开始视觉数据集成和处理测试")
    print("=" * 50)

    await test_visual_data_integration()
    await test_learning_engine()

    print("\n" + "=" * 50)
    print("🎯 所有测试完成")

if __name__ == "__main__":
    asyncio.run(main())