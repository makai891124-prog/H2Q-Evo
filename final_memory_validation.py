#!/usr/bin/env python3
"""
AGI内存优化最终验证
测试完整的AGI训练系统是否能在3GB内存限制内稳定运行
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import time
import psutil
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def check_memory_usage():
    """检查当前内存使用情况"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def test_memory_optimized_agi_system():
    """测试内存优化后的AGI系统"""
    print("🚀 开始AGI内存优化最终验证...")

    initial_memory = check_memory_usage()
    print(f"初始内存使用: {initial_memory:.1f} MB")

    try:
        # 1. 测试数据生成器
        print("\n📊 测试数据生成器...")
        from agi_data_generator import AGIDataGenerator

        data_generator = AGIDataGenerator()
        data_generator.initialize_model('microsoft/DialoGPT-medium')

        after_model_memory = check_memory_usage()
        print(f"模型加载后内存: {after_model_memory:.1f} MB")

        # 生成少量数据进行测试
        data_file = data_generator.generate_incremental_data(
            evolution_generation=1,
            output_file='./final_test_data.jsonl'
        )

        after_data_memory = check_memory_usage()
        print(f"数据生成后内存: {after_data_memory:.1f} MB")

        # 检查生成的数据
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                lines = f.readlines()
            print(f"✅ 生成了 {len(lines)} 条训练数据")

        # 2. 测试训练器初始化
        print("\n🤖 测试训练器初始化...")
        from agi_persistent_evolution import PersistentAGITrainer, PersistentAGIConfig

        config = PersistentAGIConfig()
        trainer = PersistentAGITrainer(config)

        after_trainer_memory = check_memory_usage()
        print(f"训练器初始化后内存: {after_trainer_memory:.1f} MB")

        # 3. 测试单个训练周期
        print("\n🔄 测试训练周期执行...")
        if hasattr(trainer, 'run_training_cycle'):
            trainer.run_training_cycle()

            after_cycle_memory = check_memory_usage()
            print(f"训练周期后内存: {after_cycle_memory:.1f} MB")

            # 检查内存是否在合理范围内 (3GB = 3072MB)
            if after_cycle_memory < 3072:
                print("✅ 内存使用在3GB限制内")
            else:
                print(f"⚠️ 内存使用超出限制: {after_cycle_memory:.1f} MB")
        else:
            print("❌ 训练器缺少run_training_cycle方法")

        # 4. 清理测试
        print("\n🧹 执行清理...")
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()

        # 强制垃圾回收
        gc.collect()
        final_memory = check_memory_usage()
        print(f"清理后内存: {final_memory:.1f} MB")

        # 5. 总结
        print("\n📈 内存使用总结:")
        print(f"  初始: {initial_memory:.1f} MB")
        print(f"  模型加载后: {after_model_memory:.1f} MB (+{after_model_memory-initial_memory:.1f})")
        print(f"  数据生成后: {after_data_memory:.1f} MB (+{after_data_memory-after_model_memory:.1f})")
        print(f"  训练器初始化后: {after_trainer_memory:.1f} MB (+{after_trainer_memory-after_data_memory:.1f})")
        print(f"  最终: {final_memory:.1f} MB")

        max_memory = max(initial_memory, after_model_memory, after_data_memory, after_trainer_memory, final_memory)
        print(f"  峰值内存使用: {max_memory:.1f} MB")

        if max_memory < 3072:  # 3GB
            print("🎉 成功！AGI系统内存使用在3GB限制内")
            return True
        else:
            print(f"❌ 失败！峰值内存使用 {max_memory:.1f} MB 超出3GB限制")
            return False

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        test_files = [
            './final_test_data.jsonl',
            './agi_persistent_training/data/generated_data.jsonl'
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"已清理测试文件: {file}")

if __name__ == "__main__":
    success = test_memory_optimized_agi_system()
    if success:
        print("\n✅ AGI内存优化验证通过！可以开始AGI实验了。")
    else:
        print("\n❌ AGI内存优化验证失败！需要进一步优化。")
    sys.exit(0 if success else 1)