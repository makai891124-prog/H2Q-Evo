#!/usr/bin/env python3
"""
内存优化集成测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import time
import psutil
from memory_optimized_system import MemoryOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def test_memory_optimizer():
    """测试内存优化器"""
    print("🧪 开始内存优化器测试...")

    # 初始化内存优化器
    optimizer = MemoryOptimizer(max_memory_gb=3.0)
    optimizer.start_monitoring()

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"初始内存使用: {initial_memory:.1f} MB")

    # 模拟内存密集操作
    print("创建测试数据...")
    test_data = []
    for i in range(500):
        test_data.append([i] * 2000)  # 创建一些大列表

    peak_memory = process.memory_info().rss / 1024 / 1024
    print(f"数据创建后内存: {peak_memory:.1f} MB (增加 {peak_memory - initial_memory:.1f} MB)")

    # 主动清理
    print("清理数据...")
    del test_data
    import gc
    gc.collect()

    cleaned_memory = process.memory_info().rss / 1024 / 1024
    print(f"清理后内存: {cleaned_memory:.1f} MB")

    # 测试优化器的监控
    time.sleep(3)

    optimizer.stop_monitoring()
    print("✅ 内存优化器测试完成")

def test_data_generator_memory():
    """测试数据生成器的内存优化"""
    print("\n🧪 开始数据生成器内存测试...")

    from agi_data_generator import AGIDataGenerator

    generator = AGIDataGenerator()
    generator.initialize_model('microsoft/DialoGPT-medium')

    process = psutil.Process()
    before_memory = process.memory_info().rss / 1024 / 1024
    print(f"初始化后内存: {before_memory:.1f} MB")

    # 生成少量数据
    output_file = generator.generate_training_data(num_samples=10, output_file='./memory_test_data.jsonl')

    after_memory = process.memory_info().rss / 1024 / 1024
    print(f"生成数据后内存: {after_memory:.1f} MB (增加 {after_memory - before_memory:.1f} MB)")

    # 检查生成的文件
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
        print(f"生成了 {len(lines)} 条数据")
        os.remove(output_file)  # 清理测试文件

    print("✅ 数据生成器内存测试完成")

if __name__ == "__main__":
    test_memory_optimizer()
    test_data_generator_memory()
    print("\n🎉 所有内存优化测试完成！")