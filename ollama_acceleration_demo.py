#!/usr/bin/env python3
"""
H2Q-Evo Ollama加速演示脚本
"""

import sys
import time
import psutil
from typing import Dict, Any

# 添加项目路径
sys.path.append('/Users/imymm/H2Q-Evo')

from h2q_ollama_accelerator import get_h2q_accelerator


def main():
    """主函数"""
    print("🚀 H2Q-Evo Ollama加速演示")
    print("=" * 50)

    try:
        # 初始化加速器
        accelerator = get_h2q_accelerator(max_memory_gb=6.0)

        # 显示系统状态
        show_system_status()

        # 加速可用模型
        test_models = ["deepseek-coder:33b"]  # 可以根据需要修改

        for model_name in test_models:
            if accelerator._check_ollama_model(model_name):
                print(f"\n⚡ 加速模型: {model_name}")

                # 应用H2Q加速
                result = accelerator.accelerate_ollama_model(model_name)

                if result["success"]:
                    print("✅ 加速成功!")
                    print(f"   加速模型: {result['accelerated_model']}")
                    print(f"   压缩率: {result['compression_ratio']:.1f}x")
                    print(f"   内存节省: {result['memory_reduction_mb']:.0f}MB")
                    print(f"   吞吐量提升: {result['throughput_improvement']:.1f}x")

                    # 测试推理
                    test_inference(result['accelerated_model'])
                else:
                    print(f"❌ 加速失败: {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️  模型不存在: {model_name}")

        # 显示最终统计
        show_final_stats(accelerator)

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


def show_system_status():
    """显示系统状态"""
    print("\n📊 系统状态:")
    print(f"   CPU核心数: {psutil.cpu_count()}")
    print(f"   总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")


def test_inference(model_name: str):
    """测试推理"""
    print(f"\n🧪 测试推理: {model_name}")
    try:
        import subprocess

        test_prompt = "请解释什么是机器学习？"
        print(f"   提示: {test_prompt}")

        start_time = time.time()
        result = subprocess.run(
            ["ollama", "run", model_name, test_prompt],
            capture_output=True,
            text=True,
            input=test_prompt,
            timeout=30
        )
        end_time = time.time()

        if result.returncode == 0:
            response = result.stdout.strip()
            latency = end_time - start_time
            print(f"   ✅ 推理成功 (耗时: {latency:.2f}s)")
            print(f"   响应长度: {len(response)} 字符")
        else:
            print(f"   ❌ 推理失败: {result.stderr}")

    except Exception as e:
        print(f"   ❌ 推理错误: {e}")


def show_final_stats(accelerator):
    """显示最终统计"""
    print("\n📈 最终统计:")

    stats = accelerator.get_performance_stats()

    print(f"   活跃加速模型: {stats['active_models']}")
    print(f"   当前内存使用: {stats['memory_usage_mb']:.1f} MB")
    print(f"   总加速模型数: {stats['total_accelerated_models']}")

    perf = stats.get('performance_metrics', {})
    if perf:
        print(f"   总推理次数: {perf.get('total_inferences', 0)}")
        print(f"   平均延迟: {perf.get('average_latency_seconds', 0):.2f}s")
        print(f"   平均吞吐量: {perf.get('average_throughput_tokens_per_second', 0):.1f} tokens/s")


if __name__ == "__main__":
    main()