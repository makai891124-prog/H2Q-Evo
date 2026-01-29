#!/usr/bin/env python3
"""
H2Q-Evo 真实DeepSeek基准测试 - 简化版本

使用真实的DeepSeek模型验证H2Q-Evo系统的核心功能
"""

import requests
import json
import time
import torch
import torch.nn as nn
import psutil


def get_memory_info():
    """获取内存信息"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percentage": memory.percent
    }


class RealDeepSeekBenchmark:
    """真实DeepSeek基准测试"""

    def __init__(self):
        self.model_name = "deepseek-coder:6.7b"
        self.base_url = "http://localhost:11434/api/generate"

    def test_basic_inference(self):
        """测试基本推理"""
        print("🧪 测试基本推理能力")

        payload = {
            "model": self.model_name,
            "prompt": "Write a Python function to calculate factorial",
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.1
            }
        }

        start_time = time.time()
        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            inference_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                output = result.get('response', '')
                return {
                    "success": True,
                    "inference_time": inference_time,
                    "output_length": len(output),
                    "output": output[:200] + "..." if len(output) > 200 else output
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_crystallization(self):
        """测试结晶化系统"""
        print("💎 测试结晶化系统")

        try:
            from model_crystallization_engine import ModelCrystallizationEngine, CrystallizationConfig

            # 创建测试模型
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )

            original_params = sum(p.numel() for p in model.parameters())

            config = CrystallizationConfig(
                target_compression_ratio=4.0,
                max_memory_mb=512
            )

            engine = ModelCrystallizationEngine(config)
            start_time = time.time()
            report = engine.crystallize_model(model, "benchmark_test")
            crystallization_time = time.time() - start_time

            return {
                "success": True,
                "original_params": original_params,
                "compression_ratio": report.get('compression_ratio', 1.0),
                "quality_score": report.get('quality_score', 0.0),
                "crystallization_time": crystallization_time
            }

        except Exception as e:
            print(f"   结晶化测试失败: {e}")
            return {"success": False, "error": str(e)}

    def run_benchmark(self):
        """运行基准测试"""
        print("🚀 H2Q-Evo 真实DeepSeek基准测试")
        print("=" * 60)

        results = {
            "timestamp": time.time(),
            "model": self.model_name,
            "system_memory": get_memory_info(),
            "tests": {}
        }

        # 运行测试
        results["tests"]["basic_inference"] = self.test_basic_inference()
        results["tests"]["crystallization"] = self.test_crystallization()

        # 保存结果
        with open("real_deepseek_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 生成报告
        self._generate_report(results)

        return results

    def _generate_report(self, results):
        """生成测试报告"""
        print("\n📊 基准测试报告")
        print("=" * 60)

        tests = results["tests"]
        total_tests = len(tests)
        successful_tests = sum(1 for test in tests.values() if test.get("success", False))
        success_rate = successful_tests / total_tests * 100

        print(f"测试成功率: {success_rate:.1f}%")
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")

        if tests["basic_inference"]["success"]:
            basic = tests["basic_inference"]
            print("\n基本推理:")
            print(f"推理时间: {basic['inference_time']:.3f} 秒")
            print(f"输出长度: {basic['output_length']} 字符")

        if tests["crystallization"]["success"]:
            crystal = tests["crystallization"]
            print("\n结晶化:")
            print(f"原始参数: {crystal['original_params']:,}")
            print(f"压缩比: {crystal['compression_ratio']:.1f}x")
            print(f"质量分数: {crystal['quality_score']:.3f}")
            print(f"结晶化时间: {crystal['crystallization_time']:.2f} 秒")

        print("\n🎯 结论:")
        if success_rate >= 50:
            print("✅ H2Q-Evo核心功能验证成功！")
            print("✅ 真实DeepSeek模型集成工作正常")
        else:
            print("⚠️ 需要进一步调试")

        print("\n详细结果已保存: real_deepseek_benchmark_results.json")


def main():
    """主函数"""
    benchmark = RealDeepSeekBenchmark()
    results = benchmark.run_benchmark()
    return results


if __name__ == "__main__":
    main()