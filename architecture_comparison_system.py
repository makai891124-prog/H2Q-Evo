#!/usr/bin/env python3
"""
H2Q-Evo 架构对比分析系统
比较核心机能力 vs 一般架构 vs 现有模型的性能和开销
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import requests

sys.path.append('/Users/imymm/H2Q-Evo')

from h2q_project.src.h2q.tokenizer_simple import default_tokenizer
from hierarchical_concept_encoder import HierarchicalConceptEncoder
from simple_hierarchical_encoder import SimpleHierarchicalEncoder
from real_code_completion_system import RealCodeCompletionSystem


class ArchitectureComparisonSystem:
    """架构对比分析系统"""

    def __init__(self):
        self.tokenizer = default_tokenizer
        self.results = {}

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统性能指标"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percentage": memory.percent,
            "cpu_percentage": cpu_percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
        }

    def test_core_machine_architecture(self) -> Dict[str, Any]:
        """测试核心机架构性能"""
        print("🔬 测试核心机架构 (H2Q-Evo Hierarchical Concept Encoder)")
        print("-" * 60)

        start_time = time.time()
        start_metrics = self.get_system_metrics()

        try:
            # 初始化核心机系统
            encoder = HierarchicalConceptEncoder(max_depth=3, compression_ratio=46.0)

            # 测试编码性能
            test_texts = [
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "class NeuralNetwork(nn.Module): def __init__(self): super().__init__()",
                "import torch; model = torch.nn.Linear(10, 1)"
            ]

            encoding_times = []
            for text in test_texts:
                encode_start = time.time()
                result = encoder.encode_hierarchical(text)
                encode_time = time.time() - encode_start
                encoding_times.append(encode_time)
                print(".4f")

            # 计算平均性能
            avg_encoding_time = sum(encoding_times) / len(encoding_times)
            total_time = time.time() - start_time
            end_metrics = self.get_system_metrics()

            return {
                "architecture": "core_machine",
                "status": "success",
                "total_time": total_time,
                "avg_encoding_time": avg_encoding_time,
                "memory_delta_mb": (end_metrics["memory_used_gb"] - start_metrics["memory_used_gb"]) * 1024,
                "cpu_overhead": end_metrics["cpu_percentage"] - start_metrics["cpu_percentage"],
                "gpu_memory_delta_mb": (end_metrics["gpu_memory_allocated"] - start_metrics["gpu_memory_allocated"]) * 1024,
                "compression_ratio": encoder.compression_ratio,
                "max_depth": encoder.max_depth,
                "uses_quaternion_mapping": True,
                "uses_wordnet": True,
                "uses_fractal_structure": True
            }

        except Exception as e:
            return {
                "architecture": "core_machine",
                "status": "failed",
                "error": str(e),
                "total_time": time.time() - start_time
            }

    def test_general_architecture(self) -> Dict[str, Any]:
        """测试一般架构性能"""
        print("🔬 测试一般架构 (Standard Transformer)")
        print("-" * 60)

        start_time = time.time()
        start_metrics = self.get_system_metrics()

        try:
            # 初始化一般架构系统
            system = RealCodeCompletionSystem()

            # 测试生成性能
            test_prompts = [
                "def calculate_fibonacci(n):",
                "class NeuralNetwork(nn.Module):",
                "import torch"
            ]

            generation_times = []
            for prompt in test_prompts:
                gen_start = time.time()
                result = system.generate_completion(prompt, max_length=20)
                gen_time = time.time() - gen_start
                generation_times.append(gen_time)
                print(".4f")

            # 计算平均性能
            avg_generation_time = sum(generation_times) / len(generation_times)
            total_time = time.time() - start_time
            end_metrics = self.get_system_metrics()

            return {
                "architecture": "general_transformer",
                "status": "success",
                "total_time": total_time,
                "avg_generation_time": avg_generation_time,
                "memory_delta_mb": (end_metrics["memory_used_gb"] - start_metrics["memory_used_gb"]) * 1024,
                "cpu_overhead": end_metrics["cpu_percentage"] - start_metrics["cpu_percentage"],
                "gpu_memory_delta_mb": (end_metrics["gpu_memory_allocated"] - start_metrics["gpu_memory_allocated"]) * 1024,
                "model_size_mb": self._get_model_size(system.model),
                "uses_quaternion_mapping": False,
                "uses_wordnet": False,
                "uses_fractal_structure": False
            }

        except Exception as e:
            return {
                "architecture": "general_transformer",
                "status": "failed",
                "error": str(e),
                "total_time": time.time() - start_time
            }

    def test_deepseek_model(self) -> Dict[str, Any]:
        """测试DeepSeek模型性能"""
        print("🔬 测试DeepSeek模型")
        print("-" * 60)

        start_time = time.time()
        start_metrics = self.get_system_metrics()

        try:
            # 测试DeepSeek API连接
            api_results = self._test_deepseek_api()

            total_time = time.time() - start_time
            end_metrics = self.get_system_metrics()

            return {
                "architecture": "deepseek_api",
                "status": "success" if api_results["connected"] else "api_unavailable",
                "total_time": total_time,
                "api_response_time": api_results.get("response_time", 0),
                "memory_delta_mb": (end_metrics["memory_used_gb"] - start_metrics["memory_used_gb"]) * 1024,
                "cpu_overhead": end_metrics["cpu_percentage"] - start_metrics["cpu_percentage"],
                "model_hosted_remotely": True,
                "uses_quaternion_mapping": False,
                "uses_wordnet": False,
                "uses_fractal_structure": False,
                **api_results
            }

        except Exception as e:
            return {
                "architecture": "deepseek_api",
                "status": "failed",
                "error": str(e),
                "total_time": time.time() - start_time
            }

    def _test_deepseek_api(self) -> Dict[str, Any]:
        """测试DeepSeek API连接"""
        try:
            # 测试基本连接
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])

                # 查找DeepSeek相关模型
                deepseek_models = [m for m in models if 'deepseek' in m['name'].lower()]

                if deepseek_models:
                    # 测试推理
                    test_payload = {
                        "model": deepseek_models[0]['name'],
                        "prompt": "def hello_world():",
                        "stream": False
                    }

                    infer_start = time.time()
                    response = requests.post("http://localhost:11434/api/generate",
                                           json=test_payload, timeout=30)
                    response_time = time.time() - infer_start

                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "connected": True,
                            "response_time": response_time,
                            "model_name": deepseek_models[0]['name'],
                            "model_size_gb": deepseek_models[0]['size'] / (1024**3),
                            "generated_text": result.get('response', '')[:100]
                        }

            return {"connected": False, "reason": "No DeepSeek models found or API unavailable"}

        except Exception as e:
            return {"connected": False, "error": str(e)}

    def _get_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024**2)

    def run_comparison_analysis(self) -> Dict[str, Any]:
        """运行完整的架构对比分析"""
        print("🚀 H2Q-Evo 架构对比分析")
        print("=" * 80)

        # 测试各个架构
        results = {}

        print("\n1. 测试核心机架构...")
        results["core_machine"] = self.test_core_machine_architecture()

        print("\n2. 测试一般架构...")
        results["general_transformer"] = self.test_general_architecture()

        print("\n3. 测试DeepSeek模型...")
        results["deepseek"] = self.test_deepseek_model()

        # 生成分析报告
        analysis = self._generate_analysis_report(results)

        # 保存结果
        self._save_results(results, analysis)

        return {"results": results, "analysis": analysis}

    def _generate_analysis_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告"""
        analysis = {
            "summary": {},
            "performance_comparison": {},
            "resource_efficiency": {},
            "capability_analysis": {},
            "recommendations": []
        }

        # 成功率分析
        successful_architectures = [k for k, v in results.items() if v.get("status") == "success"]
        analysis["summary"]["successful_architectures"] = successful_architectures
        analysis["summary"]["success_rate"] = len(successful_architectures) / len(results)

        # 性能对比
        if "core_machine" in successful_architectures and "general_transformer" in successful_architectures:
            core_time = results["core_machine"]["avg_encoding_time"]
            general_time = results["general_transformer"]["avg_generation_time"]

            analysis["performance_comparison"] = {
                "core_machine_vs_general": {
                    "speed_ratio": core_time / general_time if general_time > 0 else float('inf'),
                    "core_machine_faster": core_time < general_time
                }
            }

        # 资源效率分析
        for arch_name, arch_result in results.items():
            if arch_result.get("status") == "success":
                analysis["resource_efficiency"][arch_name] = {
                    "memory_efficiency": arch_result.get("memory_delta_mb", 0),
                    "cpu_efficiency": arch_result.get("cpu_overhead", 0),
                    "gpu_efficiency": arch_result.get("gpu_memory_delta_mb", 0)
                }

        # 能力分析
        analysis["capability_analysis"] = {
            "core_machine_uses_advanced_features": results.get("core_machine", {}).get("uses_quaternion_mapping", False),
            "general_architecture_simple": not results.get("general_transformer", {}).get("uses_quaternion_mapping", True),
            "deepseek_hosted_remotely": results.get("deepseek", {}).get("model_hosted_remotely", False)
        }

        # 推荐
        if analysis["capability_analysis"]["core_machine_uses_advanced_features"]:
            analysis["recommendations"].append("核心机架构提供了先进的数学建模能力，适合需要复杂概念理解的任务")

        if analysis["summary"]["success_rate"] < 1.0:
            analysis["recommendations"].append("某些架构可能需要额外的设置或依赖")

        return analysis

    def _save_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """保存分析结果"""
        output = {
            "timestamp": time.time(),
            "system_info": self.get_system_metrics(),
            "results": results,
            "analysis": analysis
        }

        output_path = "/Users/imymm/H2Q-Evo/architecture_comparison_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存到: {output_path}")


def main():
    """主函数"""
    analyzer = ArchitectureComparisonSystem()
    results = analyzer.run_comparison_analysis()

    # 打印总结
    print("\n📊 分析总结")
    print("=" * 80)

    analysis = results["analysis"]

    print(f"成功架构数量: {len(analysis['summary']['successful_architectures'])}/{len(results['results'])}")
    print(f"成功率: {analysis['summary']['success_rate']:.1%}")

    print("\n🏆 架构能力对比:")
    for arch_name, arch_result in results["results"].items():
        status = arch_result.get("status", "unknown")
        print(f"  {arch_name}: {status}")
        if status == "success":
            if "avg_encoding_time" in arch_result:
                print(f"    📏 平均编码时间: {arch_result['avg_encoding_time']:.4f}s")
            elif "avg_generation_time" in arch_result:
                print(f"    📏 平均生成时间: {arch_result['avg_generation_time']:.4f}s")
            if arch_result.get("uses_quaternion_mapping"):
                print("    ✅ 使用四元数球面映射")
            if arch_result.get("uses_wordnet"):
                print("    ✅ 使用WordNet语义网络")
            if arch_result.get("uses_fractal_structure"):
                print("    ✅ 使用分形结构")

    print("\n💡 关键发现:")
    for rec in analysis.get("recommendations", []):
        print(f"  • {rec}")

    print("\n✅ 架构对比分析完成!")


if __name__ == "__main__":
    main()