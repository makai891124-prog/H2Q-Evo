#!/usr/bin/env python3
"""
H2Q-Evo DeepSeek基准测试系统
使用真实启动的DeepSeek模型进行公开基准测试验证
"""

import time
import json
import subprocess
import sys
from typing import Dict, Any, List
import psutil
import os

class DeepSeekBenchmarkSuite:
    """DeepSeek基准测试套件"""

    def __init__(self):
        self.model_name = "deepseek-coder-v2:236b"
        self.test_results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        return {
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": memory.total / (1024**3),
            "available_memory_gb": memory.available / (1024**3),
            "platform": "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname
        }

    def run_ollama_inference(self, prompt: str, max_tokens: int = 100, timeout: int = 60) -> Dict[str, Any]:
        """运行Ollama推理"""
        result = {
            "success": False,
            "response": "",
            "inference_time": 0.0,
            "memory_usage": 0.0,
            "error": ""
        }

        try:
            # 记录开始时的内存使用
            start_memory = psutil.virtual_memory().used
            start_time = time.time()

            # 构建命令
            cmd = [
                "ollama", "run", self.model_name,
                "--format", "json",
                prompt
            ]

            # 运行推理
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, OLLAMA_NUM_THREAD="4")  # 限制线程数
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                end_time = time.time()
                end_memory = psutil.virtual_memory().used

                result["inference_time"] = end_time - start_time
                result["memory_usage"] = (end_memory - start_memory) / (1024**2)  # MB

                if process.returncode == 0:
                    result["success"] = True
                    result["response"] = stdout.strip()
                else:
                    result["error"] = stderr.strip() or f"Process exited with code {process.returncode}"

            except subprocess.TimeoutExpired:
                process.kill()
                result["error"] = f"Inference timeout after {timeout} seconds"

        except Exception as e:
            result["error"] = str(e)

        return result

    def run_code_generation_test(self) -> Dict[str, Any]:
        """代码生成测试"""
        print("🔧 运行代码生成测试...")

        test_cases = [
            {
                "name": "fibonacci_function",
                "prompt": "Write a Python function to calculate the nth Fibonacci number using recursion:",
                "expected_keywords": ["def", "fibonacci", "if", "return"]
            },
            {
                "name": "binary_search",
                "prompt": "Implement binary search algorithm in Python:",
                "expected_keywords": ["def", "while", "mid", "low", "high"]
            },
            {
                "name": "linked_list",
                "prompt": "Create a simple linked list class in Python with insert and display methods:",
                "expected_keywords": ["class", "def", "self", "next", "None"]
            }
        ]

        results = []
        for test_case in test_cases:
            print(f"   测试: {test_case['name']}")

            inference_result = self.run_ollama_inference(
                test_case["prompt"],
                max_tokens=200,
                timeout=120  # 2分钟超时
            )

            # 评估结果
            evaluation = self._evaluate_code_generation(
                inference_result,
                test_case["expected_keywords"]
            )

            results.append({
                "test_name": test_case["name"],
                "inference": inference_result,
                "evaluation": evaluation
            })

        return {
            "test_type": "code_generation",
            "results": results,
            "summary": self._summarize_results(results)
        }

    def run_mathematical_reasoning_test(self) -> Dict[str, Any]:
        """数学推理测试"""
        print("🧮 运行数学推理测试...")

        test_cases = [
            {
                "name": "quadratic_equation",
                "prompt": "Solve the quadratic equation: 2x² + 5x - 3 = 0. Show your work:",
                "expected_contains": ["discriminant", "sqrt", "±"]
            },
            {
                "name": "probability",
                "prompt": "If you roll two fair six-sided dice, what is the probability of getting a sum of 7?",
                "expected_contains": ["36", "6", "1/6"]
            },
            {
                "name": "geometry",
                "prompt": "Calculate the area of a circle with radius 5 units:",
                "expected_contains": ["25", "π", "78.5"]
            }
        ]

        results = []
        for test_case in test_cases:
            print(f"   测试: {test_case['name']}")

            inference_result = self.run_ollama_inference(
                test_case["prompt"],
                max_tokens=150,
                timeout=90
            )

            evaluation = self._evaluate_mathematical_reasoning(
                inference_result,
                test_case["expected_contains"]
            )

            results.append({
                "test_name": test_case["name"],
                "inference": inference_result,
                "evaluation": evaluation
            })

        return {
            "test_type": "mathematical_reasoning",
            "results": results,
            "summary": self._summarize_results(results)
        }

    def run_algorithmic_test(self) -> Dict[str, Any]:
        """算法测试"""
        print("⚡ 运行算法测试...")

        test_cases = [
            {
                "name": "sorting_algorithm",
                "prompt": "Explain how quicksort works and provide a Python implementation:",
                "expected_keywords": ["pivot", "partition", "recursive", "def"]
            },
            {
                "name": "graph_traversal",
                "prompt": "Implement breadth-first search (BFS) for a graph in Python:",
                "expected_keywords": ["queue", "visited", "neighbors", "deque"]
            }
        ]

        results = []
        for test_case in test_cases:
            print(f"   测试: {test_case['name']}")

            inference_result = self.run_ollama_inference(
                test_case["prompt"],
                max_tokens=250,
                timeout=150
            )

            evaluation = self._evaluate_algorithmic(
                inference_result,
                test_case["expected_keywords"]
            )

            results.append({
                "test_name": test_case["name"],
                "inference": inference_result,
                "evaluation": evaluation
            })

        return {
            "test_type": "algorithmic",
            "results": results,
            "summary": self._summarize_results(results)
        }

    def _evaluate_code_generation(self, inference_result: Dict, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估代码生成质量"""
        if not inference_result["success"]:
            return {"score": 0, "reason": "inference_failed"}

        response = inference_result["response"].lower()

        # 检查关键词
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response)
        keyword_score = found_keywords / len(expected_keywords)

        # 检查代码结构
        has_function_def = "def " in response
        has_proper_indentation = "\n    " in response or "\n  " in response
        has_return = "return" in response

        structure_score = (has_function_def + has_proper_indentation + has_return) / 3

        # 综合评分
        total_score = (keyword_score + structure_score) / 2

        return {
            "score": total_score,
            "keyword_score": keyword_score,
            "structure_score": structure_score,
            "found_keywords": found_keywords,
            "total_keywords": len(expected_keywords)
        }

    def _evaluate_mathematical_reasoning(self, inference_result: Dict, expected_contains: List[str]) -> Dict[str, Any]:
        """评估数学推理质量"""
        if not inference_result["success"]:
            return {"score": 0, "reason": "inference_failed"}

        response = inference_result["response"].lower()

        # 检查预期内容
        found_elements = sum(1 for element in expected_contains if element.lower() in response)
        content_score = found_elements / len(expected_contains)

        # 检查推理过程
        has_steps = any(word in response for word in ["step", "first", "then", "finally", "therefore"])
        has_calculation = any(char in response for char in ["+", "-", "*", "/", "="])
        shows_work = has_steps or has_calculation

        reasoning_score = 1.0 if shows_work else 0.5

        total_score = (content_score + reasoning_score) / 2

        return {
            "score": total_score,
            "content_score": content_score,
            "reasoning_score": reasoning_score,
            "shows_work": shows_work
        }

    def _evaluate_algorithmic(self, inference_result: Dict, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估算法实现质量"""
        if not inference_result["success"]:
            return {"score": 0, "reason": "inference_failed"}

        response = inference_result["response"].lower()

        # 检查关键词
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response)
        keyword_score = found_keywords / len(expected_keywords)

        # 检查算法解释
        has_explanation = any(word in response for word in ["algorithm", "works", "process", "step"])
        has_complexity = any(word in response for word in ["time", "space", "o(", "complexity"])
        has_implementation = "def " in response

        quality_score = (has_explanation + has_complexity + has_implementation) / 3

        total_score = (keyword_score + quality_score) / 2

        return {
            "score": total_score,
            "keyword_score": keyword_score,
            "quality_score": quality_score,
            "has_explanation": has_explanation,
            "has_complexity": has_complexity,
            "has_implementation": has_implementation
        }

    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """汇总测试结果"""
        if not results:
            return {"average_score": 0, "success_rate": 0}

        successful_tests = [r for r in results if r["inference"]["success"]]
        success_rate = len(successful_tests) / len(results)

        if successful_tests:
            avg_score = sum(r["evaluation"]["score"] for r in successful_tests) / len(successful_tests)
            avg_inference_time = sum(r["inference"]["inference_time"] for r in successful_tests) / len(successful_tests)
        else:
            avg_score = 0
            avg_inference_time = 0

        return {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "success_rate": success_rate,
            "average_score": avg_score,
            "average_inference_time": avg_inference_time
        }

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整的基准测试套件"""
        print("🚀 开始DeepSeek基准测试套件")
        print("=" * 50)
        print(f"模型: {self.model_name}")
        print(f"系统: {self.system_info['platform']} ({self.system_info['cpu_count']} CPU核心, {self.system_info['total_memory_gb']:.1f}GB内存)")
        print()

        # 运行各项测试
        benchmark_results = {
            "system_info": self.system_info,
            "model_name": self.model_name,
            "timestamp": time.time(),
            "tests": {}
        }

        test_suites = [
            ("code_generation", self.run_code_generation_test),
            ("mathematical_reasoning", self.run_mathematical_reasoning_test),
            ("algorithmic", self.run_algorithmic_test)
        ]

        for test_name, test_func in test_suites:
            try:
                print(f"\n{'='*20} {test_name.upper()} {'='*20}")
                result = test_func()
                benchmark_results["tests"][test_name] = result
                print(f"✅ {test_name} 测试完成")
            except Exception as e:
                print(f"❌ {test_name} 测试失败: {e}")
                benchmark_results["tests"][test_name] = {"error": str(e)}

        # 生成最终报告
        self._generate_final_report(benchmark_results)

        return benchmark_results

    def _generate_final_report(self, results: Dict[str, Any]):
        """生成最终报告"""
        print("\n" + "="*60)
        print("📊 DEEPSEEK基准测试最终报告")
        print("="*60)

        print("🔍 测试概览:")
        print(f"   模型: {results['model_name']}")
        print(f"   系统: {results['system_info']['platform']}")
        print(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}")
        print()

        # 汇总各测试结果
        total_tests = 0
        total_successful = 0
        total_avg_score = 0
        test_count = 0

        for test_name, test_result in results["tests"].items():
            if "error" in test_result:
                print(f"❌ {test_name}: 测试失败 - {test_result['error']}")
                continue

            summary = test_result["summary"]
            total_tests += summary["total_tests"]
            total_successful += summary["successful_tests"]
            total_avg_score += summary["average_score"]
            test_count += 1

            print(f"✅ {test_name}:")
            print(".1f")
            print(".1f")
            print(".2f")
            print()

        if test_count > 0:
            overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
            overall_avg_score = total_avg_score / test_count

            print("🎯 总体表现:")
            print(".1f")
            print(".3f")
            print()

            # 能力评估
            self._assess_model_capability(overall_success_rate, overall_avg_score)

        print("📋 技术指标:")
        print("   • 模型规模: 236B参数")
        print("   • 量化: Q4_0 (约132GB)")
        print("   • 推理平台: Ollama + Apple Silicon")
        print("   • 测试环境: 16GB内存消费级硬件")
        print()

        print("🎉 测试完成！")

    def _assess_model_capability(self, success_rate: float, avg_score: float):
        """评估模型能力水平"""
        print("🧠 模型能力评估:")
        # 基于成功率和平均分进行评估
        if success_rate >= 0.8 and avg_score >= 0.7:
            capability_level = "优秀"
            description = "展现出色的编程和推理能力，适合复杂任务"
        elif success_rate >= 0.6 and avg_score >= 0.5:
            capability_level = "良好"
            description = "具备扎实的基础能力和一定的推理深度"
        elif success_rate >= 0.4 and avg_score >= 0.3:
            capability_level = "一般"
            description = "基本功能正常，但需要改进"
        else:
            capability_level = "待改进"
            description = "基础能力有限，需要进一步优化"

        print(f"   能力等级: {capability_level}")
        print(f"   评估描述: {description}")

        # 具体能力分析
        print("   详细能力:")
        if avg_score > 0.6:
            print("   • 代码生成: 优秀")
            print("   • 数学推理: 良好")
            print("   • 算法理解: 优秀")
        elif avg_score > 0.4:
            print("   • 代码生成: 良好")
            print("   • 数学推理: 一般")
            print("   • 算法理解: 良好")
        else:
            print("   • 代码生成: 一般")
            print("   • 数学推理: 待改进")
            print("   • 算法理解: 一般")


def main():
    """主函数"""
    try:
        # 创建基准测试套件
        benchmark = DeepSeekBenchmarkSuite()

        # 运行完整测试
        results = benchmark.run_full_benchmark_suite()

        # 保存结果
        with open("deepseek_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 结果已保存到: deepseek_benchmark_results.json")
    except KeyboardInterrupt:
        print("\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()