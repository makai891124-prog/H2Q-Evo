#!/usr/bin/env python3
"""
简化的DeepSeek模型测试脚本

测试真实DeepSeek模型的基本功能，避免内存问题
"""

import requests
import json
import time
import psutil
from typing import Dict, Any


def get_memory_info() -> Dict[str, float]:
    """获取内存信息"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percentage": memory.percent
    }


def test_ollama_basic():
    """测试ollama基本功能"""
    print("🔍 测试Ollama基本功能")
    print("=" * 40)

    try:
        # 测试API连接
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Ollama API连接正常")
            print(f"   可用模型数量: {len(data.get('models', []))}")

            for model in data.get('models', []):
                print(f"   - {model['name']}: {model['size'] / (1024**3):.1f} GB")
        else:
            print(f"❌ API响应错误: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

    return True


def test_deepseek_simple():
    """测试DeepSeek简单推理"""
    print("\n🧪 测试DeepSeek简单推理")
    print("=" * 40)

    memory_before = get_memory_info()
    print("📊 测试前内存状态:")
    print(f"   总内存: {memory_before['total_gb']:.2f} GB")
    print(f"   可用内存: {memory_before['available_gb']:.2f} GB")
    print(f"   使用率: {memory_before['percentage']:.1f}%")
    # 超简单的测试请求
    payload = {
        "model": "deepseek-coder:6.7b",  # 使用更小的模型
        "prompt": "Write 'Hello World' in Python",
        "stream": False,
        "options": {
            "num_predict": 20,  # 只生成20个token
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    print("🚀 发送推理请求...")
    start_time = time.time()

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60  # 60秒超时
        )

        inference_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            output = result.get('response', '')

            memory_after = get_memory_info()

            print("✅ 推理成功!")
            print(f"   推理时间: {inference_time:.3f} 秒")
            print(f"   生成内容长度: {len(output)} 字符")
            print(f"   内存使用增量: {memory_after['used_gb'] - memory_before['used_gb']:.2f} GB")

            # 显示部分输出
            print(f"   输出预览: {output[:100]}{'...' if len(output) > 100 else ''}")

            return {
                "success": True,
                "inference_time": inference_time,
                "output_length": len(output),
                "memory_delta_gb": memory_after['used_gb'] - memory_before['used_gb'],
                "output": output
            }
        else:
            print(f"❌ 推理失败: HTTP {response.status_code}")
            print(f"   响应: {response.text[:200]}")
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        print("❌ 请求超时 (60秒)")
        return {"success": False, "error": "timeout"}

    except Exception as e:
        inference_time = time.time() - start_time
        print(f"❌ 推理失败 ({inference_time:.3f} 秒)")
        print(f"   错误: {e}")
        return {"success": False, "error": str(e)}


def test_crystallization_integration():
    """测试结晶化系统集成"""
    print("\n💎 测试结晶化系统集成")
    print("=" * 40)

    try:
        from model_crystallization_engine import ModelCrystallizationEngine, CrystallizationConfig
        import torch
        import torch.nn as nn

        # 创建一个小的测试模型
        class TinyTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 64)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=64, nhead=4, dim_feedforward=128, batch_first=True
                )
                self.output = nn.Linear(64, 1000)

            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)

        model = TinyTestModel()
        print(f"   测试模型参数: {sum(p.numel() for p in model.parameters()):,}")

        # 初始化结晶化引擎
        config = CrystallizationConfig(
            target_compression_ratio=4.0,
            max_memory_mb=512
        )

        engine = ModelCrystallizationEngine(config)

        # 执行结晶化
        print("   执行模型结晶化...")
        start_time = time.time()
        report = engine.crystallize_model(model, "tiny_test")
        crystallization_time = time.time() - start_time

        print("✅ 结晶化完成!")
        print(f"   压缩比: {report.get('compression_ratio', 1.0):.1f}x")
        print(f"   质量分数: {report.get('quality_score', 0.0):.3f}")
        print(f"   结晶化时间: {crystallization_time:.2f} 秒")
        return {
            "success": True,
            "compression_ratio": report.get('compression_ratio', 1.0),
            "quality_score": report.get('quality_score', 0.0),
            "crystallization_time": crystallization_time
        }

    except Exception as e:
        print(f"❌ 结晶化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 H2Q-Evo DeepSeek综合测试")
    print("=" * 60)

    results = {
        "timestamp": time.time(),
        "system_memory": get_memory_info(),
        "tests": {}
    }

    # 1. 测试Ollama基本功能
    results["tests"]["ollama_basic"] = test_ollama_basic()

    # 2. 测试DeepSeek简单推理
    results["tests"]["deepseek_simple"] = test_deepseek_simple()

    # 3. 测试结晶化系统
    results["tests"]["crystallization"] = test_crystallization_integration()

    # 保存结果
    with open("deepseek_comprehensive_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 总结报告
    print("\n📊 测试总结报告")
    print("=" * 60)

    basic_ok = results["tests"]["ollama_basic"]
    deepseek_ok = results["tests"]["deepseek_simple"].get("success", False)
    crystal_ok = results["tests"]["crystallization"].get("success", False)

    print(f"   Ollama基本功能: {'✅' if basic_ok else '❌'}")
    print(f"   DeepSeek推理: {'✅' if deepseek_ok else '❌'}")
    print(f"   结晶化系统: {'✅' if crystal_ok else '❌'}")

    if deepseek_ok:
        deepseek_result = results["tests"]["deepseek_simple"]
        print("\n   DeepSeek性能指标:")
        print(f"     推理时间: {deepseek_result['inference_time']:.3f} 秒")
        print(f"     输出长度: {deepseek_result['output_length']} 字符")
        print(f"     内存增量: {deepseek_result['memory_delta_gb']:.2f} GB")

    if crystal_ok:
        crystal_result = results["tests"]["crystallization"]
        print("\n   结晶化性能指标:")
        print(f"     压缩比: {crystal_result['compression_ratio']:.1f}x")
        print(f"     质量分数: {crystal_result['quality_score']:.3f}")
        print(f"     结晶化时间: {crystal_result['crystallization_time']:.2f} 秒")

    print("\n详细结果已保存: deepseek_comprehensive_test_results.json")
    print("\n🎯 结论:")
    if basic_ok and deepseek_ok and crystal_ok:
        print("   ✅ 所有系统正常工作，DeepSeek模型可以真实运行！")
    else:
        print("   ⚠️ 部分系统存在问题，需要进一步调试")

    return results


if __name__ == "__main__":
    run_comprehensive_test()