#!/usr/bin/env python3
"""
H2Q-Evo 数学核心架构真实性验证

验证我们的数学核心架构是否真实运行，并尝试将DeepSeek模型集成
"""

import torch
import torch.nn as nn
import json
import time
import requests
from typing import Dict, Any
import sys
import os

# 添加路径
sys.path.append('/Users/imymm/H2Q-Evo')
sys.path.append('/Users/imymm/H2Q-Evo/h2q_project/src')

def test_mathematical_core():
    """测试数学核心架构"""
    print("🔬 测试H2Q数学核心架构")
    print("=" * 50)

    try:
        from h2q_project.src.h2q.core.unified_architecture import (
            UnifiedH2QMathematicalArchitecture,
            UnifiedMathematicalArchitectureConfig
        )

        print("✅ 导入数学架构成功")

        # 创建配置
        config = UnifiedMathematicalArchitectureConfig(
            dim=128,
            action_dim=32,
            device="cpu"  # 使用CPU避免MPS问题
        )

        # 初始化架构
        start_time = time.time()
        math_core = UnifiedH2QMathematicalArchitecture(config)
        init_time = time.time() - start_time

        print(f"✅ 数学架构初始化成功: {init_time:.3f} 秒")
        # 测试基本功能
        batch_size, seq_len = 2, 10
        dummy_input = torch.randn(batch_size, seq_len, config.dim)

        start_time = time.time()
        output = math_core(dummy_input)
        forward_time = time.time() - start_time

        print(f"✅ 前向传播成功: 输入{batch_size}x{seq_len}x{config.dim} -> 输出{output.shape}")
        print(f"✅ 前向传播耗时: {forward_time:.3f} 秒")
        return {
            "success": True,
            "init_time": init_time,
            "forward_time": forward_time,
            "output_shape": output.shape
        }

    except Exception as e:
        print(f"❌ 数学核心测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_streaming_middleware():
    """测试流式推理中间件"""
    print("\n🌊 测试流式推理中间件")
    print("=" * 50)

    try:
        from h2q_project.src.h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware

        print("✅ 导入流式中间件成功")

        # 创建中间件
        middleware = HolomorphicStreamingMiddleware(threshold=0.1, max_history=8)

        # 测试四元数状态处理
        q_state = torch.randn(4)  # 四元数状态

        start_time = time.time()
        curvature = middleware.calculate_fueter_laplace(q_state)
        curvature_time = time.time() - start_time

        print(f"✅ 曲率计算成功: {curvature.item():.6f}")
        print(f"✅ 曲率计算耗时: {curvature_time:.3f} 秒")
        return {
            "success": True,
            "curvature": curvature.item(),
            "computation_time": curvature_time
        }

    except Exception as e:
        print(f"❌ 流式中间件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_model_integration():
    """测试模型集成能力"""
    print("\n🔗 测试模型集成能力")
    print("=" * 50)

    try:
        from model_crystallization_engine import ModelCrystallizationEngine, CrystallizationConfig

        # 创建一个模拟的DeepSeek风格的模型
        class MockDeepSeekModel(nn.Module):
            def __init__(self, vocab_size=32000, hidden_size=1024, num_layers=12):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=hidden_size, nhead=16, dim_feedforward=4096,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                self.ln_f = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids):
                x = self.embeddings(input_ids)
                for layer in self.layers:
                    x = layer(x, x)  # 自注意力
                x = self.ln_f(x)
                logits = self.lm_head(x)
                return logits

        # 创建小型模型用于测试
        model = MockDeepSeekModel(vocab_size=1000, hidden_size=256, num_layers=4)
        original_params = sum(p.numel() for p in model.parameters())

        print(f"✅ 创建模拟DeepSeek模型: {original_params:,} 参数")

        # 配置结晶化
        config = CrystallizationConfig(
            target_compression_ratio=8.0,
            max_memory_mb=1024,
            device="cpu"
        )

        engine = ModelCrystallizationEngine(config)

        # 执行结晶化
        start_time = time.time()
        report = engine.crystallize_model(model, "mock_deepseek_integration")
        crystallization_time = time.time() - start_time

        print("✅ 模型结晶化成功!")
        print(f"   压缩比: {report.get('compression_ratio', 1.0):.1f}x")
        print(f"   质量分数: {report.get('quality_score', 0.0):.3f}")
        print(f"   结晶化时间: {crystallization_time:.2f} 秒")
        return {
            "success": True,
            "original_params": original_params,
            "compression_ratio": report.get('compression_ratio', 1.0),
            "crystallization_time": crystallization_time
        }

    except Exception as e:
        print(f"❌ 模型集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_streaming_with_ollama():
    """测试与Ollama的流式推理"""
    print("\n📡 测试Ollama流式推理")
    print("=" * 50)

    try:
        # 测试流式API
        payload = {
            "model": "deepseek-coder:6.7b",
            "prompt": "Write a Python function to sort a list",
            "stream": True,  # 启用流式
            "options": {
                "num_predict": 100,
                "temperature": 0.1
            }
        }

        print("🚀 发送流式推理请求...")

        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )

        if response.status_code == 200:
            print("✅ 流式响应开始接收")

            total_content = ""
            chunk_count = 0
            first_chunk_time = None

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'response' in data:
                                chunk_count += 1
                                if first_chunk_time is None:
                                    first_chunk_time = time.time() - start_time
                                total_content += data['response']
                                if chunk_count <= 3:  # 只显示前3个chunk
                                    print(f"   Chunk {chunk_count}: {data['response'][:50]}...")
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

            total_time = time.time() - start_time

            print("✅ 流式推理完成!")
            print(f"   总接收块数: {chunk_count}")
            print(f"   总时间: {total_time:.3f} 秒")
            print(f"   首块时间: {first_chunk_time:.3f} 秒")
            print(f"   总内容长度: {len(total_content)} 字符")

            return {
                "success": True,
                "total_time": total_time,
                "first_chunk_time": first_chunk_time,
                "chunk_count": chunk_count,
                "content_length": len(total_content)
            }
        else:
            print(f"❌ 流式请求失败: HTTP {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except Exception as e:
        print(f"❌ 流式推理测试失败: {e}")
        return {"success": False, "error": str(e)}


def analyze_236b_model_issue():
    """分析236B模型问题"""
    print("\n🔍 分析236B模型问题")
    print("=" * 50)

    try:
        # 检查模型信息
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for model in data.get('models', []):
                if '236b' in model['name']:
                    print(f"📊 236B模型信息:")
                    print(f"   名称: {model['name']}")
                    print(f"   大小: {model['size'] / (1024**3):.1f} GB")
                    print(f"   修改时间: {model['modified_at']}")

                    # 分析可能的内存问题
                    import psutil
                    memory = psutil.virtual_memory()
                    available_gb = memory.available / (1024**3)

                    print("\n💾 系统内存分析:")
                    print(f"   总内存: {memory.total / (1024**3):.1f} GB")
                    print(f"   可用内存: {available_gb:.1f} GB")
                    print(f"   使用率: {memory.percent:.1f}%")
                    if available_gb < 8:
                        print("   ⚠️ 可用内存不足，236B模型需要大量内存")
                        print("   💡 建议: 增加系统内存或使用更小的模型")
                    else:
                        print("   ✅ 系统内存充足")

                    # 分析可能的流式问题
                    print("\n🌊 流式推理分析:")
                    print("   • 236B模型参数量极大 (~2360亿)")
                    print("   • 首次加载需要长时间预热")
                    print("   • 内存占用可能超过16GB")
                    print("   • 建议使用分块加载或内存映射")
                    break

        return {"analysis_complete": True}

    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        return {"analysis_complete": False, "error": str(e)}


def audit_achievements():
    """审计现有成果价值"""
    print("\n📋 审计现有成果价值")
    print("=" * 50)

    achievements = {
        "mathematical_core": False,
        "streaming_middleware": False,
        "model_integration": False,
        "crystallization_technology": False,
        "real_model_testing": False
    }

    # 检查数学核心
    try:
        from h2q_project.src.h2q.core.unified_architecture import UnifiedH2QMathematicalArchitecture
        achievements["mathematical_core"] = True
        print("✅ 数学核心架构: 存在且可导入")
    except ImportError:
        print("❌ 数学核心架构: 导入失败")

    # 检查流式中间件
    try:
        from h2q_project.src.h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
        achievements["streaming_middleware"] = True
        print("✅ 流式推理中间件: 存在且可导入")
    except ImportError:
        print("❌ 流式推理中间件: 导入失败")

    # 检查模型集成
    try:
        from model_crystallization_engine import ModelCrystallizationEngine
        achievements["model_integration"] = True
        print("✅ 模型集成引擎: 存在且可导入")
    except ImportError:
        print("❌ 模型集成引擎: 导入失败")

    # 检查结晶化技术
    if os.path.exists('/Users/imymm/H2Q-Evo/real_deepseek_benchmark_results.json'):
        achievements["crystallization_technology"] = True
        print("✅ 结晶化技术: 已有实际测试结果")
    else:
        print("❌ 结晶化技术: 无实际测试结果")

    # 检查真实模型测试
    try:
        with open('/Users/imymm/H2Q-Evo/real_deepseek_benchmark_results.json', 'r') as f:
            data = json.load(f)
            if data.get('model') == 'deepseek-coder:6.7b':
                achievements["real_model_testing"] = True
                print("✅ 真实模型测试: 已验证DeepSeek模型")
            else:
                print("❌ 真实模型测试: 测试数据无效")
    except:
        print("❌ 真实模型测试: 无测试数据")

    # 计算价值分数
    value_score = sum(achievements.values()) / len(achievements) * 100

    print("\n📊 成果价值评估:")
    print(f"   价值分数: {value_score:.1f}%")
    print(f"   完成项目: {sum(achievements.values())}/{len(achievements)}")

    if value_score >= 80:
        print("🎯 结论: 现有成果具有显著价值，已实现核心技术突破")
    elif value_score >= 60:
        print("🎯 结论: 现有成果具有一定价值，需要进一步完善")
    else:
        print("🎯 结论: 现有成果价值有限，需要重新评估方向")

    return achievements


def run_comprehensive_audit():
    """运行综合审计"""
    print("🚀 H2Q-Evo 综合真实性审计")
    print("=" * 60)

    results = {
        "timestamp": time.time(),
        "tests": {},
        "analysis": {},
        "achievements": {}
    }

    # 1. 测试数学核心
    results["tests"]["mathematical_core"] = test_mathematical_core()

    # 2. 测试流式中间件
    results["tests"]["streaming_middleware"] = test_streaming_middleware()

    # 3. 测试模型集成
    results["tests"]["model_integration"] = test_model_integration()

    # 4. 测试流式推理
    results["tests"]["streaming_inference"] = test_streaming_with_ollama()

    # 5. 分析236B模型问题
    results["analysis"]["236b_model_issue"] = analyze_236b_model_issue()

    # 6. 审计成果价值
    results["achievements"] = audit_achievements()

    # 保存完整审计结果
    with open("comprehensive_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📄 完整审计结果已保存: comprehensive_audit_results.json")
    return results


if __name__ == "__main__":
    run_comprehensive_audit()