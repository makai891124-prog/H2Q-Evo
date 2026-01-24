#!/usr/bin/env python3
"""
测试h2q_server重构的核心数学组件（不需要FastAPI）
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "h2q_project"))

def test_unified_architecture_integration():
    """测试统一架构集成"""
    try:
        from h2q.core.unified_architecture import (
            UnifiedH2QMathematicalArchitecture,
            get_unified_h2q_architecture
        )
        from h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge
        
        print("📐 测试统一架构...")
        
        # 创建架构
        unified = get_unified_h2q_architecture(dim=128, action_dim=32, device='cpu')
        print(f"   ✅ 创建成功: {type(unified).__name__}")
        
        # 测试前向传播
        x = torch.randn(4, 128)
        output, info = unified(x)
        
        print(f"   ✅ 前向传播成功")
        print(f"      输入: {x.shape}")
        print(f"      输出: {output.shape}")
        print(f"      启用模块: {info.get('enabled_modules', [])}")
        print(f"      全局完整性: {info.get('global_integrity', 0.0):.4f}")
        
        return True, info
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_evolution_bridge():
    """测试进化桥接器"""
    try:
        from h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge
        
        print("\n🌉 测试进化桥接器...")
        
        # 创建桥接器
        bridge = MathematicalArchitectureEvolutionBridge(dim=128, action_dim=32, device='cpu')
        print(f"   ✅ 创建成功")
        
        # 运行进化步骤
        x = torch.randn(4, 128)
        learning_signal = torch.tensor([0.5])
        
        generations = 3
        history = []
        
        for gen in range(generations):
            results = bridge(x, learning_signal)
            history.append(results)
            print(f"   世代 {gen+1}: norm={results['evolution_metrics']['output_norm']:.4f}")
        
        print(f"   ✅ 完成{generations}代进化")
        print(f"      最终世代数: {bridge.generation_count}")
        print(f"      历史记录: {len(bridge.evolution_history)} 条")
        
        return True, history
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def simulate_chat_processing():
    """模拟聊天处理流程"""
    try:
        from h2q.core.unified_architecture import get_unified_h2q_architecture
        
        print("\n💬 模拟聊天处理...")
        
        # 创建架构
        unified = get_unified_h2q_architecture(dim=256, action_dim=64, device='cpu')
        
        # 模拟用户输入
        prompt = "Hello, H2Q!"
        
        # 文本到张量（简化版）
        tokens = [ord(c) for c in prompt[:256]]
        tokens += [0] * (256 - len(tokens))
        input_tensor = torch.tensor(tokens, dtype=torch.float32).view(1, -1)
        
        print(f"   输入: '{prompt}'")
        print(f"   张量: {input_tensor.shape}")
        
        # 通过数学架构处理
        with torch.no_grad():
            output, info = unified(input_tensor)
        
        # 提取数学性质
        fueter_curvature = info.get('holomorphic_consistency', {}).get('fueter_gradient_norm', 0.0)
        spectral_shift = info.get('lie_group_properties', {}).get('lie_exponential_norm', 0.0)
        integrity = info.get('global_integrity', 1.0)
        
        print(f"   ✅ 处理成功")
        print(f"      Fueter曲率: {fueter_curvature:.6f}")
        print(f"      谱移: {spectral_shift:.6f}")
        print(f"      完整性: {integrity:.6f}")
        print(f"      状态: {'Analytic' if fueter_curvature <= 0.05 else 'Pruned/Healed'}")
        
        return True, info
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def simulate_generate_processing():
    """模拟生成处理流程"""
    try:
        from h2q.core.unified_architecture import get_unified_h2q_architecture
        
        print("\n🔮 模拟文本生成...")
        
        unified = get_unified_h2q_architecture(dim=256, action_dim=64, device='cpu')
        
        prompt = "Generate:"
        max_new_tokens = 64
        
        # 初始化
        tokens = [ord(c) for c in prompt[:256]]
        tokens += [0] * (256 - len(tokens))
        input_tensor = torch.tensor(tokens, dtype=torch.float32).view(1, -1)
        
        print(f"   输入: '{prompt}'")
        print(f"   最大新token: {max_new_tokens}")
        
        # 生成
        with torch.no_grad():
            output, info = unified(input_tensor)
        
        # 提取前max_new_tokens个值
        generated = output[0, :max_new_tokens]
        
        print(f"   ✅ 生成成功")
        print(f"      输出形状: {output.shape}")
        print(f"      生成tokens: {generated.shape}")
        print(f"      完整性: {info.get('global_integrity', 1.0):.6f}")
        
        return True, output
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    print("=" * 60)
    print("H2Q Server 核心数学组件测试")
    print("=" * 60)
    print()
    
    results = {}
    
    # 测试1: 统一架构
    success, data = test_unified_architecture_integration()
    results['unified_architecture'] = success
    
    # 测试2: 进化桥接
    success, data = test_evolution_bridge()
    results['evolution_bridge'] = success
    
    # 测试3: 聊天处理
    success, data = simulate_chat_processing()
    results['chat_processing'] = success
    
    # 测试4: 文本生成
    success, data = simulate_generate_processing()
    results['generate_processing'] = success
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print(f"\n通过率: {total_pass}/{total_tests} ({total_pass/total_tests*100:.1f}%)")
    
    if total_pass == total_tests:
        print("\n🏆 所有测试通过！h2q_server重构的核心数学组件完全正常。")
        print("\n💡 下一步:")
        print("   1. 安装FastAPI: pip3 install fastapi uvicorn")
        print("   2. 备份原服务器: mv h2q_project/h2q_server.py h2q_project/h2q_server_backup.py")
        print("   3. 应用重构: mv h2q_project/h2q_server_refactored.py h2q_project/h2q_server.py")
        print("   4. 启动服务: cd h2q_project && python3 -m uvicorn h2q_server:app --reload")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
