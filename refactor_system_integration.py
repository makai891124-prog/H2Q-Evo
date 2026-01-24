#!/usr/bin/env python3
"""
H2Q-Evo 系统集成重构验证器

任务:
1. 检查所有应用层文件
2. 验证它们与核心数学架构的集成
3. 运行测试验证重构效果
4. 生成集成报告
"""

import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "h2q_project"))

def test_unified_architecture_import():
    """测试统一架构导入"""
    try:
        from h2q.core.unified_architecture import (
            UnifiedH2QMathematicalArchitecture,
            UnifiedMathematicalArchitectureConfig,
            get_unified_h2q_architecture
        )
        return True, "统一架构导入成功"
    except Exception as e:
        return False, f"导入失败: {e}"

def test_evolution_integration():
    """测试进化集成模块"""
    try:
        from h2q.core.evolution_integration import (
            MathematicalArchitectureEvolutionBridge,
            H2QEvolutionSystemIntegration,
            create_mathematical_core_for_evolution_system
        )
        
        # 创建桥接器
        bridge = MathematicalArchitectureEvolutionBridge(dim=64, action_dim=16, device='cpu')
        
        # 测试前向传播
        x = torch.randn(4, 64)
        results = bridge(x)
        
        checks = {
            'bridge_created': bridge is not None,
            'forward_works': 'evolution_metrics' in results,
            'generation_tracked': bridge.generation_count > 0,
            'unified_arch_exists': bridge.unified_arch is not None,
        }
        
        return all(checks.values()), f"进化集成测试: {checks}"
        
    except Exception as e:
        return False, f"测试失败: {e}"

def test_all_core_modules():
    """测试所有核心模块"""
    modules_to_test = [
        ('h2q.core.lie_automorphism_engine', 'AutomaticAutomorphismOrchestrator'),
        ('h2q.core.noncommutative_geometry_operators', 'ComprehensiveReflectionOperatorModule'),
        ('h2q.core.automorphic_dde', 'LieGroupAutomorphicDecisionEngine'),
        ('h2q.core.knot_invariant_hub', 'KnotInvariantCentralHub'),
    ]
    
    results = {}
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results[module_name] = {'status': 'OK', 'class': class_name}
        except Exception as e:
            results[module_name] = {'status': 'FAIL', 'error': str(e)}
    
    all_ok = all(r['status'] == 'OK' for r in results.values())
    return all_ok, results

def test_complete_pipeline():
    """测试完整流程"""
    try:
        from h2q.core.unified_architecture import get_unified_h2q_architecture
        from h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge
        
        # 创建架构
        unified = get_unified_h2q_architecture(dim=128, action_dim=32, device='cpu')
        
        # 创建桥接器
        bridge = MathematicalArchitectureEvolutionBridge(dim=128, action_dim=32, device='cpu')
        
        # 测试数据
        batch_size = 8
        x = torch.randn(batch_size, 128)
        learning_signal = torch.tensor([0.5])
        
        # 运行多代进化
        generations = 5
        history = []
        
        for gen in range(generations):
            results = bridge(x, learning_signal)
            history.append({
                'generation': results['generation'],
                'metrics': results.get('evolution_metrics', {})
            })
            
            # 更新输入为输出（迭代进化）
            output, info = unified(x)
            x = output
        
        # 验证进化历史
        checks = {
            'generations_completed': len(history) == generations,
            'generation_increments': all(
                history[i]['generation'] == i+1 for i in range(generations)
            ),
            'metrics_tracked': all('metrics' in h for h in history),
            'bridge_state_updated': bridge.generation_count == generations,
        }
        
        return all(checks.values()), {
            'checks': checks,
            'final_generation': bridge.generation_count,
            'history_length': len(history)
        }
        
    except Exception as e:
        return False, f"流程测试失败: {e}"

def analyze_application_files():
    """分析应用层文件"""
    app_files = [
        'h2q_project/h2q_server.py',
        'evolution_system.py',
        'h2q_project/run_experiment.py',
    ]
    
    analysis = {}
    project_root = Path(__file__).parent
    
    for file_path in app_files:
        full_path = project_root / file_path
        if full_path.exists():
            content = full_path.read_text()
            
            # 检查是否使用了核心数学模块
            uses_unified_arch = 'UnifiedH2QMathematicalArchitecture' in content
            uses_evolution_bridge = 'MathematicalArchitectureEvolutionBridge' in content
            uses_core_modules = any(module in content for module in [
                'lie_automorphism_engine',
                'noncommutative_geometry_operators',
                'automorphic_dde',
                'knot_invariant_hub',
            ])
            
            analysis[file_path] = {
                'exists': True,
                'size': len(content),
                'uses_unified_arch': uses_unified_arch,
                'uses_evolution_bridge': uses_evolution_bridge,
                'uses_core_modules': uses_core_modules,
                'refactor_needed': not (uses_unified_arch or uses_evolution_bridge),
            }
        else:
            analysis[file_path] = {
                'exists': False,
                'refactor_needed': False,
            }
    
    return analysis

def generate_refactoring_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """生成重构建议"""
    recommendations = []
    
    for file_path, info in analysis.items():
        if not info.get('exists'):
            continue
            
        if info.get('refactor_needed'):
            recommendations.append(
                f"📝 {file_path} 需要重构以使用 UnifiedH2QMathematicalArchitecture"
            )
        elif info.get('uses_unified_arch'):
            recommendations.append(
                f"✅ {file_path} 已集成统一架构"
            )
    
    return recommendations

def run_full_system_audit():
    """运行完整系统审计"""
    print("=" * 60)
    print("H2Q-Evo 系统集成审计")
    print("=" * 60)
    print()
    
    audit_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'application_analysis': {},
        'recommendations': [],
        'overall_status': 'UNKNOWN',
    }
    
    # 测试1: 统一架构导入
    print("📦 测试1: 统一架构导入...")
    success, msg = test_unified_architecture_import()
    audit_results['tests']['unified_architecture_import'] = {
        'pass': success,
        'message': msg
    }
    print(f"   {'✅ PASS' if success else '❌ FAIL'}: {msg}")
    print()
    
    # 测试2: 进化集成
    print("🔗 测试2: 进化集成模块...")
    success, msg = test_evolution_integration()
    audit_results['tests']['evolution_integration'] = {
        'pass': success,
        'message': str(msg)
    }
    print(f"   {'✅ PASS' if success else '❌ FAIL'}")
    if isinstance(msg, dict):
        for k, v in msg.items():
            print(f"      {k}: {v}")
    print()
    
    # 测试3: 所有核心模块
    print("🧮 测试3: 核心数学模块...")
    success, results = test_all_core_modules()
    audit_results['tests']['core_modules'] = {
        'pass': success,
        'modules': results
    }
    for module, info in results.items():
        status = '✅' if info['status'] == 'OK' else '❌'
        print(f"   {status} {module}")
    print()
    
    # 测试4: 完整流程
    print("🚀 测试4: 完整进化流程...")
    success, info = test_complete_pipeline()
    audit_results['tests']['complete_pipeline'] = {
        'pass': success,
        'info': info
    }
    print(f"   {'✅ PASS' if success else '❌ FAIL'}")
    if isinstance(info, dict) and 'checks' in info:
        for check_name, check_pass in info['checks'].items():
            print(f"      {check_name}: {'✅' if check_pass else '❌'}")
    print()
    
    # 分析应用文件
    print("📋 应用层文件分析...")
    analysis = analyze_application_files()
    audit_results['application_analysis'] = analysis
    
    for file_path, info in analysis.items():
        if info.get('exists'):
            status = '✅' if not info.get('refactor_needed') else '🔄'
            print(f"   {status} {file_path}")
            print(f"      - 大小: {info['size']} bytes")
            print(f"      - 使用统一架构: {info['uses_unified_arch']}")
            print(f"      - 需要重构: {info['refactor_needed']}")
        else:
            print(f"   ⚠️  {file_path} (不存在)")
    print()
    
    # 生成建议
    print("💡 重构建议...")
    recommendations = generate_refactoring_recommendations(analysis)
    audit_results['recommendations'] = recommendations
    
    for rec in recommendations:
        print(f"   {rec}")
    print()
    
    # 总体评估
    all_tests_pass = all(t['pass'] for t in audit_results['tests'].values())
    files_need_refactor = sum(
        1 for info in analysis.values() 
        if info.get('refactor_needed', False)
    )
    
    if all_tests_pass and files_need_refactor == 0:
        audit_results['overall_status'] = 'EXCELLENT'
        status_msg = "🏆 优秀 - 所有测试通过，所有文件已集成"
    elif all_tests_pass and files_need_refactor <= 2:
        audit_results['overall_status'] = 'GOOD'
        status_msg = f"✅ 良好 - 所有测试通过，{files_need_refactor}个文件需要重构"
    elif all_tests_pass:
        audit_results['overall_status'] = 'NEEDS_WORK'
        status_msg = f"🔄 需要工作 - 测试通过但{files_need_refactor}个文件需要重构"
    else:
        audit_results['overall_status'] = 'FAILING'
        failed_tests = [
            name for name, result in audit_results['tests'].items()
            if not result['pass']
        ]
        status_msg = f"❌ 失败 - {len(failed_tests)}个测试未通过"
    
    print("=" * 60)
    print("审计结果")
    print("=" * 60)
    print(f"状态: {status_msg}")
    print(f"测试通过率: {sum(1 for t in audit_results['tests'].values() if t['pass'])}/{len(audit_results['tests'])}")
    print(f"需要重构的文件: {files_need_refactor}")
    print("=" * 60)
    print()
    
    # 保存报告
    report_path = Path(__file__).parent / 'system_integration_audit_report.json'
    with open(report_path, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    
    print(f"📄 完整报告已保存到: {report_path}")
    print()
    
    return audit_results

if __name__ == "__main__":
    results = run_full_system_audit()
    
    # 退出代码
    sys.exit(0 if results['overall_status'] in ['EXCELLENT', 'GOOD'] else 1)
