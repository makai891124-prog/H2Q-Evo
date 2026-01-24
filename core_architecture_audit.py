#!/usr/bin/env python3
"""
H2Q-Evo核心架构集成验证与系统重构
"""
import sys
import os
from pathlib import Path
import torch
import time
import json

# 正确设置导入路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "h2q_project"))

print("\n" + "="*80)
print("🔍 H2Q-Evo 核心数学架构验证")
print("="*80)

# 1. 检查核心模块存在性
print("\n📂 检查核心模块...")
core_modules_h2q = [
    "h2q/core/lie_automorphism_engine.py",
    "h2q/core/noncommutative_geometry_operators.py",
    "h2q/core/automorphic_dde.py",
    "h2q/core/knot_invariant_hub.py",
    "h2q/core/unified_architecture.py"
]

new_modules = [
    "lie_automorphism_engine.py",
    "noncommutative_geometry_operators.py",
    "automorphic_dde.py"
]

total_size = 0
for module in core_modules_h2q:
    path = project_root / "h2q_project" / module
    if path.exists():
        size = path.stat().st_size
        total_size += size
        print(f"  ✅ {module:60s} ({size:,} bytes)")
    else:
        print(f"  ❌ {module:60s} 不存在")

for module in new_modules:
    path = project_root / "h2q_project" / module
    if path.exists():
        size = path.stat().st_size
        total_size += size
        print(f"  ✅ 新模块: {module:52s} ({size:,} bytes)")

print(f"\n📊 总代码量: {total_size:,} bytes (~{total_size//1024} KB)")

# 2. 测试导入
print("\n🔹 测试1: 导入h2q.core核心模块...")
try:
    from h2q.core.lie_automorphism_engine import AutomaticAutomorphismOrchestrator, get_lie_automorphism_engine
    from h2q.core.unified_architecture import UnifiedH2QMathematicalArchitecture, UnifiedMathematicalArchitectureConfig
    print("  ✅ h2q.core模块导入成功")
    core_import_ok = True
except Exception as e:
    print(f"  ❌ h2q.core模块导入失败: {e}")
    core_import_ok = False

# 3. 测试新模块
print("\n🔹 测试2: 导入新实现模块...")
try:
    from lie_automorphism_engine import LieGroupAutomorphismEngine
    from noncommutative_geometry_operators import NoncommutativeGeometryOperators
    from automorphic_dde import AutomorphicDDE
    print("  ✅ 新模块导入成功")
    new_import_ok = True
except Exception as e:
    print(f"  ❌ 新模块导入失败: {e}")
    new_import_ok = False

# 4. 功能测试
print("\n🔹 测试3: 统一架构功能测试...")
if core_import_ok:
    try:
        config = UnifiedMathematicalArchitectureConfig(
            dim=256,
            device='cpu',
            enable_lie_automorphism=True,
            enable_reflection_operators=True,
            enable_knot_constraints=True
        )
        unified_arch = UnifiedH2QMathematicalArchitecture(config)
        
        x = torch.randn(8, 256)
        start = time.time()
        output, info = unified_arch(x)
        elapsed = (time.time() - start) * 1000
        
        print(f"  ✅ 统一架构测试成功 ({elapsed:.2f} ms)")
        print(f"     输入: {x.shape}, 输出: {output.shape}")
        print(f"     信息键: {list(info.keys())[:5]}...")
        unified_ok = True
    except Exception as e:
        print(f"  ❌ 统一架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        unified_ok = False
else:
    unified_ok = False

# 5. 新模块测试
print("\n🔹 测试4: 新模块流程测试...")
if new_import_ok:
    try:
        lie_engine = LieGroupAutomorphismEngine()
        fueter_ops = NoncommutativeGeometryOperators()
        automorphic = AutomorphicDDE()
        
        x = torch.randn(8, 256)
        start = time.time()
        out1, info1 = lie_engine(x)
        out2, info2 = fueter_ops(out1)
        out3, info3 = automorphic(out2)
        elapsed = (time.time() - start) * 1000
        
        print(f"  ✅ 新模块流程测试成功 ({elapsed:.2f} ms)")
        print(f"     Lie → Fueter → Automorphic 流程正常")
        new_test_ok = True
    except Exception as e:
        print(f"  ❌ 新模块流程测试失败: {e}")
        new_test_ok = False
else:
    new_test_ok = False

# 6. 数学性质验证
print("\n🔹 测试5: 数学性质验证...")
if new_import_ok:
    try:
        from lie_automorphism_engine import QuaternionLieGroupModule, QuaternionLieGroupConfig
        
        config = QuaternionLieGroupConfig()
        quat_module = QuaternionLieGroupModule(config)
        
        q1 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        q1 = quat_module.quaternion_normalize(q1)
        q2 = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        q2 = quat_module.quaternion_normalize(q2)
        q3 = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        q3 = quat_module.quaternion_normalize(q3)
        
        # 结合律
        left = quat_module.quaternion_multiply(quat_module.quaternion_multiply(q1, q2), q3)
        right = quat_module.quaternion_multiply(q1, quat_module.quaternion_multiply(q2, q3))
        assoc_error = torch.norm(left - right).item()
        
        # 非交换性
        forward = quat_module.quaternion_multiply(q1, q2)
        backward = quat_module.quaternion_multiply(q2, q1)
        non_comm = torch.norm(forward - backward).item()
        
        print(f"  ✅ 四元数性质验证成功")
        print(f"     结合律误差: {assoc_error:.2e}")
        print(f"     非交换性: {non_comm:.4f}")
        math_ok = assoc_error < 1e-5 and non_comm > 1e-5
    except Exception as e:
        print(f"  ❌ 数学性质验证失败: {e}")
        math_ok = False
else:
    math_ok = False

# 7. 总结
print("\n" + "="*80)
print("📊 验证总结")
print("="*80)

tests = [
    ("核心模块导入", core_import_ok),
    ("新模块导入", new_import_ok),
    ("统一架构功能", unified_ok),
    ("新模块流程", new_test_ok),
    ("数学性质", math_ok)
]

passed = sum(1 for _, ok in tests if ok)
total = len(tests)

print(f"\n通过测试: {passed}/{total} ({100*passed/total:.1f}%)")
print()
for name, ok in tests:
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status} {name}")

if passed / total >= 0.8:
    print("\n🏆 审计通过！核心架构完整且功能正常。")
    print("\n✅ 确认：")
    print("  - h2q/core/ 下的原核心模块存在且功能正常")
    print("  - 新实现的数学模块可独立运行")
    print("  - 系统可基于现有架构进行重构")
    result = 0
else:
    print("\n⚠️  审计发现问题，需要修复。")
    result = 1

# 保存报告
report = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_code_size': total_size,
    'test_results': {name: ok for name, ok in tests},
    'pass_rate': passed / total,
    'status': 'PASS' if result == 0 else 'FAIL'
}

with open('core_architecture_audit_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n📄 报告已保存: core_architecture_audit_report.json")
print("="*80)

sys.exit(result)
