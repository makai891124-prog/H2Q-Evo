#!/usr/bin/env python3
"""
执行所有可执行脚本并检查错误
"""
import subprocess
import sys
from pathlib import Path

# 关键脚本列表
SCRIPTS_TO_TEST = [
    "comprehensive_validation_final.py",
    "comprehensive_validation_v2.py",
    "verify_geometric_automation.py",
    "api_inspection.py",
    "h2q_project/run_experiment_fixed.py",
]

print("=" * 80)
print("H2Q-Evo 全面可执行代码检查")
print("=" * 80)

results = {
    "passed": [],
    "failed": [],
    "timeout": [],
    "missing": [],
}

for script in SCRIPTS_TO_TEST:
    script_path = Path(f"/Users/imymm/H2Q-Evo/{script}")
    
    print(f"\n测试: {script}")
    print("-" * 60)
    
    if not script_path.exists():
        print(f"❌ 文件不存在")
        results["missing"].append(script)
        continue
    
    try:
        # 运行脚本，限时20秒
        proc = subprocess.run(
            ["python3", str(script_path)],
            cwd="/Users/imymm/H2Q-Evo",
            capture_output=True,
            text=True,
            timeout=20
        )
        
        if proc.returncode == 0:
            print(f"✅ 成功执行 (返回码: 0)")
            results["passed"].append(script)
        else:
            # 提取错误信息
            error_msg = proc.stderr.split('\n')[-2] if proc.stderr else "Unknown error"
            print(f"❌ 执行失败 (返回码: {proc.returncode})")
            print(f"   错误: {error_msg[:80]}")
            results["failed"].append(script)
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  超时 (>20秒)")
        results["timeout"].append(script)
    except Exception as e:
        print(f"❌ 异常: {str(e)[:80]}")
        results["failed"].append(script)

# 生成报告
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

total = len(SCRIPTS_TO_TEST)
passed = len(results["passed"])
failed = len(results["failed"])
timeout = len(results["timeout"])
missing = len(results["missing"])

print(f"\n📊 执行统计:")
print(f"  ✅ 通过:     {passed}/{total}")
print(f"  ❌ 失败:     {failed}/{total}")
print(f"  ⏱️  超时:     {timeout}/{total}")
print(f"  ❓ 缺失:     {missing}/{total}")

if results["passed"]:
    print(f"\n✅ 通过的脚本:")
    for script in results["passed"]:
        print(f"   • {script}")

if results["failed"]:
    print(f"\n❌ 失败的脚本:")
    for script in results["failed"]:
        print(f"   • {script}")

if results["timeout"]:
    print(f"\n⏱️  超时的脚本:")
    for script in results["timeout"]:
        print(f"   • {script}")

print(f"\n🎯 结论:")
if passed == total:
    print(f"   ✅ 所有脚本都通过了")
    sys.exit(0)
elif passed >= total * 0.7:
    print(f"   ⚠️  {passed}/{total}个脚本通过，质量可接受")
    sys.exit(0)
else:
    print(f"   ❌ 通过率过低，需要修复")
    sys.exit(1)
