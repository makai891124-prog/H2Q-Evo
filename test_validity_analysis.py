#!/usr/bin/env python3
"""
H2Q-Evo 测试有效性分析和代码审计
分析概念理解和数学推理测试的有效性问题
"""

import json
import os
import sys
from typing import Dict, List, Any


def analyze_test_validity():
    """分析测试有效性"""
    print("🔍 H2Q-Evo 测试有效性分析")
    print("=" * 50)

    issues_found = []

    # 分析概念理解测试
    print("\n🧠 分析概念理解测试有效性:")
    concept_test_issues = analyze_concept_understanding_test()
    issues_found.extend(concept_test_issues)

    # 分析数学推理测试
    print("\n🔢 分析数学推理测试有效性:")
    math_test_issues = analyze_mathematical_reasoning_test()
    issues_found.extend(math_test_issues)

    # 分析代码生成测试
    print("\n💻 分析代码生成测试有效性:")
    code_test_issues = analyze_code_generation_test()
    issues_found.extend(code_test_issues)

    # 分析文本生成测试
    print("\n📝 分析文本生成测试有效性:")
    text_test_issues = analyze_text_generation_test()
    issues_found.extend(text_test_issues)

    # 生成改进建议
    print("\n💡 测试改进建议:")
    improvement_suggestions = generate_improvement_suggestions(issues_found)

    for suggestion in improvement_suggestions:
        print(f"  • {suggestion}")

    # 保存分析报告
    report = {
        'analysis_timestamp': '2026-01-27',
        'issues_found': issues_found,
        'improvement_suggestions': improvement_suggestions,
        'overall_assessment': '需要重大改进 - 当前测试主要检查统计指标而非实际能力'
    }

    report_file = "/Users/imymm/H2Q-Evo/test_validity_analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 分析报告已保存: {report_file}")

    return issues_found


def analyze_concept_understanding_test() -> List[str]:
    """分析概念理解测试"""
    issues = []

    print("  ❌ 问题识别:")

    # 问题1: 只检查输出一致性，不验证实际理解
    issues.append({
        'test': 'concept_understanding',
        'severity': 'critical',
        'issue': '只检查输出logits的方差，不验证概念理解的准确性',
        'impact': '无法区分真正理解概念的模型和随机输出的模型'
    })
    print("    - 只检查输出一致性，不验证概念含义理解")

    # 问题2: 测试概念过于简单
    issues.append({
        'test': 'concept_understanding',
        'severity': 'high',
        'issue': '测试概念过于基础，没有验证深度理解',
        'impact': '无法评估模型对复杂概念的理解能力'
    })
    print("    - 测试概念过于简单，缺乏深度验证")

    # 问题3: 没有验证概念关系
    issues.append({
        'test': 'concept_understanding',
        'severity': 'medium',
        'issue': '不测试概念之间的关系和推理',
        'impact': '无法评估概念关联和推理能力'
    })
    print("    - 不验证概念间的关系和推理")

    return issues


def analyze_mathematical_reasoning_test() -> List[str]:
    """分析数学推理测试"""
    issues = []

    print("  ❌ 问题识别:")

    # 问题1: 不验证计算结果正确性
    issues.append({
        'test': 'mathematical_reasoning',
        'severity': 'critical',
        'issue': '只检查输出复杂度，不验证数学计算的正确性',
        'impact': '无法区分正确计算和错误但复杂的输出'
    })
    print("    - 不验证计算结果的正确性")

    # 问题2: 测试问题过于简单
    issues.append({
        'test': 'mathematical_reasoning',
        'severity': 'high',
        'issue': '只测试基础算术，不包含代数、几何等高级数学',
        'impact': '无法评估高级数学推理能力'
    })
    print("    - 测试问题过于基础，缺乏高级数学")

    # 问题3: 没有步骤推理验证
    issues.append({
        'test': 'mathematical_reasoning',
        'severity': 'medium',
        'issue': '不验证解题步骤和推理过程',
        'impact': '无法评估数学思维的逻辑性'
    })
    print("    - 不验证推理步骤和思维过程")

    return issues


def analyze_code_generation_test() -> List[str]:
    """分析代码生成测试"""
    issues = []

    print("  ❌ 问题识别:")

    # 问题1: 只检查语法结构，不验证功能正确性
    issues.append({
        'test': 'code_generation',
        'severity': 'high',
        'issue': '只检查代码结构关键词，不验证代码功能和正确性',
        'impact': '无法区分语法正确但功能错误的代码'
    })
    print("    - 只检查语法结构，不验证功能正确性")

    # 问题2: 测试用例过于简单
    issues.append({
        'test': 'code_generation',
        'severity': 'medium',
        'issue': '测试prompt过于基础，缺乏复杂编程任务',
        'impact': '无法评估复杂代码生成能力'
    })
    print("    - 测试用例过于简单")

    return issues


def analyze_text_generation_test() -> List[str]:
    """分析文本生成测试"""
    issues = []

    print("  ❌ 问题识别:")

    # 问题1: 评估标准过于宽泛
    issues.append({
        'test': 'text_generation',
        'severity': 'medium',
        'issue': '评估标准基于长度和常见词汇，缺乏质量评估',
        'impact': '无法准确评估文本生成的质量和连贯性'
    })
    print("    - 评估标准过于宽泛，缺乏质量验证")

    # 问题2: 不考虑上下文相关性
    issues.append({
        'test': 'text_generation',
        'severity': 'low',
        'issue': '不评估生成文本与输入prompt的相关性',
        'impact': '可能接受不相关的输出'
    })
    print("    - 不评估生成内容的相关性")

    return issues


def generate_improvement_suggestions(issues: List[Dict]) -> List[str]:
    """生成改进建议"""
    suggestions = []

    # 按严重程度排序
    critical_issues = [i for i in issues if i['severity'] == 'critical']
    high_issues = [i for i in issues if i['severity'] == 'high']
    medium_issues = [i for i in issues if i['severity'] == 'medium']

    # 关键改进建议
    suggestions.extend([
        "实现真正的能力验证而非统计指标检查",
        "添加正确性验证和质量评估机制",
        "扩展测试用例覆盖更多复杂场景",
        "实现推理步骤和思维过程验证",
        "添加跨概念关系和逻辑推理测试"
    ])

    # 具体测试改进
    if critical_issues:
        suggestions.append("优先修复关键问题：实际验证计算结果、概念理解准确性")

    if high_issues:
        suggestions.append("改进测试深度：添加高级数学、复杂编程任务")

    if medium_issues:
        suggestions.append("增强评估标准：实现功能验证、相关性检查")

    return suggestions


def audit_code_quality():
    """代码质量审计"""
    print("\n🔧 代码质量审计:")
    print("-" * 30)

    audit_issues = []

    # 检查文件
    files_to_audit = [
        "pure_core_machine_validation.py",
        "deepseek_enhanced_agi_evolution.py",
        "hierarchical_concept_encoder.py"
    ]

    for file_path in files_to_audit:
        if os.path.exists(file_path):
            issues = audit_single_file(file_path)
            audit_issues.extend(issues)

    if audit_issues:
        print("❌ 发现代码问题:")
        for issue in audit_issues:
            print(f"  - {issue['file']}: {issue['issue']}")
    else:
        print("✅ 代码审计通过")

    return audit_issues


def audit_single_file(file_path: str) -> List[Dict]:
    """审计单个文件"""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # 检查硬编码值
        for i, line in enumerate(lines):
            if 'return 0.' in line and any(char.isdigit() for char in line):
                if not any(word in line for word in ['min(', 'max(', 'abs(']):  # 排除函数调用
                    issues.append({
                        'file': file_path,
                        'line': i+1,
                        'issue': f'可能的硬编码返回值: {line.strip()}'
                    })

        # 检查异常处理
        exception_count = content.count('except Exception')
        if exception_count > 10:
            issues.append({
                'file': file_path,
                'issue': f'过度使用通用异常处理 ({exception_count} 次)'
            })

        # 检查代码复杂度
        if len(lines) > 1000:
            issues.append({
                'file': file_path,
                'issue': f'文件过大 ({len(lines)} 行)，建议拆分'
            })

    except Exception as e:
        issues.append({
            'file': file_path,
            'issue': f'文件读取失败: {e}'
        })

    return issues


def main():
    """主函数"""
    # 分析测试有效性
    test_issues = analyze_test_validity()

    # 代码质量审计
    code_issues = audit_code_quality()

    # 总结
    print("\n📊 审计总结:")
    print(f"  测试问题: {len(test_issues)} 个")
    print(f"  代码问题: {len(code_issues)} 个")
    print(f"  总体状态: {'需要改进' if test_issues or code_issues else '良好'}")

    if test_issues or code_issues:
        print("\n⚠️  建议在启动AGI进化前修复关键问题")
    else:
        print("\n✅ 可以启动AGI进化训练")


if __name__ == "__main__":
    main()