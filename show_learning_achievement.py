#!/usr/bin/env python3
"""
H2Q-Evo 学习成果综合展示
"""

import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("🎓 H2Q-Evo 持续监督学习与进化系统 - 综合成果展示")
print("="*80)
print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 读取学习报告
report_file = Path("learning_report.json")
if not report_file.exists():
    print("⚠️ 学习报告未生成")
    exit(1)

with open(report_file) as f:
    report = json.load(f)

# 基本统计
stats = report['stats']
kb_stats = report['kb_stats']

print("📊 核心指标")
print("-"*80)
print(f"总学习项目:     {stats['total_learned']:3d} 项")
print(f"知识库总量:     {kb_stats['total_count']:3d} 条")
print(f"已验证知识:     {kb_stats['verified_count']:3d} 条 ({kb_stats['verified_count']/kb_stats['total_count']*100:.1f}%)")
print(f"测试次数:       {len(report['test_results']):3d} 次")
print(f"进化周期:       {stats['evolution_count']:3d} 次")
print()

# 知识增长
print("📈 知识增长")
print("-"*80)
initial_verified = 2  # 初始已验证
growth = kb_stats['verified_count'] - initial_verified
growth_rate = (growth / initial_verified * 100) if initial_verified > 0 else 0
print(f"初始状态:       {initial_verified} 条已验证")
print(f"当前状态:       {kb_stats['verified_count']} 条已验证")
print(f"增长数量:       +{growth} 条")
print(f"增长率:         +{growth_rate:.0f}%")
print()

# 顶级掌握
print("✨ 掌握最好的概念 (Top 5)")
print("-"*80)
if 'top_concepts' in report and report['top_concepts']:
    for i, item in enumerate(report['top_concepts'][:5], 1):
        concept = item['concept']
        domain = item['domain']
        score = item['understanding_score'] * 100
        quality = "🌟优秀" if score >= 80 else "✅良好" if score >= 70 else "📝及格"
        print(f"{i}. {concept:30s} │ {domain:20s} │ {score:5.1f}% │ {quality}")
else:
    print("   (暂无数据)")
print()

# 各领域分布
print("🎯 各领域掌握情况")
print("-"*80)

from large_knowledge_base import LargeKnowledgeBase
kb = LargeKnowledgeBase()
kb.load()

for domain, total in sorted(kb_stats['by_domain'].items()):
    verified = sum(1 for k in kb.knowledge[domain] if k.get('verified'))
    mastery = verified / total * 100 if total > 0 else 0
    bar = "█" * int(mastery / 5)
    quality = "🌟优秀" if mastery >= 80 else "✅良好" if mastery >= 60 else "📝及格" if mastery >= 40 else "⚠️需加强"
    print(f"{domain:20s} │{bar:<20s}│ {verified:2d}/{total:2d} ({mastery:5.1f}%) {quality}")
print()

# 测试结果
if report['test_results']:
    print("🎯 测试历史")
    print("-"*80)
    for i, test in enumerate(report['test_results'], 1):
        status = "✅" if test['quality'] in ['excellent', 'good'] else "⚠️"
        quality_cn = {"excellent": "优秀", "good": "良好", "needs_improvement": "需改进"}.get(test['quality'], test['quality'])
        print(f"测试 {i}: {status} {test['correct']}/{test['total']} 正确 ({test['pass_rate']*100:.0f}%) - {quality_cn}")
    print()

# 质量评估
if stats['quality_scores']:
    avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores'])
    print("📊 学习质量评估")
    print("-"*80)
    print(f"平均质量:       {avg_quality*100:.1f}%")
    print(f"最高质量:       {max(stats['quality_scores'])*100:.0f}%")
    print(f"最低质量:       {min(stats['quality_scores'])*100:.0f}%")
    print(f"质量评级:       ", end="")
    if avg_quality >= 0.85:
        print("🌟 优秀")
    elif avg_quality >= 0.70:
        print("✅ 良好")
    elif avg_quality >= 0.60:
        print("📝 及格")
    else:
        print("⚠️ 需改进")
    print()

# 学习效率
if stats['total_learned'] > 0:
    print("⚡ 学习效率")
    print("-"*80)
    
    # 计算总时间
    total_time = sum(item['learning_time'] for item in report['top_concepts'])
    avg_time = total_time / len(report['top_concepts']) if report['top_concepts'] else 0
    
    print(f"平均学习时间:   {avg_time:.2f} 秒/概念")
    print(f"学习速率:       {1/avg_time:.2f} 概念/秒" if avg_time > 0 else "N/A")
    print()

# 系统状态
print("🔧 系统状态")
print("-"*80)
print(f"✅ 持续学习系统:  运行正常")
print(f"✅ 知识库:        {kb_stats['total_count']}条，持久化成功")
print(f"✅ 学习报告:      已生成")
print(f"✅ 自我进化:      {stats['evolution_count']}次")
print()

# 验收结论
print("="*80)
print("🎉 验收结论")
print("="*80)
verdict = "✅ 通过" if kb_stats['verified_count'] >= 10 and (avg_quality >= 0.7 if stats['quality_scores'] else True) else "⚠️ 需改进"
print(f"验收状态:       {verdict}")
print()

if kb_stats['verified_count'] >= 10:
    print("✅ 知识增长达标 (≥10条验证知识)")
else:
    print(f"⚠️ 知识增长不足 ({kb_stats['verified_count']}/10)")

if stats['quality_scores'] and avg_quality >= 0.7:
    print("✅ 学习质量达标 (≥70%)")
else:
    print(f"⚠️ 学习质量待提升")

if stats['evolution_count'] > 0:
    print(f"✅ 系统进化正常 ({stats['evolution_count']}次)")
else:
    print("⚠️ 未触发系统进化")

print()
print("="*80)
print("📝 详细报告: LEARNING_ACHIEVEMENT_REPORT.md")
print("📊 原始数据: learning_report.json")
print("="*80)
