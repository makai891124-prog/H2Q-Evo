#!/usr/bin/env python3
"""
实时监控监督学习进度
"""

import json
import time
import os
from pathlib import Path

def monitor_learning():
    """监控学习进度"""
    print("🔍 监督学习实时监控")
    print("="*80)
    
    log_file = Path("supervised_learning.log")
    report_file = Path("learning_report.json")
    
    if not log_file.exists():
        print("⚠️ 学习系统尚未启动")
        print("启动命令: python3 supervised_learning_evolution.py 30 8")
        return
    
    # 读取日志
    with open(log_file) as f:
        lines = f.readlines()
    
    # 统计学习情况
    learned_count = 0
    passed_count = 0
    failed_count = 0
    test_count = 0
    evolution_count = 0
    
    for line in lines:
        if "学习通过" in line:
            passed_count += 1
            learned_count += 1
        elif "需要重新学习" in line:
            failed_count += 1
        elif "测试通过" in line:
            test_count += 1
        elif "进化周期" in line:
            evolution_count += 1
    
    print(f"\n📊 当前进度:")
    print(f"   学习项目: {learned_count + failed_count}")
    print(f"   ✅ 通过: {passed_count}")
    print(f"   ⚠️  失败: {failed_count}")
    print(f"   通过率: {passed_count/(learned_count+failed_count)*100 if (learned_count+failed_count)>0 else 0:.1f}%")
    print(f"   测试次数: {test_count}")
    print(f"   进化周期: {evolution_count}")
    
    # 显示最近10行日志
    print(f"\n📝 最近日志 (最后10行):")
    print("-"*80)
    for line in lines[-10:]:
        print(line.rstrip())
    
    # 如果有报告，显示报告
    if report_file.exists():
        print(f"\n{'='*80}")
        print("📊 最终学习报告已生成")
        print(f"{'='*80}")
        
        with open(report_file) as f:
            report = json.load(f)
        
        stats = report['stats']
        kb_stats = report['kb_stats']
        
        print(f"\n总学习: {stats['total_learned']}")
        print(f"测试通过: {stats['tests_passed']}/{stats['tests_passed']+stats['tests_failed']}")
        print(f"进化周期: {stats['evolution_count']}")
        print(f"知识库: {kb_stats['verified_count']}/{kb_stats['total_count']} 已验证 ({kb_stats['verified_count']/kb_stats['total_count']*100:.1f}%)")
        
        if 'top_concepts' in report and report['top_concepts']:
            print(f"\n✨ 掌握最好的概念:")
            for i, item in enumerate(report['top_concepts'][:5], 1):
                print(f"   {i}. {item['concept']} - {item['understanding_score']*100:.0f}%")

if __name__ == "__main__":
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            monitor_learning()
            
            # 检查是否完成
            if Path("learning_report.json").exists():
                print(f"\n✅ 学习已完成！")
                break
            
            print(f"\n⏳ 等待5秒后刷新... (Ctrl+C 退出)")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")
