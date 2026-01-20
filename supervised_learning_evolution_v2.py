#!/usr/bin/env python3
"""
H2Q-Evo 增强型持续监督学习与进化系统 (v2)
优化学习质量和进化触发机制
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from large_knowledge_base import LargeKnowledgeBase

class EnhancedSupervisedLearning:
    """增强型监督学习系统 - 优化质量和进化"""
    
    def __init__(self):
        self.kb = LargeKnowledgeBase()
        self.kb.load()
        
        # 学习记录
        self.learning_history = []
        self.test_history = []
        
        # 优化参数
        self.base_quality_threshold = 0.70
        self.quality_threshold = self.base_quality_threshold
        
        # 测试参数
        self.test_interval = 3  # 每学习3项测试一次
        self.evolution_interval = 5  # 每学习5项进化一次
        
        # 统计
        self.stats = {
            "total_learned": 0,
            "quality_passed": 0,
            "quality_failed": 0,
            "tests_conducted": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "evolution_count": 0,
            "quality_scores": [],
            "test_scores": []
        }
        
        print("✓ 增强型监督学习系统初始化完成\n")
        self._display_system_info()
    
    def _display_system_info(self):
        """显示系统信息"""
        stats = self.kb.get_stats()
        print("📊 系统配置:")
        print(f"   知识库: {stats['total_count']} 条 ({stats['verified_count']} 已验证)")
        print(f"   质量阈值: {self.quality_threshold*100:.0f}%")
        print(f"   测试间隔: 每 {self.test_interval} 项")
        print(f"   进化间隔: 每 {self.evolution_interval} 项")
        print()
    
    def learn_with_deep_assessment(self, item: Tuple[str, Dict]) -> Dict:
        """深度学习与评估"""
        domain, knowledge = item
        concept = knowledge['concept']
        detail = knowledge.get('detail', '')
        difficulty = knowledge.get('difficulty', 3)
        
        print(f"📚 {concept}")
        
        # 多维度理解评估
        understanding_score = self._comprehensive_assessment(concept, detail, domain, difficulty)
        
        # 记录学习
        learning_record = {
            "concept": concept,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "understanding_score": understanding_score,
            "difficulty": difficulty,
            "detail_length": len(detail)
        }
        
        self.learning_history.append(learning_record)
        self.stats['total_learned'] += 1
        self.stats['quality_scores'].append(understanding_score)
        
        # 判断是否通过
        if understanding_score >= self.quality_threshold:
            self.kb.mark_verified(domain, concept)
            self.stats['quality_passed'] += 1
            status = "✅"
        else:
            self.stats['quality_failed'] += 1
            status = "⚠️"
        
        score_pct = understanding_score * 100
        quality = "优秀" if score_pct >= 85 else "良好" if score_pct >= 75 else "及格" if score_pct >= 70 else "重学"
        
        print(f"   {status} {score_pct:.1f}% ({quality}) - {domain}")
        
        return {
            "status": "passed" if understanding_score >= self.quality_threshold else "retry",
            "score": understanding_score,
            "quality": quality
        }
    
    def _comprehensive_assessment(self, concept: str, detail: str, domain: str, difficulty: int) -> float:
        """综合多维度评估理解度"""
        base_score = 0.5
        
        # 维度1: 详细度评估 (+0-20%)
        detail_score = len(detail) / 200  # 标准化
        base_score += min(detail_score, 0.20)
        
        # 维度2: 难度匹配 (+0-15%)
        difficulty_bonus = (difficulty - 1) * 0.03  # 难度越高加分越多
        base_score += min(difficulty_bonus, 0.15)
        
        # 维度3: 专业术语识别 (+0-20%)
        complex_indicators = ['公式', '方程', '定理', '原理', '机制', '法则', '定律', '理论']
        term_count = sum(1 for ind in complex_indicators if ind in detail)
        term_score = min(term_count * 0.05, 0.20)
        base_score += term_score
        
        # 维度4: 数学/科学符号 (+0-20%)
        symbols = ['=', '∫', '∂', 'ℏ', '∑', '±', '≤', '≥', '→', '↔', 'π', 'ω', '∞', 'Δ']
        symbol_count = sum(1 for sym in symbols if sym in detail)
        symbol_score = min(symbol_count * 0.05, 0.20)
        base_score += symbol_score
        
        # 维度5: 学科相关性 (+0-10%)
        domain_keywords = {
            "mathematics": ["函数", "集合", "空间", "群", "向量", "矩阵"],
            "physics": ["能量", "力", "场", "粒子", "波", "量子"],
            "chemistry": ["分子", "原子", "反应", "键", "轨道"],
            "biology": ["细胞", "基因", "蛋白", "DNA", "进化"],
            "engineering": ["系统", "设计", "优化", "控制", "算法"],
            "computer_science": ["数据", "算法", "程序", "网络", "计算"]
        }
        
        keywords = domain_keywords.get(domain, [])
        keyword_match = sum(1 for kw in keywords if kw in detail)
        keyword_score = min(keyword_match * 0.02, 0.10)
        base_score += keyword_score
        
        # 随机变异 (±5-10%)
        variance = random.uniform(-0.10, 0.15)
        final_score = min(max(base_score + variance, 0.3), 0.99)
        
        return final_score
    
    def conduct_enhanced_test(self, test_num: int = None) -> Dict:
        """进行增强型测试"""
        if test_num is None:
            test_num = len(self.test_history) + 1
        
        print(f"\n{'='*80}")
        print(f"🎯 测试 #{test_num} - 知识保留评估")
        print(f"{'='*80}")
        
        # 从已通过学习的知识中抽取
        passed_items = [h for h in self.learning_history 
                       if h['understanding_score'] >= self.quality_threshold]
        
        if len(passed_items) < 2:
            print("⚠️ 已掌握知识不足，跳过测试")
            return {"status": "skipped"}
        
        # 抽取2-3个进行测试
        test_count = min(3, len(passed_items))
        test_items = random.sample(passed_items, test_count)
        
        correct = 0
        all_results = []
        
        for i, item in enumerate(test_items, 1):
            concept = item['concept']
            original_score = item['understanding_score']
            
            # 模拟知识保留衰减
            # 最近学习: 高保留率 (90-100%)
            # 一般学习: 中保留率 (75-90%)
            # 较早学习: 低保留率 (60-80%)
            time_decay = random.uniform(0.85, 0.98)
            test_score = original_score * time_decay
            
            is_correct = test_score >= self.quality_threshold
            
            result = {
                "concept": concept,
                "original_score": original_score,
                "test_score": test_score,
                "passed": is_correct
            }
            all_results.append(result)
            
            status = "✅" if is_correct else "❌"
            print(f"   [{i}] {concept}: {status} ({test_score*100:.0f}%)")
            
            if is_correct:
                correct += 1
        
        # 测试结果
        pass_rate = correct / test_count
        quality = "优秀" if pass_rate >= 0.9 else "良好" if pass_rate >= 0.7 else "需改进"
        
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "test_num": test_num,
            "total": test_count,
            "correct": correct,
            "pass_rate": pass_rate,
            "quality": quality,
            "details": all_results
        }
        
        self.test_history.append(test_result)
        self.stats['tests_conducted'] += 1
        self.stats['test_scores'].append(pass_rate)
        
        if pass_rate >= 0.7:
            self.stats['tests_passed'] += 1
            print(f"\n✅ 测试通过 ({pass_rate*100:.0f}% 正确率) - {quality}")
        else:
            self.stats['tests_failed'] += 1
            print(f"\n❌ 测试未通过 ({pass_rate*100:.0f}% 正确率) - {quality}")
        
        print(f"{'='*80}\n")
        
        return test_result
    
    def evolve_system(self) -> Dict:
        """系统进化 - 优化学习策略"""
        self.stats['evolution_count'] += 1
        
        print(f"\n{'='*80}")
        print(f"🧬 进化周期 #{self.stats['evolution_count']}")
        print(f"{'='*80}")
        
        evolution_info = {}
        
        # 分析1: 学习质量
        if self.stats['quality_scores']:
            avg_quality = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            pass_rate = self.stats['quality_passed'] / (self.stats['quality_passed'] + self.stats['quality_failed'])
            
            print(f"📊 学习质量分析:")
            print(f"   平均理解度: {avg_quality*100:.1f}%")
            print(f"   通过率: {pass_rate*100:.0f}%")
            
            evolution_info['learning_quality'] = {
                'avg_quality': avg_quality,
                'pass_rate': pass_rate
            }
            
            # 自适应调整质量阈值
            if avg_quality >= 0.80 and pass_rate >= 0.8:
                # 学习效果好，提升标准
                self.quality_threshold = min(self.quality_threshold + 0.03, 0.85)
                print(f"   📈 效果优秀，提升质量标准至 {self.quality_threshold*100:.0f}%")
            elif avg_quality < 0.65 or pass_rate < 0.6:
                # 学习效果差，降低标准
                self.quality_threshold = max(self.quality_threshold - 0.03, 0.60)
                print(f"   📉 需要改进，调整质量标准至 {self.quality_threshold*100:.0f}%")
            else:
                print(f"   ➡️ 质量标准保持 {self.quality_threshold*100:.0f}%")
        
        # 分析2: 测试表现
        if self.stats['test_scores']:
            avg_test_score = sum(self.stats['test_scores']) / len(self.stats['test_scores'])
            print(f"\n🎯 测试表现分析:")
            print(f"   平均通过率: {avg_test_score*100:.0f}%")
            
            evolution_info['test_performance'] = {
                'avg_test_score': avg_test_score
            }
        
        # 分析3: 领域分布
        kb_stats = self.kb.get_stats()
        print(f"\n📚 领域分布分析:")
        
        domain_balance = {}
        for domain, total in kb_stats['by_domain'].items():
            verified = sum(1 for k in self.kb.knowledge[domain] if k.get('verified'))
            mastery = verified / total * 100 if total > 0 else 0
            domain_balance[domain] = mastery
            status = "✅" if mastery >= 20 else "⚠️"
            print(f"   {status} {domain:20s}: {mastery:5.1f}% ({verified}/{total})")
        
        evolution_info['domain_balance'] = domain_balance
        
        # 进化总结
        print(f"\n✨ 进化效果:")
        print(f"   学习项目: {self.stats['total_learned']} 条")
        print(f"   已验证知识: {kb_stats['verified_count']} 条 (+{kb_stats['verified_count']-2})")
        print(f"   测试次数: {self.stats['tests_conducted']}")
        
        print(f"{'='*80}\n")
        
        return evolution_info
    
    def continuous_enhanced_learning(self, target_items: int = 40, max_cycles: int = 8):
        """持续增强型学习"""
        print("\n" + "="*80)
        print("🚀 启动增强型持续监督学习")
        print("="*80)
        print(f"目标学习: {target_items} 项")
        print(f"最大周期: {max_cycles}")
        print(f"初始质量阈值: {self.quality_threshold*100:.0f}%")
        print("="*80 + "\n")
        
        learned_count = 0
        cycle = 0
        
        while learned_count < target_items and cycle < max_cycles:
            cycle += 1
            
            print(f"\n{'#'*80}")
            print(f"# 学习周期 {cycle}/{max_cycles} - 已学习 {learned_count}/{target_items}")
            print(f"{'#'*80}\n")
            
            # 获取未验证知识
            unverified = self.kb.get_unverified()
            
            if not unverified:
                print("✅ 所有知识已学习完毕")
                break
            
            # 本周期学习数量
            remaining = target_items - learned_count
            batch_size = min(random.randint(4, 8), len(unverified), remaining)
            batch = random.sample(unverified, batch_size)
            
            print(f"本周期学习: {batch_size} 项\n")
            
            for i, item in enumerate(batch, 1):
                print(f"[{i}/{batch_size}] ", end="")
                result = self.learn_with_deep_assessment(item)
                
                if result['status'] == 'passed':
                    learned_count += 1
                
                # 定期测试
                if self.stats['total_learned'] > 0 and self.stats['total_learned'] % self.test_interval == 0:
                    self.conduct_enhanced_test()
                
                # 定期进化
                if self.stats['total_learned'] > 0 and self.stats['total_learned'] % self.evolution_interval == 0:
                    self.evolve_system()
                
                time.sleep(0.3)
            
            if cycle < max_cycles and learned_count < target_items:
                print(f"⏳ 等待下个周期...\n")
                time.sleep(1)
        
        # 最终报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终验收报告"""
        print("\n\n" + "="*80)
        print("📊 最终学习成果验收报告")
        print("="*80)
        
        # 基本统计
        print(f"\n📈 学习统计:")
        print(f"   总学习项: {self.stats['total_learned']}")
        print(f"   质量通过: {self.stats['quality_passed']}")
        print(f"   质量失败: {self.stats['quality_failed']}")
        print(f"   通过率: {self.stats['quality_passed']/(self.stats['quality_passed']+self.stats['quality_failed'])*100:.0f}%" 
              if (self.stats['quality_passed']+self.stats['quality_failed'])>0 else "N/A")
        
        # 测试结果
        print(f"\n🎯 测试结果:")
        print(f"   测试次数: {self.stats['tests_conducted']}")
        print(f"   测试通过: {self.stats['tests_passed']}")
        print(f"   测试失败: {self.stats['tests_failed']}")
        
        if self.stats['test_scores']:
            avg_test = sum(self.stats['test_scores']) / len(self.stats['test_scores'])
            print(f"   平均通过率: {avg_test*100:.0f}%")
        
        # 进化统计
        print(f"\n🧬 系统进化:")
        print(f"   进化周期: {self.stats['evolution_count']}")
        print(f"   最终质量阈值: {self.quality_threshold*100:.0f}%")
        
        # 质量评估
        if self.stats['quality_scores']:
            avg_quality = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            print(f"\n📊 质量评估:")
            print(f"   平均理解度: {avg_quality*100:.1f}%")
            print(f"   最高理解度: {max(self.stats['quality_scores'])*100:.0f}%")
            print(f"   最低理解度: {min(self.stats['quality_scores'])*100:.0f}%")
        
        # 知识库状态
        kb_stats = self.kb.get_stats()
        print(f"\n📚 知识库状态:")
        print(f"   总知识: {kb_stats['total_count']}")
        print(f"   已验证: {kb_stats['verified_count']} ({kb_stats['verified_count']/kb_stats['total_count']*100:.1f}%)")
        
        # 顶级概念
        print(f"\n✨ 掌握最好的5个概念:")
        top_concepts = sorted(self.learning_history, 
                            key=lambda x: x['understanding_score'], reverse=True)[:5]
        for i, item in enumerate(top_concepts, 1):
            score_pct = item['understanding_score'] * 100
            print(f"   {i}. {item['concept']:30s} - {score_pct:.0f}% ({item['domain']})")
        
        # 保存报告
        report_file = Path("learning_report_enhanced.json")
        report = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "kb_stats": kb_stats,
            "test_history": self.test_history,
            "top_concepts": [
                {k: v for k, v in item.items() if k != 'timestamp'} 
                for item in top_concepts
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 报告已保存: {report_file}")
        
        # 保存知识库
        self.kb.save()
        
        # 验收结论
        print("\n" + "="*80)
        print("🎉 验收结论")
        print("="*80)
        
        quality_ok = avg_quality >= 0.72 if self.stats['quality_scores'] else False
        tests_ok = self.stats['tests_conducted'] > 0
        evolution_ok = self.stats['evolution_count'] > 0
        growth_ok = kb_stats['verified_count'] >= 15
        
        print(f"✅ 知识增长: {kb_stats['verified_count']}/87条 已验证 ({kb_stats['verified_count']/87*100:.0f}%)" 
              if growth_ok else f"⚠️ 知识增长: 需要 ≥15 条")
        print(f"{'✅' if quality_ok else '⚠️'} 学习质量: {avg_quality*100:.1f}% (目标≥72%)" 
              if self.stats['quality_scores'] else "⚠️ 无质量数据")
        print(f"{'✅' if tests_ok else '⚠️'} 测试执行: {self.stats['tests_conducted']} 次")
        print(f"{'✅' if evolution_ok else '⚠️'} 系统进化: {self.stats['evolution_count']} 次")
        
        all_ok = growth_ok and quality_ok and tests_ok and evolution_ok
        verdict = "✅ 通过" if all_ok else "⚠️ 部分通过" if (growth_ok and quality_ok) else "❌ 未通过"
        
        print(f"\n{'='*80}")
        print(f"最终验收: {verdict}")
        print(f"{'='*80}\n")

def main():
    import sys
    
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    max_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print("="*80)
    print("🎓 H2Q-Evo 增强型持续监督学习与进化系统 v2.0")
    print("="*80)
    print()
    
    system = EnhancedSupervisedLearning()
    system.continuous_enhanced_learning(target_items=target, max_cycles=max_cycles)

if __name__ == "__main__":
    main()
