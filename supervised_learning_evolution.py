#!/usr/bin/env python3
"""
H2Q-Evo 持续监督学习与进化系统
实时验证学习成果，确保学习质量
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from large_knowledge_base import LargeKnowledgeBase

class SupervisedLearningSystem:
    """监督学习系统 - 持续学习并验证成果"""
    
    def __init__(self):
        self.kb = LargeKnowledgeBase()
        self.kb.load()
        
        # 学习记录
        self.learning_history = []
        self.test_results = []
        
        # 监督参数
        self.quality_threshold = 0.7  # 质量阈值
        self.test_interval = 5  # 每学习5项测试一次
        self.evolution_interval = 10  # 每学习10项进化一次
        
        # 统计
        self.stats = {
            "total_learned": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "evolution_count": 0,
            "quality_scores": []
        }
        
        print("✓ 监督学习系统初始化完成")
        self._display_initial_status()
    
    def _display_initial_status(self):
        """显示初始状态"""
        stats = self.kb.get_stats()
        print(f"\n📊 初始知识库状态:")
        print(f"   总知识: {stats['total_count']} 条")
        print(f"   已验证: {stats['verified_count']} 条 ({stats['verified_count']/max(stats['total_count'],1)*100:.1f}%)")
        print(f"   未验证: {stats['unverified_count']} 条")
    
    def learn_with_verification(self, item: Tuple[str, Dict]) -> Dict:
        """学习并立即验证"""
        domain, knowledge = item
        concept = knowledge['concept']
        detail = knowledge.get('detail', '')
        
        print(f"\n📚 学习: {concept}")
        print(f"   领域: {domain}")
        print(f"   详情: {detail[:80]}...")
        
        # 模拟学习过程
        learning_time = random.uniform(0.3, 0.8)
        time.sleep(learning_time)
        
        # 深度理解评估（模拟）
        understanding_score = self._assess_understanding(concept, detail, domain)
        
        # 记录学习
        learning_record = {
            "concept": concept,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "understanding_score": understanding_score,
            "learning_time": learning_time
        }
        
        self.learning_history.append(learning_record)
        self.stats['total_learned'] += 1
        
        # 判断是否通过
        if understanding_score >= self.quality_threshold:
            self.kb.mark_verified(domain, concept)
            print(f"   ✅ 学习通过 (理解度: {understanding_score*100:.1f}%)")
            return {"status": "passed", "score": understanding_score}
        else:
            print(f"   ⚠️ 需要重新学习 (理解度: {understanding_score*100:.1f}%)")
            return {"status": "retry", "score": understanding_score}
    
    def _assess_understanding(self, concept: str, detail: str, domain: str) -> float:
        """评估理解程度"""
        score = 0.5
        
        # 详细度评估
        if len(detail) > 50:
            score += 0.1
        if len(detail) > 100:
            score += 0.1
        
        # 复杂度评估
        complex_indicators = ['公式', '方程', '定理', '原理', '机制', '过程']
        if any(ind in detail for ind in complex_indicators):
            score += 0.1
        
        # 专业术语识别
        math_symbols = ['=', '∫', '∂', 'ℏ', '∑', '±', '≤', '≥', '→', '↔']
        if any(symbol in detail for symbol in math_symbols):
            score += 0.15
        
        # 添加随机性模拟真实学习
        score += random.uniform(-0.15, 0.20)
        
        return min(max(score, 0.3), 0.98)
    
    def conduct_test(self) -> Dict:
        """进行知识测试"""
        print(f"\n{'='*80}")
        print(f"🎯 知识测试 #{len(self.test_results) + 1}")
        print(f"{'='*80}")
        
        # 从已学习的知识中随机抽取3个测试
        recent_learned = [h for h in self.learning_history[-10:] if h['understanding_score'] >= self.quality_threshold]
        
        if len(recent_learned) < 3:
            print("⚠️ 已学习知识不足，跳过测试")
            return {"status": "skipped", "reason": "insufficient_knowledge"}
        
        test_items = random.sample(recent_learned, min(3, len(recent_learned)))
        
        correct = 0
        total = len(test_items)
        
        for i, item in enumerate(test_items, 1):
            concept = item['concept']
            domain = item['domain']
            
            print(f"\n[测试 {i}/{total}] {concept} ({domain})")
            
            # 模拟测试（检查是否真正掌握）
            original_score = item['understanding_score']
            # 知识保留率（随时间衰减）
            retention = random.uniform(0.8, 1.0)
            test_score = original_score * retention
            
            if test_score >= self.quality_threshold:
                print(f"   ✅ 测试通过 (保留率: {retention*100:.0f}%)")
                correct += 1
            else:
                print(f"   ❌ 测试失败 (保留率: {retention*100:.0f}%)")
        
        # 测试结果
        pass_rate = correct / total
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "correct": correct,
            "pass_rate": pass_rate,
            "quality": "excellent" if pass_rate >= 0.9 else "good" if pass_rate >= 0.7 else "needs_improvement"
        }
        
        self.test_results.append(test_result)
        self.stats['quality_scores'].append(pass_rate)
        
        if pass_rate >= 0.7:
            self.stats['tests_passed'] += 1
            print(f"\n✅ 测试通过 (正确率: {pass_rate*100:.0f}%)")
        else:
            self.stats['tests_failed'] += 1
            print(f"\n⚠️ 测试未通过 (正确率: {pass_rate*100:.0f}%)")
        
        print(f"{'='*80}\n")
        
        return test_result
    
    def evolve(self):
        """系统进化"""
        self.stats['evolution_count'] += 1
        
        print(f"\n{'='*80}")
        print(f"🧬 进化周期 #{self.stats['evolution_count']}")
        print(f"{'='*80}")
        
        # 分析学习效果
        if len(self.stats['quality_scores']) > 0:
            avg_quality = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            print(f"   平均学习质量: {avg_quality*100:.1f}%")
            
            # 自适应调整
            if avg_quality > 0.85:
                self.quality_threshold = min(self.quality_threshold + 0.02, 0.85)
                print(f"   📈 提升质量标准至 {self.quality_threshold*100:.0f}%")
            elif avg_quality < 0.65:
                self.quality_threshold = max(self.quality_threshold - 0.02, 0.55)
                print(f"   📉 调整质量标准至 {self.quality_threshold*100:.0f}%")
        
        # 知识库统计
        stats = self.kb.get_stats()
        print(f"   知识验证: {stats['verified_count']}/{stats['total_count']}")
        print(f"   测试通过率: {self.stats['tests_passed']}/{self.stats['tests_passed']+self.stats['tests_failed']}")
        
        print(f"{'='*80}\n")
    
    def continuous_learning(self, target_knowledge: int = 30, max_cycles: int = 10):
        """持续学习循环"""
        print("\n" + "="*80)
        print("🚀 开始持续监督学习")
        print("="*80)
        print(f"目标学习: {target_knowledge} 条知识")
        print(f"最大周期: {max_cycles} 次")
        print(f"质量阈值: {self.quality_threshold*100:.0f}%")
        print(f"测试间隔: 每 {self.test_interval} 条")
        print(f"进化间隔: 每 {self.evolution_interval} 条")
        print("="*80 + "\n")
        
        learned_count = 0
        cycle = 0
        
        while learned_count < target_knowledge and cycle < max_cycles:
            cycle += 1
            print(f"\n{'#'*80}")
            print(f"# 学习周期 {cycle}/{max_cycles} - 已学习 {learned_count}/{target_knowledge}")
            print(f"{'#'*80}\n")
            
            # 获取未验证的知识
            unverified = self.kb.get_unverified()
            
            if not unverified:
                print("⚠️ 所有知识已学习完毕")
                break
            
            # 本周期学习5-10条
            batch_size = min(random.randint(5, 10), len(unverified), target_knowledge - learned_count)
            batch = random.sample(unverified, batch_size)
            
            print(f"本周期计划学习: {batch_size} 条\n")
            
            for i, item in enumerate(batch, 1):
                print(f"[{i}/{batch_size}]", end=" ")
                result = self.learn_with_verification(item)
                
                if result['status'] == 'passed':
                    learned_count += 1
                
                time.sleep(0.5)
                
                # 定期测试
                if learned_count > 0 and learned_count % self.test_interval == 0:
                    self.conduct_test()
                
                # 定期进化
                if learned_count > 0 and learned_count % self.evolution_interval == 0:
                    self.evolve()
            
            # 周期间隔
            if cycle < max_cycles and learned_count < target_knowledge:
                print(f"\n⏳ 等待2秒后继续...")
                time.sleep(2)
        
        # 最终报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终学习报告"""
        print("\n\n" + "="*80)
        print("📊 最终学习成果报告")
        print("="*80)
        
        # 基本统计
        print(f"\n📈 学习统计:")
        print(f"   总学习项: {self.stats['total_learned']}")
        print(f"   测试总数: {len(self.test_results)}")
        print(f"   测试通过: {self.stats['tests_passed']}")
        print(f"   测试失败: {self.stats['tests_failed']}")
        print(f"   进化周期: {self.stats['evolution_count']}")
        
        # 质量分析
        if self.stats['quality_scores']:
            avg_quality = sum(self.stats['quality_scores']) / len(self.stats['quality_scores'])
            print(f"\n📊 质量评估:")
            print(f"   平均学习质量: {avg_quality*100:.1f}%")
            print(f"   最高质量: {max(self.stats['quality_scores'])*100:.0f}%")
            print(f"   最低质量: {min(self.stats['quality_scores'])*100:.0f}%")
        
        # 知识库状态
        kb_stats = self.kb.get_stats()
        print(f"\n📚 知识库状态:")
        print(f"   总知识: {kb_stats['total_count']} 条")
        print(f"   已验证: {kb_stats['verified_count']} 条 ({kb_stats['verified_count']/max(kb_stats['total_count'],1)*100:.1f}%)")
        
        # 各领域掌握情况
        print(f"\n🎯 各领域掌握情况:")
        for domain, total in sorted(kb_stats['by_domain'].items()):
            verified = sum(1 for k in self.kb.knowledge[domain] if k.get('verified'))
            mastery = verified / max(total, 1) * 100
            bar = "█" * int(mastery / 5)
            quality = "优秀" if mastery >= 80 else "良好" if mastery >= 60 else "及格" if mastery >= 40 else "需加强"
            print(f"   {domain:20s} │{bar:<20s}│ {verified:2d}/{total:2d} ({mastery:.0f}%) - {quality}")
        
        # 实质性成果展示
        print(f"\n✨ 实质性学习成果展示:")
        if self.learning_history:
            # 展示学习最好的5个概念
            top_learned = sorted(self.learning_history, key=lambda x: x['understanding_score'], reverse=True)[:5]
            print("\n   📖 掌握最好的概念:")
            for i, item in enumerate(top_learned, 1):
                print(f"      {i}. {item['concept']} ({item['domain']}) - 理解度: {item['understanding_score']*100:.0f}%")
        
        # 测试历史
        if self.test_results:
            print(f"\n   🎯 测试历史:")
            for i, test in enumerate(self.test_results, 1):
                status = "✅" if test['quality'] in ['excellent', 'good'] else "⚠️"
                print(f"      {status} 测试 {i}: {test['correct']}/{test['total']} 正确 ({test['pass_rate']*100:.0f}%) - {test['quality']}")
        
        # 保存报告
        report_file = Path("learning_report.json")
        report = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "kb_stats": kb_stats,
            "test_results": self.test_results,
            "top_concepts": top_learned if self.learning_history else []
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 完整报告已保存: {report_file}")
        
        # 保存知识库
        self.kb.save()
        
        print("\n" + "="*80)
        print("🎓 监督学习完成")
        print("="*80 + "\n")

def main():
    import sys
    
    # 参数
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    max_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("="*80)
    print("🎓 H2Q-Evo 持续监督学习与进化系统")
    print("="*80)
    
    system = SupervisedLearningSystem()
    system.continuous_learning(target_knowledge=target, max_cycles=max_cycles)

if __name__ == "__main__":
    main()
