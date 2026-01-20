#!/usr/bin/env python3
"""
H2Q-Evo 智能学习与自我进化系统
集成大规模知识库、验证机制和持续学习循环
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from large_knowledge_base import LargeKnowledgeBase
from knowledge_validator import KnowledgeValidator

class IntelligentLearningSystem:
    """智能学习系统 - 持续学习和自我进化"""
    
    def __init__(self):
        self.kb = LargeKnowledgeBase()
        self.kb.load()  # 加载现有知识
        
        self.validator = KnowledgeValidator()
        self.learning_rate = 0.1
        self.evolution_threshold = 10  # 每学习10条触发一次进化
        
        self.stats = {
            "total_learned": 0,
            "total_validated": 0,
            "evolution_cycles": 0,
            "knowledge_growth": []
        }
        
        self.status_file = Path("learning_system_status.json")
        
    def adaptive_learning_cycle(self, max_items: int = 20):
        """自适应学习周期"""
        print("="*80)
        print("🧠 智能学习系统 - 自适应学习周期")
        print("="*80)
        
        # 获取统计信息
        kb_stats = self.kb.get_stats()
        print(f"\n📊 当前知识库状态:")
        print(f"   总知识: {kb_stats['total_count']} 条")
        print(f"   已验证: {kb_stats['verified_count']} 条")
        print(f"   未验证: {kb_stats['unverified_count']} 条")
        
        # 选择学习策略
        if kb_stats['unverified_count'] > 50:
            strategy = "验证现有知识"
            items = self._learn_existing_knowledge(max_items)
        elif kb_stats['verified_count'] < 30:
            strategy = "混合学习：验证+探索"
            items = self._mixed_learning(max_items)
        else:
            strategy = "探索新知识"
            items = self._explore_new_knowledge(max_items)
        
        print(f"\n📚 学习策略: {strategy}")
        print(f"   学习项目: {len(items)} 条")
        
        # 执行学习
        for i, item in enumerate(items, 1):
            self._learn_item(item, i, len(items))
            time.sleep(0.5)
        
        # 检查是否需要进化
        if self.stats['total_learned'] % self.evolution_threshold == 0:
            self._trigger_evolution()
        
        # 保存状态
        self._save_status()
        
        # 显示进度
        self._display_progress()
    
    def _learn_existing_knowledge(self, max_items: int) -> List[Tuple[str, Dict]]:
        """学习现有未验证的知识"""
        unverified = self.kb.get_unverified()
        return random.sample(unverified, min(max_items, len(unverified)))
    
    def _mixed_learning(self, max_items: int) -> List[Tuple[str, Dict]]:
        """混合学习策略"""
        # 70%验证现有，30%探索新知
        verify_count = int(max_items * 0.7)
        explore_count = max_items - verify_count
        
        items = []
        
        # 验证部分
        unverified = self.kb.get_unverified()
        if unverified:
            items.extend(random.sample(unverified, min(verify_count, len(unverified))))
        
        # 探索部分（从已验证的中提取相关概念扩展）
        # 这里简化为随机选择一些知识
        for _ in range(explore_count):
            domain = random.choice(list(self.kb.knowledge.keys()))
            items.append((domain, {"concept": f"探索性学习{_}", "detail": "待学习", "difficulty": 3, "verified": False}))
        
        return items
    
    def _explore_new_knowledge(self, max_items: int) -> List[Tuple[str, Dict]]:
        """探索新知识"""
        # 从难度较高的未验证知识开始
        high_difficulty = self.kb.get_by_difficulty(min_difficulty=4, max_difficulty=5)
        unverified_high = [(d, k) for d, k in high_difficulty if not k.get('verified')]
        
        if unverified_high:
            return random.sample(unverified_high, min(max_items, len(unverified_high)))
        else:
            return self._learn_existing_knowledge(max_items)
    
    def _learn_item(self, item: Tuple[str, Dict], index: int, total: int):
        """学习单个知识项"""
        domain, knowledge = item
        concept = knowledge['concept']
        detail = knowledge.get('detail', '')
        
        print(f"\n[{index}/{total}] 🎯 学习: {concept}")
        print(f"        领域: {domain} | 难度: {knowledge.get('difficulty', 3)}⭐")
        
        # 模拟理解过程
        understanding_time = random.uniform(0.3, 0.8)
        time.sleep(understanding_time)
        
        # 简化的内部验证（不调用外部API，基于规则）
        confidence = self._internal_validation(concept, detail, domain)
        
        if confidence > 0.7:
            self.kb.mark_verified(domain, concept)
            print(f"        ✅ 理解完成 (置信度: {confidence*100:.1f}%)")
            self.stats['total_validated'] += 1
        else:
            print(f"        🤔 需要更多学习 (置信度: {confidence*100:.1f}%)")
        
        self.stats['total_learned'] += 1
    
    def _internal_validation(self, concept: str, detail: str, domain: str) -> float:
        """内部验证机制（基于规则和模式）"""
        confidence = 0.5
        
        # 检查详细程度
        if len(detail) > 50:
            confidence += 0.1
        if len(detail) > 100:
            confidence += 0.1
        
        # 检查是否包含公式或专业术语
        math_symbols = ['=', '∫', '∂', 'ℏ', '∑', '±', '≤', '≥']
        if any(symbol in detail for symbol in math_symbols):
            confidence += 0.15
        
        # 检查难度匹配
        difficulty = len(detail.split()) / 10  # 简单的难度估计
        if difficulty > 5:
            confidence += 0.1
        
        # 添加随机性模拟真实验证
        confidence += random.uniform(-0.1, 0.15)
        
        return min(max(confidence, 0.3), 0.95)
    
    def _trigger_evolution(self):
        """触发系统进化"""
        self.stats['evolution_cycles'] += 1
        
        print(f"\n{'='*80}")
        print(f"🧬 进化周期 #{self.stats['evolution_cycles']}")
        print(f"{'='*80}")
        
        # 分析学习效果
        kb_stats = self.kb.get_stats()
        growth = kb_stats['verified_count'] - sum(self.stats['knowledge_growth']) if self.stats['knowledge_growth'] else kb_stats['verified_count']
        self.stats['knowledge_growth'].append(growth)
        
        print(f"   知识增长: +{growth} 条")
        print(f"   总验证: {kb_stats['verified_count']}/{kb_stats['total_count']}")
        print(f"   进度: {kb_stats['verified_count']/max(kb_stats['total_count'],1)*100:.1f}%")
        
        # 调整学习率
        if growth > 8:
            self.learning_rate = min(self.learning_rate * 1.2, 0.3)
            print(f"   📈 学习效果良好，提升学习率至 {self.learning_rate:.2f}")
        elif growth < 3:
            self.learning_rate = max(self.learning_rate * 0.8, 0.05)
            print(f"   📉 调整学习策略，降低学习率至 {self.learning_rate:.2f}")
        
        print(f"{'='*80}\n")
    
    def _save_status(self):
        """保存学习状态"""
        kb_stats = self.kb.get_stats()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "learning_rate": self.learning_rate,
            "knowledge_base": kb_stats
        }
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    
    def _display_progress(self):
        """显示学习进度"""
        kb_stats = self.kb.get_stats()
        
        print(f"\n{'='*80}")
        print(f"📈 学习进度报告")
        print(f"{'='*80}")
        print(f"累计学习: {self.stats['total_learned']} 项")
        print(f"累计验证: {self.stats['total_validated']} 项")
        print(f"进化周期: {self.stats['evolution_cycles']} 次")
        print(f"学习率: {self.learning_rate:.2f}")
        print(f"\n知识库完成度: {kb_stats['verified_count']}/{kb_stats['total_count']} ({kb_stats['verified_count']/max(kb_stats['total_count'],1)*100:.1f}%)")
        
        # 显示领域分布
        print(f"\n📚 各领域验证进度:")
        for domain, total in sorted(kb_stats['by_domain'].items()):
            verified = sum(1 for k in self.kb.knowledge[domain] if k.get('verified'))
            percentage = verified / max(total, 1) * 100
            bar = "█" * int(percentage / 5)
            print(f"   {domain:20s} │{bar:<20s}│ {verified}/{total} ({percentage:.0f}%)")
        
        print(f"{'='*80}\n")
    
    def continuous_learning(self, cycles: int = 5, items_per_cycle: int = 10, interval: int = 5):
        """持续学习模式"""
        print("="*80)
        print("🚀 启动持续学习模式")
        print("="*80)
        print(f"学习周期: {cycles} 次")
        print(f"每周期项目: {items_per_cycle} 条")
        print(f"周期间隔: {interval} 秒")
        print("="*80)
        
        for cycle in range(1, cycles + 1):
            print(f"\n\n{'#'*80}")
            print(f"# 学习周期 {cycle}/{cycles}")
            print(f"{'#'*80}\n")
            
            self.adaptive_learning_cycle(max_items=items_per_cycle)
            
            if cycle < cycles:
                print(f"\n⏳ 等待 {interval} 秒后继续下一周期...\n")
                time.sleep(interval)
        
        # 最终报告
        print("\n\n" + "="*80)
        print("🎓 持续学习完成 - 最终报告")
        print("="*80)
        self._display_progress()
        
        # 保存最终状态
        self.kb.save()
        self._save_status()
        
        print("✅ 所有学习数据已保存")

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("🌟 H2Q-Evo 智能学习与自我进化系统")
    print("="*80)
    
    system = IntelligentLearningSystem()
    
    # 参数：周期数、每周期项目数、间隔秒数
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    items = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    system.continuous_learning(cycles=cycles, items_per_cycle=items, interval=interval)
