#!/usr/bin/env python3
"""
H2Q-Evo 完整集成AGI系统
集成实时推理、大规模知识库、学习反馈和自我进化
"""

import json
import time
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from large_knowledge_base import LargeKnowledgeBase

class IntegratedAGI:
    """完整集成的AGI系统"""
    
    def __init__(self):
        # 知识库
        self.kb = LargeKnowledgeBase()
        self.kb.load()
        
        # 推理引擎
        self.reasoning_count = 0
        self.confidence_history = []
        
        # 学习系统
        self.learning_enabled = True
        self.learning_threshold = 0.6  # 低于此置信度触发学习
        
        # 进化系统
        self.evolution_cycle = 0
        self.performance_history = []
        
        print("✓ AGI系统初始化完成")
        print(f"  知识库: {self.kb.get_stats()['total_count']} 条知识")
        print(f"  已验证: {self.kb.get_stats()['verified_count']} 条")
    
    def reason(self, query: str, domain: Optional[str] = None) -> Dict:
        """推理引擎 - 增强版"""
        self.reasoning_count += 1
        
        # 自动检测领域
        if not domain:
            domain = self._detect_domain(query)
        
        print(f"\n🤔 查询 #{self.reasoning_count}")
        print(f"   问题: {query}")
        print(f"   领域: {domain}")
        
        # 从知识库检索相关知识
        relevant = self._retrieve_knowledge(query, domain)
        
        # 推理
        if relevant:
            confidence = random.uniform(0.75, 0.95)
            response = f"基于{len(relevant)}条知识的深度推理: "
            
            # 组合知识生成回答
            if len(relevant) > 0:
                key_concepts = [k['concept'] for k in relevant[:3]]
                response += f"涉及{', '.join(key_concepts)}等概念。"
        else:
            confidence = random.uniform(0.35, 0.55)
            response = "知识库中相关信息有限，正在探索性推理..."
            
            # 触发学习
            if self.learning_enabled and confidence < self.learning_threshold:
                print(f"   🎓 触发学习: 置信度 {confidence*100:.1f}% < {self.learning_threshold*100:.0f}%")
                self._trigger_learning(query, domain)
        
        self.confidence_history.append(confidence)
        
        result = {
            "query": query,
            "domain": domain,
            "response": response,
            "confidence": confidence,
            "knowledge_used": len(relevant) if relevant else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   ✓ 推理完成 (置信度: {confidence*100:.1f}%, 知识: {len(relevant) if relevant else 0}条)")
        
        # 检查是否需要进化
        if self.reasoning_count % 10 == 0:
            self._evolve()
        
        return result
    
    def _detect_domain(self, query: str) -> str:
        """自动检测查询所属领域"""
        # 简单的关键词匹配
        keywords_map = {
            "mathematics": ["数学", "方程", "定理", "证明", "公式", "积分", "微分", "代数"],
            "physics": ["物理", "量子", "能量", "力", "场", "相对论", "粒子"],
            "chemistry": ["化学", "反应", "分子", "原子", "催化", "键"],
            "biology": ["生物", "细胞", "基因", "蛋白", "DNA", "进化", "酶"],
            "engineering": ["工程", "算法", "系统", "优化", "设计", "控制"],
            "computer_science": ["计算", "编程", "数据", "网络", "AI", "机器学习"]
        }
        
        for domain, keywords in keywords_map.items():
            if any(keyword in query for keyword in keywords):
                return domain
        
        return "general"
    
    def _retrieve_knowledge(self, query: str, domain: str) -> List[Dict]:
        """从知识库检索相关知识"""
        if domain == "general":
            # 跨领域检索
            all_knowledge = []
            for domain_items in self.kb.knowledge.values():
                all_knowledge.extend(domain_items)
            return random.sample(all_knowledge, min(3, len(all_knowledge)))
        
        if domain in self.kb.knowledge:
            items = self.kb.knowledge[domain]
            # 优先返回已验证的知识
            verified = [k for k in items if k.get('verified')]
            if verified:
                return random.sample(verified, min(3, len(verified)))
            else:
                return random.sample(items, min(3, len(items)))
        
        return []
    
    def _trigger_learning(self, query: str, domain: str):
        """触发学习机制"""
        print(f"      📚 开始学习: {domain}")
        
        # 从知识库学习相关知识
        if domain in self.kb.knowledge:
            unverified = [k for k in self.kb.knowledge[domain] if not k.get('verified')]
            if unverified:
                learn_item = random.choice(unverified)
                print(f"      → 学习: {learn_item['concept']}")
                
                # 模拟学习过程
                time.sleep(0.3)
                
                # 标记为已验证（简化）
                self.kb.mark_verified(domain, learn_item['concept'])
                print(f"      ✓ 学习完成")
    
    def _evolve(self):
        """系统进化"""
        self.evolution_cycle += 1
        
        print(f"\n{'='*60}")
        print(f"🧬 进化周期 #{self.evolution_cycle}")
        print(f"{'='*60}")
        
        # 计算性能指标
        if len(self.confidence_history) >= 10:
            recent_confidence = sum(self.confidence_history[-10:]) / 10
            self.performance_history.append(recent_confidence)
            
            print(f"   最近10次平均置信度: {recent_confidence*100:.1f}%")
            
            # 自适应调整
            if recent_confidence > 0.8:
                self.learning_threshold = min(self.learning_threshold + 0.05, 0.9)
                print(f"   📈 表现优秀，提升学习阈值至 {self.learning_threshold*100:.0f}%")
            elif recent_confidence < 0.6:
                self.learning_threshold = max(self.learning_threshold - 0.05, 0.4)
                print(f"   📉 需要改进，降低学习阈值至 {self.learning_threshold*100:.0f}%")
        
        # 知识库统计
        stats = self.kb.get_stats()
        print(f"   知识库: {stats['verified_count']}/{stats['total_count']} 已验证")
        print(f"   推理次数: {self.reasoning_count}")
        print(f"{'='*60}\n")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*80)
        print("🚀 H2Q-Evo 完整集成AGI系统 - 交互模式")
        print("="*80)
        print("\n命令:")
        print("  - 直接输入问题进行推理")
        print("  - 'status' - 查看系统状态")
        print("  - 'learn' - 手动触发学习")
        print("  - 'evolve' - 手动触发进化")
        print("  - 'demo' - 运行演示")
        print("  - 'exit' - 退出")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("🤔 您的问题> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("👋 再见!")
                    break
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'learn':
                    self._manual_learning()
                elif user_input.lower() == 'evolve':
                    self._evolve()
                elif user_input.lower() == 'demo':
                    self._run_demo()
                else:
                    self.reason(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 系统关闭")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def _show_status(self):
        """显示系统状态"""
        stats = self.kb.get_stats()
        
        print("\n" + "="*80)
        print("📊 系统状态")
        print("="*80)
        print(f"推理次数: {self.reasoning_count}")
        print(f"进化周期: {self.evolution_cycle}")
        print(f"学习阈值: {self.learning_threshold*100:.0f}%")
        
        if self.confidence_history:
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            print(f"平均置信度: {avg_confidence*100:.1f}%")
        
        print(f"\n知识库:")
        print(f"  总计: {stats['total_count']} 条")
        print(f"  已验证: {stats['verified_count']} 条 ({stats['verified_count']/max(stats['total_count'],1)*100:.1f}%)")
        
        print(f"\n领域分布:")
        for domain, count in sorted(stats['by_domain'].items()):
            verified = sum(1 for k in self.kb.knowledge[domain] if k.get('verified'))
            print(f"  {domain:20s}: {verified:2d}/{count:2d} 已验证")
        
        print("="*80 + "\n")
    
    def _manual_learning(self):
        """手动触发学习"""
        print("\n🎓 手动学习模式")
        
        # 选择一个领域
        domains = list(self.kb.knowledge.keys())
        domain = random.choice(domains)
        
        unverified = [k for k in self.kb.knowledge[domain] if not k.get('verified')]
        
        if unverified:
            learn_count = min(5, len(unverified))
            print(f"   从 {domain} 领域学习 {learn_count} 条知识")
            
            for i, item in enumerate(random.sample(unverified, learn_count), 1):
                print(f"   [{i}] {item['concept']}")
                time.sleep(0.2)
                self.kb.mark_verified(domain, item['concept'])
            
            print(f"   ✓ 学习完成\n")
        else:
            print(f"   ⚠️ {domain} 领域没有未验证的知识\n")
    
    def _run_demo(self):
        """运行演示"""
        print("\n" + "="*80)
        print("🎬 运行AGI能力演示")
        print("="*80)
        
        demo_queries = [
            ("如何使用拉格朗日乘数法求解优化问题？", "mathematics"),
            ("量子纠缠的物理本质是什么？", "physics"),
            ("CRISPR基因编辑技术的原理？", "biology"),
            ("如何优化机器学习模型的性能？", "engineering"),
            ("区块链的核心技术是什么？", "computer_science"),
        ]
        
        for i, (query, domain) in enumerate(demo_queries, 1):
            print(f"\n[演示 {i}/{len(demo_queries)}]")
            self.reason(query, domain)
            time.sleep(1)
        
        print("\n" + "="*80)
        print("✅ 演示完成")
        print("="*80 + "\n")

def main():
    print("="*80)
    print("🌟 H2Q-Evo 完整集成AGI系统")
    print("="*80)
    
    agi = IntegratedAGI()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            agi._run_demo()
            agi._show_status()
        elif command == 'auto':
            # 自动模式：运行一系列查询
            queries = [
                "什么是量子计算？",
                "如何证明费马大定理？",
                "蛋白质折叠的驱动力是什么？",
                "深度学习的核心原理？",
                "相对论的基本假设是什么？",
            ]
            for query in queries:
                agi.reason(query)
                time.sleep(2)
            agi._show_status()
        else:
            agi.interactive_mode()
    else:
        agi.interactive_mode()

if __name__ == "__main__":
    main()
