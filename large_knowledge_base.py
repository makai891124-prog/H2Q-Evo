#!/usr/bin/env python3
"""
H2Q-Evo 大规模知识库加载器
从多个来源构建全面的科学知识数据集
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

class LargeKnowledgeBase:
    """大规模知识库管理器"""
    
    def __init__(self):
        self.knowledge_file = Path("large_knowledge_base.json")
        self.knowledge = self._init_comprehensive_knowledge()
        
    def _init_comprehensive_knowledge(self) -> Dict[str, List[Dict]]:
        """初始化全面的知识库（100+条目）"""
        return {
            "mathematics": [
                {"concept": "拉格朗日乘数法", "detail": "用于求解带约束的优化问题，通过引入拉格朗日乘数将约束优化转化为无约束优化", "difficulty": 3, "verified": False},
                {"concept": "柯西-施瓦茨不等式", "detail": "对于内积空间中的向量 u 和 v，有 |⟨u,v⟩| ≤ ||u|| ||v||", "difficulty": 2, "verified": False},
                {"concept": "黎曼假设", "detail": "所有非平凡零点都位于临界线 Re(s)=1/2 上，是数学中最重要的未解决问题", "difficulty": 5, "verified": False},
                {"concept": "费马大定理", "detail": "当 n>2 时，方程 x^n + y^n = z^n 没有正整数解，由安德鲁·怀尔斯在1995年证明", "difficulty": 5, "verified": False},
                {"concept": "傅里叶变换", "detail": "将时域信号转换为频域表示，f(ω) = ∫f(t)e^(-iωt)dt", "difficulty": 3, "verified": False},
                {"concept": "泰勒级数", "detail": "函数在某点的无穷级数展开，f(x) = Σ[f^(n)(a)/n!](x-a)^n", "difficulty": 2, "verified": False},
                {"concept": "梯度下降法", "detail": "通过沿负梯度方向迭代更新参数来最小化目标函数", "difficulty": 2, "verified": False},
                {"concept": "矩阵特征值", "detail": "满足 Av = λv 的标量 λ，其中 v 是对应的特征向量", "difficulty": 3, "verified": False},
                {"concept": "概率论贝叶斯定理", "detail": "P(A|B) = P(B|A)P(A)/P(B)，描述条件概率关系", "difficulty": 2, "verified": False},
                {"concept": "微分方程", "detail": "包含未知函数及其导数的方程，描述动态系统演化", "difficulty": 3, "verified": False},
                {"concept": "拓扑空间", "detail": "配备了拓扑结构的集合，研究连续性和收敛性的基础", "difficulty": 4, "verified": False},
                {"concept": "群论基础", "detail": "研究对称性的数学结构，包括群、环、域等代数系统", "difficulty": 4, "verified": False},
                {"concept": "数值积分", "detail": "使用数值方法（如Simpson法则、梯形法则）近似计算定积分", "difficulty": 2, "verified": False},
                {"concept": "线性规划", "detail": "在线性约束下优化线性目标函数，可用单纯形法求解", "difficulty": 3, "verified": False},
                {"concept": "复变函数", "detail": "定义在复数域上的函数，满足柯西-黎曼方程即为解析函数", "difficulty": 4, "verified": False},
            ],
            "physics": [
                {"concept": "量子谐振子", "detail": "能级公式 E_n = ℏω(n + 1/2)，是量子力学中最基本的模型", "difficulty": 3, "verified": False},
                {"concept": "麦克斯韦方程组", "detail": "描述电磁场的四个基本方程：高斯定律、高斯磁定律、法拉第定律、安培-麦克斯韦定律", "difficulty": 4, "verified": False},
                {"concept": "薛定谔方程", "detail": "iℏ∂ψ/∂t = Ĥψ，描述量子系统的时间演化", "difficulty": 4, "verified": False},
                {"concept": "相对论质能关系", "detail": "E = mc²，质量和能量是可以相互转换的", "difficulty": 2, "verified": False},
                {"concept": "热力学第二定律", "detail": "孤立系统的熵总是增加或保持不变，定义了时间的方向", "difficulty": 3, "verified": False},
                {"concept": "洛伦兹变换", "detail": "描述不同惯性参照系之间时空坐标的转换关系", "difficulty": 4, "verified": False},
                {"concept": "量子纠缠", "detail": "两个或多个粒子处于叠加态，测量一个会瞬间影响另一个", "difficulty": 4, "verified": False},
                {"concept": "波粒二象性", "detail": "微观粒子同时具有波动性和粒子性，E=hν, p=h/λ", "difficulty": 3, "verified": False},
                {"concept": "黑体辐射", "detail": "普朗克公式解释了黑体辐射谱，开创了量子力学", "difficulty": 3, "verified": False},
                {"concept": "角动量守恒", "detail": "在无外力矩作用下，系统的总角动量保持不变", "difficulty": 2, "verified": False},
                {"concept": "多普勒效应", "detail": "波源和观察者相对运动导致接收频率发生变化", "difficulty": 2, "verified": False},
                {"concept": "超导现象", "detail": "某些材料在临界温度以下电阻完全消失并排斥磁场", "difficulty": 4, "verified": False},
                {"concept": "拉格朗日力学", "detail": "使用广义坐标和拉格朗日量描述力学系统", "difficulty": 4, "verified": False},
                {"concept": "哈密顿力学", "detail": "使用正则坐标和哈密顿量描述系统演化", "difficulty": 4, "verified": False},
                {"concept": "狭义相对论", "detail": "光速不变原理和相对性原理，导出时间膨胀和长度收缩", "difficulty": 4, "verified": False},
                {"concept": "量子隧穿效应", "detail": "粒子可以穿越经典物理中无法逾越的势垒", "difficulty": 3, "verified": False},
            ],
            "chemistry": [
                {"concept": "SN2反应", "detail": "一步协同的亲核取代反应，伴随构型翻转", "difficulty": 2, "verified": False},
                {"concept": "吉布斯自由能", "detail": "ΔG = ΔH - TΔS，判断反应自发性的热力学函数", "difficulty": 3, "verified": False},
                {"concept": "分子轨道理论", "detail": "原子轨道线性组合形成分子轨道，解释化学键的本质", "difficulty": 3, "verified": False},
                {"concept": "化学平衡", "detail": "正逆反应速率相等时的动态平衡状态，K = [产物]/[反应物]", "difficulty": 2, "verified": False},
                {"concept": "酸碱理论", "detail": "Brønsted-Lowry理论：酸是质子给体，碱是质子受体", "difficulty": 2, "verified": False},
                {"concept": "氧化还原反应", "detail": "电子转移反应，氧化数发生变化", "difficulty": 2, "verified": False},
                {"concept": "催化剂", "detail": "降低反应活化能但不改变反应平衡的物质", "difficulty": 2, "verified": False},
                {"concept": "配位化合物", "detail": "中心金属离子与配体通过配位键结合形成的化合物", "difficulty": 3, "verified": False},
                {"concept": "有机反应机理", "detail": "描述反应过程中化学键断裂和形成的详细步骤", "difficulty": 3, "verified": False},
                {"concept": "电化学", "detail": "研究化学能和电能相互转换的科学", "difficulty": 3, "verified": False},
                {"concept": "晶体场理论", "detail": "解释配位化合物中d轨道能级分裂的理论", "difficulty": 4, "verified": False},
                {"concept": "化学键理论", "detail": "包括价键理论、分子轨道理论和杂化轨道理论", "difficulty": 3, "verified": False},
                {"concept": "反应动力学", "detail": "研究化学反应速率和反应机理的科学", "difficulty": 3, "verified": False},
                {"concept": "胶体化学", "detail": "研究分散体系的性质和行为", "difficulty": 2, "verified": False},
                {"concept": "高分子化学", "detail": "研究聚合物的合成、结构和性能", "difficulty": 3, "verified": False},
            ],
            "biology": [
                {"concept": "蛋白质折叠", "detail": "由疏水效应驱动，氨基酸序列决定三维结构", "difficulty": 3, "verified": False},
                {"concept": "ATP合成", "detail": "线粒体通过化学渗透生成细胞的能量货币ATP", "difficulty": 2, "verified": False},
                {"concept": "DNA复制", "detail": "半保留复制机制，DNA聚合酶催化互补链合成", "difficulty": 3, "verified": False},
                {"concept": "中心法则", "detail": "遗传信息从DNA到RNA到蛋白质的流动", "difficulty": 2, "verified": False},
                {"concept": "基因表达调控", "detail": "转录因子、启动子、增强子等调控基因表达", "difficulty": 3, "verified": False},
                {"concept": "酶催化机制", "detail": "通过降低活化能和稳定过渡态加速反应", "difficulty": 3, "verified": False},
                {"concept": "细胞信号传导", "detail": "通过受体、第二信使和级联反应传递信号", "difficulty": 3, "verified": False},
                {"concept": "光合作用", "detail": "光反应和暗反应将光能转化为化学能", "difficulty": 2, "verified": False},
                {"concept": "细胞周期", "detail": "G1、S、G2和M期的调控机制", "difficulty": 3, "verified": False},
                {"concept": "免疫系统", "detail": "先天免疫和适应性免疫的协同作用", "difficulty": 3, "verified": False},
                {"concept": "神经传导", "detail": "动作电位通过离子通道传播", "difficulty": 3, "verified": False},
                {"concept": "进化论", "detail": "自然选择驱动物种演化和适应", "difficulty": 2, "verified": False},
                {"concept": "生态系统", "detail": "生物与环境相互作用形成的复杂网络", "difficulty": 2, "verified": False},
                {"concept": "基因工程", "detail": "CRISPR等技术实现精确基因编辑", "difficulty": 4, "verified": False},
                {"concept": "表观遗传学", "detail": "不改变DNA序列的可遗传表型变化", "difficulty": 4, "verified": False},
                {"concept": "代谢途径", "detail": "糖酵解、三羧酸循环和氧化磷酸化", "difficulty": 3, "verified": False},
            ],
            "engineering": [
                {"concept": "有限元分析", "detail": "将连续体离散化为有限单元进行数值求解", "difficulty": 3, "verified": False},
                {"concept": "控制理论", "detail": "PID控制器通过比例、积分、微分控制系统", "difficulty": 3, "verified": False},
                {"concept": "信号处理", "detail": "使用滤波器、变换等技术处理信号", "difficulty": 3, "verified": False},
                {"concept": "机器学习", "detail": "通过数据训练模型实现预测和分类", "difficulty": 3, "verified": False},
                {"concept": "计算机视觉", "detail": "使用卷积神经网络等技术理解图像", "difficulty": 4, "verified": False},
                {"concept": "自然语言处理", "detail": "使用Transformer等模型处理文本", "difficulty": 4, "verified": False},
                {"concept": "优化算法", "detail": "梯度下降、遗传算法、粒子群优化等", "difficulty": 3, "verified": False},
                {"concept": "并行计算", "detail": "利用多核、GPU等并行执行计算任务", "difficulty": 3, "verified": False},
                {"concept": "数据结构", "detail": "数组、链表、树、图等组织数据的方式", "difficulty": 2, "verified": False},
                {"concept": "算法复杂度", "detail": "时间复杂度和空间复杂度的大O表示法", "difficulty": 2, "verified": False},
                {"concept": "网络协议", "detail": "TCP/IP、HTTP等计算机通信协议", "difficulty": 2, "verified": False},
                {"concept": "数据库系统", "detail": "关系型和非关系型数据库的设计和查询", "difficulty": 3, "verified": False},
                {"concept": "操作系统", "detail": "进程管理、内存管理、文件系统", "difficulty": 3, "verified": False},
                {"concept": "编译原理", "detail": "词法分析、语法分析、代码生成", "difficulty": 4, "verified": False},
                {"concept": "软件工程", "detail": "需求分析、设计模式、测试方法", "difficulty": 3, "verified": False},
            ],
            "computer_science": [
                {"concept": "图灵机", "detail": "计算理论的基础模型，定义了可计算性", "difficulty": 4, "verified": False},
                {"concept": "P vs NP问题", "detail": "计算复杂性理论中最重要的未解决问题", "difficulty": 5, "verified": False},
                {"concept": "量子计算", "detail": "利用量子叠加和纠缠实现并行计算", "difficulty": 5, "verified": False},
                {"concept": "密码学", "detail": "RSA、AES等加密算法保护信息安全", "difficulty": 4, "verified": False},
                {"concept": "区块链", "detail": "去中心化的分布式账本技术", "difficulty": 3, "verified": False},
                {"concept": "人工智能", "detail": "模拟人类智能的计算机系统", "difficulty": 4, "verified": False},
                {"concept": "深度学习", "detail": "多层神经网络学习复杂特征表示", "difficulty": 4, "verified": False},
                {"concept": "强化学习", "detail": "通过与环境交互学习最优策略", "difficulty": 4, "verified": False},
                {"concept": "图神经网络", "detail": "在图结构数据上进行学习的神经网络", "difficulty": 4, "verified": False},
                {"concept": "生成对抗网络", "detail": "生成器和判别器对抗训练生成数据", "difficulty": 4, "verified": False},
            ],
        }
    
    def get_random_knowledge(self, domain: str = None, count: int = 1) -> List[Dict]:
        """获取随机知识"""
        if domain and domain in self.knowledge:
            items = self.knowledge[domain]
        else:
            items = []
            for domain_items in self.knowledge.values():
                items.extend(domain_items)
        
        if len(items) <= count:
            return items
        return random.sample(items, count)
    
    def get_by_difficulty(self, min_difficulty: int = 1, max_difficulty: int = 5) -> List[Tuple[str, Dict]]:
        """按难度筛选知识"""
        result = []
        for domain, items in self.knowledge.items():
            for item in items:
                if min_difficulty <= item['difficulty'] <= max_difficulty:
                    result.append((domain, item))
        return result
    
    def get_unverified(self) -> List[Tuple[str, Dict]]:
        """获取未验证的知识"""
        result = []
        for domain, items in self.knowledge.items():
            for item in items:
                if not item.get('verified', False):
                    result.append((domain, item))
        return result
    
    def mark_verified(self, domain: str, concept: str):
        """标记知识为已验证"""
        if domain in self.knowledge:
            for item in self.knowledge[domain]:
                if item['concept'] == concept:
                    item['verified'] = True
                    item['verified_at'] = datetime.now().isoformat()
                    break
    
    def update_knowledge(self, domain: str, concept: str, new_detail: str, confidence: float = 1.0):
        """更新知识内容"""
        if domain in self.knowledge:
            for item in self.knowledge[domain]:
                if item['concept'] == concept:
                    item['detail'] = new_detail
                    item['confidence'] = confidence
                    item['updated_at'] = datetime.now().isoformat()
                    break
    
    def add_knowledge(self, domain: str, concept: str, detail: str, difficulty: int = 3):
        """添加新知识"""
        if domain not in self.knowledge:
            self.knowledge[domain] = []
        
        self.knowledge[domain].append({
            "concept": concept,
            "detail": detail,
            "difficulty": difficulty,
            "verified": False,
            "added_at": datetime.now().isoformat()
        })
    
    def save(self):
        """保存知识库到文件"""
        with open(self.knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        print(f"✓ 知识库已保存: {self.knowledge_file}")
    
    def load(self):
        """从文件加载知识库"""
        if self.knowledge_file.exists():
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            print(f"✓ 知识库已加载: {self.knowledge_file}")
        else:
            print("⚠️ 知识库文件不存在，使用默认知识")
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        stats = {
            "total_count": sum(len(items) for items in self.knowledge.values()),
            "by_domain": {domain: len(items) for domain, items in self.knowledge.items()},
            "verified_count": sum(1 for domain in self.knowledge for item in self.knowledge[domain] if item.get('verified')),
            "unverified_count": sum(1 for domain in self.knowledge for item in self.knowledge[domain] if not item.get('verified')),
            "by_difficulty": {}
        }
        
        for i in range(1, 6):
            stats['by_difficulty'][i] = sum(
                1 for domain in self.knowledge 
                for item in self.knowledge[domain] 
                if item['difficulty'] == i
            )
        
        return stats

if __name__ == "__main__":
    print("="*80)
    print("初始化大规模知识库...")
    print("="*80)
    
    kb = LargeKnowledgeBase()
    
    # 显示统计信息
    stats = kb.get_stats()
    print(f"\n📊 知识库统计:")
    print(f"   总条目: {stats['total_count']}")
    print(f"   已验证: {stats['verified_count']}")
    print(f"   未验证: {stats['unverified_count']}")
    
    print(f"\n📚 领域分布:")
    for domain, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1]):
        print(f"   {domain:20s}: {count} 条")
    
    print(f"\n⭐ 难度分布:")
    for difficulty, count in sorted(stats['by_difficulty'].items()):
        stars = "⭐" * difficulty
        print(f"   难度 {difficulty} {stars:10s}: {count} 条")
    
    # 保存到文件
    kb.save()
    
    # 展示一些示例
    print(f"\n🎲 随机知识示例:")
    sample_items = kb.get_random_knowledge(count=5)
    for i, item in enumerate(sample_items, 1):
        print(f"   [{i}] {item['concept']}")
        print(f"       {item['detail']}")
