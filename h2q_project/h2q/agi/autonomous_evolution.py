"""H2Q 自主进化系统 (Autonomous Evolution System).

实现基于兴趣驱动的完全自主学习进化:
1. 兴趣生成器 - 基于好奇心和知识缺口
2. 学习循环 - 获取、理解、整合知识
3. 能力评估 - 持续自我测试
4. 进化策略 - 自适应学习路径

安全机制:
- 资源限制 (内存、存储、网络)
- 合规检查 (仅公开资源)
- 可中断设计 (优雅停止)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator
from enum import Enum
import time
import json
import threading
from pathlib import Path
from collections import deque
import signal
import sys


# 导入依赖模块
try:
    from .fractal_memory_compression import (
        FractalMemoryDatabase, create_fractal_memory_db
    )
    from .knowledge_acquisition import (
        KnowledgeAcquisitionManager, KnowledgeResource,
        ResourceType, create_knowledge_acquisition_manager
    )
    from .multimodal_agi_core import MultimodalAGICore, create_multimodal_agi
except ImportError:
    # 开发模式
    pass


# ============================================================================
# 进化状态
# ============================================================================

class EvolutionState(Enum):
    """进化状态."""
    IDLE = "idle"
    EXPLORING = "exploring"      # 探索新知识
    LEARNING = "learning"        # 学习整合
    EVALUATING = "evaluating"    # 自我评估
    CONSOLIDATING = "consolidating"  # 记忆整合
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class Interest:
    """兴趣项."""
    topic: str
    curiosity_score: float  # 好奇心分数
    competence_score: float  # 当前能力
    knowledge_gap: float  # 知识缺口
    exploration_count: int = 0
    last_explored: float = 0.0
    
    @property
    def priority(self) -> float:
        """计算优先级 = 好奇心 × 知识缺口 / (探索次数 + 1)."""
        return (self.curiosity_score * self.knowledge_gap) / (self.exploration_count + 1)


@dataclass
class LearningEpisode:
    """学习回合."""
    id: int
    start_time: float
    end_time: Optional[float] = None
    resources_acquired: int = 0
    knowledge_integrated: int = 0
    skills_improved: List[str] = field(default_factory=list)
    evaluation_score: float = 0.0


@dataclass
class EvolutionConfig:
    """进化配置."""
    # 资源限制
    max_memory_mb: float = 100.0
    max_storage_mb: float = 500.0
    max_resources_per_cycle: int = 20
    
    # 时间控制
    cycle_duration_sec: float = 60.0  # 每个学习周期
    evaluation_interval: int = 5  # 每 N 个周期评估一次
    consolidation_interval: int = 10  # 每 N 个周期整合记忆
    
    # 学习参数
    curiosity_decay: float = 0.95  # 好奇心衰减
    learning_rate: float = 0.1
    exploration_bonus: float = 0.5
    
    # 存储路径
    state_path: Path = field(default_factory=lambda: Path("./evolution_state"))


# ============================================================================
# 兴趣生成器
# ============================================================================

class InterestGenerator:
    """兴趣生成器 - 基于好奇心和知识缺口."""
    
    # 基础兴趣领域
    BASE_DOMAINS = [
        # 数学
        ("linear algebra", ResourceType.MATH),
        ("calculus", ResourceType.MATH),
        ("number theory", ResourceType.MATH),
        ("geometry", ResourceType.MATH),
        ("probability theory", ResourceType.MATH),
        
        # 人工智能
        ("neural networks", ResourceType.SCIENCE),
        ("deep learning", ResourceType.SCIENCE),
        ("reinforcement learning", ResourceType.SCIENCE),
        ("natural language processing", ResourceType.SCIENCE),
        ("computer vision", ResourceType.SCIENCE),
        
        # 科学
        ("physics fundamentals", ResourceType.SCIENCE),
        ("chemistry basics", ResourceType.SCIENCE),
        ("biology concepts", ResourceType.SCIENCE),
        
        # 常识
        ("world history", ResourceType.GENERAL),
        ("geography", ResourceType.GENERAL),
        ("philosophy", ResourceType.GENERAL),
    ]
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.interests: Dict[str, Interest] = {}
        self.competence_map: Dict[str, float] = {}
        
        # 初始化基础兴趣
        self._init_base_interests()
    
    def _init_base_interests(self):
        """初始化基础兴趣."""
        for topic, _ in self.BASE_DOMAINS:
            self.interests[topic] = Interest(
                topic=topic,
                curiosity_score=np.random.uniform(0.5, 1.0),
                competence_score=0.1,
                knowledge_gap=0.9,
            )
    
    def generate_interest(self) -> Optional[Interest]:
        """生成下一个兴趣点."""
        if not self.interests:
            return None
        
        # 按优先级排序
        sorted_interests = sorted(
            self.interests.values(),
            key=lambda x: x.priority,
            reverse=True
        )
        
        # 选择最高优先级
        return sorted_interests[0]
    
    def update_after_learning(self, topic: str, success: bool, 
                               knowledge_gained: float = 0.0):
        """学习后更新兴趣状态."""
        if topic not in self.interests:
            return
        
        interest = self.interests[topic]
        interest.exploration_count += 1
        interest.last_explored = time.time()
        
        if success:
            # 增加能力，减少知识缺口
            interest.competence_score = min(1.0, 
                interest.competence_score + knowledge_gained * self.config.learning_rate
            )
            interest.knowledge_gap = max(0.1, 
                1.0 - interest.competence_score
            )
            
            # 好奇心衰减
            interest.curiosity_score *= self.config.curiosity_decay
        else:
            # 失败时增加好奇心
            interest.curiosity_score = min(1.0,
                interest.curiosity_score + self.config.exploration_bonus * 0.5
            )
    
    def discover_new_interest(self, topic: str, related_to: str = None):
        """发现新兴趣."""
        if topic in self.interests:
            return
        
        # 基于相关主题设置初始好奇心
        base_curiosity = 0.7
        if related_to and related_to in self.interests:
            base_curiosity = self.interests[related_to].curiosity_score * 0.8
        
        self.interests[topic] = Interest(
            topic=topic,
            curiosity_score=base_curiosity + np.random.uniform(0, 0.3),
            competence_score=0.05,
            knowledge_gap=0.95,
        )
    
    def get_top_interests(self, n: int = 5) -> List[Interest]:
        """获取前 N 个兴趣."""
        return sorted(
            self.interests.values(),
            key=lambda x: x.priority,
            reverse=True
        )[:n]


# ============================================================================
# 自主进化引擎
# ============================================================================

class AutonomousEvolutionEngine:
    """自主进化引擎 - 核心进化循环."""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.config.state_path.mkdir(parents=True, exist_ok=True)
        
        # 状态
        self.state = EvolutionState.IDLE
        self.current_cycle = 0
        self.total_episodes = 0
        
        # 组件
        self.interest_generator = InterestGenerator(self.config)
        
        # 记忆数据库
        self.memory_db = create_fractal_memory_db(
            max_memory_mb=self.config.max_memory_mb,
            storage_path=str(self.config.state_path / "memory")
        )
        
        # 知识获取器
        self.knowledge_manager = create_knowledge_acquisition_manager(
            cache_dir=str(self.config.state_path / "knowledge")
        )
        
        # AGI 核心
        self.agi_core = create_multimodal_agi()
        
        # 学习历史
        self.episodes: List[LearningEpisode] = []
        self.current_episode: Optional[LearningEpisode] = None
        
        # 控制
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        
        # 统计
        self.stats = {
            "total_cycles": 0,
            "total_resources_learned": 0,
            "total_knowledge_integrated": 0,
            "avg_evaluation_score": 0.0,
            "memory_compressions": 0,
        }
        
        # 加载保存的状态
        self._load_state()
    
    def start(self, max_cycles: int = None):
        """启动进化循环."""
        self.state = EvolutionState.EXPLORING
        self._stop_flag.clear()
        self._pause_flag.clear()
        
        print(f"[进化引擎] 启动自主进化...")
        print(f"  配置: 周期={self.config.cycle_duration_sec}s, "
              f"内存限制={self.config.max_memory_mb}MB")
        
        cycle = 0
        while not self._stop_flag.is_set():
            # 检查最大周期
            if max_cycles and cycle >= max_cycles:
                print(f"\n[进化引擎] 达到最大周期数 {max_cycles}")
                break
            
            # 暂停检查
            if self._pause_flag.is_set():
                self.state = EvolutionState.PAUSED
                time.sleep(1)
                continue
            
            # 执行一个进化周期
            self._evolution_cycle()
            
            cycle += 1
            self.current_cycle = cycle
            
            # 周期间隔
            time.sleep(0.1)  # 简短延迟
        
        self.state = EvolutionState.STOPPED
        self._save_state()
        print(f"\n[进化引擎] 进化停止，共完成 {cycle} 个周期")
    
    def stop(self):
        """停止进化."""
        self._stop_flag.set()
    
    def pause(self):
        """暂停进化."""
        self._pause_flag.set()
    
    def resume(self):
        """恢复进化."""
        self._pause_flag.clear()
    
    def _evolution_cycle(self):
        """执行一个进化周期."""
        cycle_start = time.time()
        
        # 1. 探索阶段 - 选择兴趣并获取知识
        self.state = EvolutionState.EXPLORING
        interest = self.interest_generator.generate_interest()
        
        if interest:
            print(f"\n[周期 {self.current_cycle + 1}] 探索: {interest.topic} "
                  f"(优先级={interest.priority:.2f})")
            
            # 添加到知识获取队列
            self.knowledge_manager.add_interest(interest.topic)
        
        # 2. 学习阶段 - 获取并整合知识
        self.state = EvolutionState.LEARNING
        resources_learned = 0
        
        for resource in self.knowledge_manager.poll_resources():
            # 处理资源
            success = self._learn_resource(resource)
            
            if success:
                resources_learned += 1
                self.stats["total_resources_learned"] += 1
            
            # 限制每周期资源数
            if resources_learned >= self.config.max_resources_per_cycle:
                break
        
        # 更新兴趣状态
        if interest:
            self.interest_generator.update_after_learning(
                interest.topic,
                success=resources_learned > 0,
                knowledge_gained=resources_learned * 0.1
            )
        
        print(f"  学习: {resources_learned} 个资源")
        
        # 3. 评估阶段 (定期)
        if (self.current_cycle + 1) % self.config.evaluation_interval == 0:
            self.state = EvolutionState.EVALUATING
            eval_score = self._self_evaluate()
            print(f"  评估: {eval_score:.1f}%")
        
        # 4. 整合阶段 (定期)
        if (self.current_cycle + 1) % self.config.consolidation_interval == 0:
            self.state = EvolutionState.CONSOLIDATING
            self._consolidate_memory()
        
        # 更新统计
        self.stats["total_cycles"] += 1
        
        # 记录周期时间
        cycle_time = time.time() - cycle_start
        print(f"  用时: {cycle_time:.2f}s")
    
    def _learn_resource(self, resource: KnowledgeResource) -> bool:
        """学习单个资源."""
        try:
            # 提取知识
            knowledge = self._extract_knowledge(resource)
            
            if not knowledge:
                return False
            
            # 存储到分形记忆
            block_id = self.memory_db.store(
                key=resource.id,
                data=knowledge,
                metadata={
                    "title": resource.title,
                    "source": resource.source.value,
                    "type": resource.resource_type.value,
                },
                importance=resource.quality_score
            )
            
            self.stats["total_knowledge_integrated"] += 1
            
            # 发现新兴趣
            new_topics = self._extract_topics(resource.content)
            for topic in new_topics[:3]:
                self.interest_generator.discover_new_interest(
                    topic, related_to=resource.title
                )
            
            return True
            
        except Exception as e:
            print(f"    学习失败: {e}")
            return False
    
    def _extract_knowledge(self, resource: KnowledgeResource) -> Optional[np.ndarray]:
        """从资源提取知识向量."""
        content = resource.content
        
        if not content:
            return None
        
        # 简单的 BOW 向量化
        words = content.lower().split()
        
        # 构建特征
        features = []
        
        # 词频特征
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        # 取 top 词
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
        features.extend([count for _, count in top_words])
        
        # 填充到固定长度
        while len(features) < 100:
            features.append(0)
        
        return np.array(features[:100], dtype=np.float32)
    
    def _extract_topics(self, content: str) -> List[str]:
        """从内容提取潜在主题."""
        import re
        
        # 简单的关键词提取
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # 统计词频
        word_counts = {}
        for w in words:
            if w not in {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they'}:
                word_counts[w] = word_counts.get(w, 0) + 1
        
        # 返回高频词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]
    
    def _self_evaluate(self) -> float:
        """自我评估."""
        # 使用 AGI 核心的数学推理进行测试
        correct = 0
        total = 10
        
        for _ in range(total):
            a = np.random.randint(1, 50)
            b = np.random.randint(1, 50)
            op = np.random.choice(['+', '-', '*'])
            
            pred, gt, error = self.agi_core.solve_math(a, b, op)
            
            if error < 0.5:
                correct += 1
        
        score = (correct / total) * 100
        
        # 更新平均分
        n = self.stats["total_cycles"]
        self.stats["avg_evaluation_score"] = (
            (self.stats["avg_evaluation_score"] * (n - 1) + score) / n
            if n > 0 else score
        )
        
        return score
    
    def _consolidate_memory(self):
        """整合记忆 - 压缩低优先级记忆."""
        usage = self.memory_db.get_memory_usage()
        
        if usage["utilization"] > 0.7:
            freed = self.memory_db.compress_memory(target_ratio=0.5)
            self.stats["memory_compressions"] += 1
            print(f"  记忆整合: 释放 {freed / 1024:.1f} KB")
        
        # 遗忘旧记忆
        forgotten = self.memory_db.forget(threshold=0.1)
        if forgotten > 0:
            print(f"  遗忘: {forgotten} 个低优先级记忆")
    
    def _save_state(self):
        """保存进化状态."""
        state_file = self.config.state_path / "evolution_state.json"
        
        # 保存兴趣
        interests_data = {
            topic: {
                "topic": i.topic,
                "curiosity_score": i.curiosity_score,
                "competence_score": i.competence_score,
                "knowledge_gap": i.knowledge_gap,
                "exploration_count": i.exploration_count,
            }
            for topic, i in self.interest_generator.interests.items()
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({
                "current_cycle": self.current_cycle,
                "total_episodes": self.total_episodes,
                "stats": self.stats,
                "interests": interests_data,
            }, f, indent=2)
    
    def _load_state(self):
        """加载进化状态."""
        state_file = self.config.state_path / "evolution_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_cycle = data.get("current_cycle", 0)
            self.total_episodes = data.get("total_episodes", 0)
            self.stats = data.get("stats", self.stats)
            
            # 恢复兴趣
            for topic, i_data in data.get("interests", {}).items():
                self.interest_generator.interests[topic] = Interest(
                    topic=i_data["topic"],
                    curiosity_score=i_data["curiosity_score"],
                    competence_score=i_data["competence_score"],
                    knowledge_gap=i_data["knowledge_gap"],
                    exploration_count=i_data["exploration_count"],
                )
            
            print(f"[进化引擎] 恢复状态: 周期 {self.current_cycle}")
            
        except Exception as e:
            print(f"[进化引擎] 加载状态失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态."""
        memory_usage = self.memory_db.get_memory_usage()
        knowledge_stats = self.knowledge_manager.get_stats()
        
        return {
            "state": self.state.value,
            "current_cycle": self.current_cycle,
            "stats": self.stats,
            "memory": memory_usage,
            "knowledge": knowledge_stats,
            "top_interests": [
                {"topic": i.topic, "priority": i.priority}
                for i in self.interest_generator.get_top_interests(5)
            ],
        }
    
    def generate_report(self) -> str:
        """生成进化报告."""
        status = self.get_status()
        
        report = []
        report.append("=" * 60)
        report.append("H2Q 自主进化系统 - 状态报告")
        report.append("=" * 60)
        
        report.append(f"\n状态: {status['state']}")
        report.append(f"当前周期: {status['current_cycle']}")
        
        report.append("\n统计:")
        for k, v in status['stats'].items():
            report.append(f"  {k}: {v}")
        
        report.append("\n内存使用:")
        for k, v in status['memory'].items():
            if isinstance(v, float):
                report.append(f"  {k}: {v:.2f}")
            else:
                report.append(f"  {k}: {v}")
        
        report.append("\nTop 兴趣:")
        for i in status['top_interests']:
            report.append(f"  {i['topic']}: {i['priority']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# ============================================================================
# 工厂函数
# ============================================================================

def create_evolution_engine(
    max_memory_mb: float = 100.0,
    state_path: str = None
) -> AutonomousEvolutionEngine:
    """创建自主进化引擎."""
    config = EvolutionConfig(
        max_memory_mb=max_memory_mb,
        state_path=Path(state_path) if state_path else Path("./evolution_state"),
    )
    return AutonomousEvolutionEngine(config)


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H2Q 自主进化系统 - 演示")
    print("=" * 60)
    
    # 创建引擎
    engine = create_evolution_engine(
        max_memory_mb=50.0,
        state_path="./test_evolution_state"
    )
    
    # 运行有限周期
    print("\n启动自主进化 (5 个周期)...")
    engine.start(max_cycles=5)
    
    # 生成报告
    print(engine.generate_report())
