#!/usr/bin/env python3
"""
AGI多模态全能力训练系统 - 集成Gemini 2.5 Flash

功能特性：
1. 结合本地AGI自主训练的所有功能
2. 集成Gemini 2.5 Flash API进行知识扩展
3. 增强的缓存机制和速率控制
4. 多模态学习能力（文本、代码、数学等）
5. 持续学习和进化
6. 智能API调用管理
"""

import os
import sys
import json
import time
import logging
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict
import hashlib
import pickle
from functools import lru_cache

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini API不可用，将使用本地知识扩展")

from optimized_agi_autonomous_system import OptimizedAutonomousAGI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MULTIMODAL-AGI] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('multimodal_agi_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MULTIMODAL-AGI')

class EnhancedGeminiKnowledgeExpander:
    """增强的Gemini知识扩展器 - 智能缓存和速率控制"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash"  # 使用指定的模型
        self.client = None

        # 增强的速率限制控制
        self.call_history = deque(maxlen=60)  # 记录最近60次调用
        self.max_calls_per_minute = 8  # 降低到每分钟8次调用，更保守
        self.last_call_time = 0
        self.min_interval = 8.0  # 增加到8秒间隔
        self.burst_limit = 3  # 突发限制
        self.burst_window = 30  # 30秒窗口

        # 多层缓存系统
        self.memory_cache = {}  # 内存缓存
        self.disk_cache_dir = Path("./gemini_cache")
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = 3600  # 缓存1小时

        # 统计信息
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'last_reset': time.time()
        }

        # 知识领域映射
        self.domain_experts = {
            'mathematics': '数学专家',
            'computer_science': '计算机科学专家',
            'physics': '物理学专家',
            'philosophy': '哲学专家',
            'artificial_intelligence': '人工智能专家',
            'deepseek_technologies': 'DeepSeek技术专家'
        }

        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("✅ Gemini 2.5 Flash API客户端初始化成功")
            except Exception as e:
                logger.warning(f"❌ Gemini API初始化失败: {e}")
                self.client = None
        else:
            logger.warning("⚠️  Gemini API未配置，使用本地知识扩展模式")

    def _get_cache_key(self, topic: str, context: Dict[str, Any]) -> str:
        """生成缓存键"""
        content = f"{topic}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_disk_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从磁盘缓存加载"""
        cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)
                    if time.time() - timestamp < self.cache_expiry:
                        return data
                    else:
                        # 过期删除
                        cache_file.unlink()
            except Exception as e:
                logger.warning(f"缓存文件读取失败: {e}")
        return None

    def _save_to_disk_cache(self, cache_key: str, data: Dict[str, Any]):
        """保存到磁盘缓存"""
        cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((data, time.time()), f)
        except Exception as e:
            logger.warning(f"缓存文件保存失败: {e}")

    def _check_burst_limit(self) -> bool:
        """检查突发限制"""
        current_time = time.time()
        recent_calls = [t for t in self.call_history if current_time - t < self.burst_window]
        return len(recent_calls) < self.burst_limit

    def _check_rate_limit(self) -> bool:
        """检查是否超过速率限制"""
        current_time = time.time()

        # 清理过期记录
        while self.call_history and current_time - self.call_history[0] > 60:
            self.call_history.popleft()

        # 检查间隔限制
        if current_time - self.last_call_time < self.min_interval:
            return False

        # 检查突发限制
        if not self._check_burst_limit():
            return False

        # 检查每分钟限制
        if len(self.call_history) >= self.max_calls_per_minute:
            return False

        return True

    def _record_call(self):
        """记录API调用"""
        current_time = time.time()
        self.call_history.append(current_time)
        self.last_call_time = current_time
        self.stats['api_calls'] += 1

    def _wait_for_rate_limit(self) -> float:
        """等待直到可以进行API调用，返回等待时间"""
        while not self._check_rate_limit():
            time.sleep(1.0)
        return 0.0

    async def expand_knowledge(self, topic: str, current_knowledge: Dict[str, Any],
                              modality: str = "text") -> Dict[str, Any]:
        """
        使用Gemini 2.5 Flash扩展知识网络

        Args:
            topic: 要扩展的主题
            current_knowledge: 当前已有的知识
            modality: 模态类型 (text, code, math, etc.)

        Returns:
            扩展后的知识字典
        """
        if not self.client:
            logger.info(f"📚 使用本地知识扩展模式: {topic}")
            return self._local_knowledge_expansion(topic, current_knowledge, modality)

        # 生成缓存键
        cache_key = self._get_cache_key(topic, current_knowledge)

        # 检查内存缓存
        if cache_key in self.memory_cache:
            self.stats['cache_hits'] += 1
            logger.info(f"💾 内存缓存命中: {topic}")
            return self.memory_cache[cache_key]

        # 检查磁盘缓存
        cached_data = self._load_from_disk_cache(cache_key)
        if cached_data:
            self.stats['cache_hits'] += 1
            self.memory_cache[cache_key] = cached_data  # 加载到内存缓存
            logger.info(f"💾 磁盘缓存命中: {topic}")
            return cached_data

        self.stats['cache_misses'] += 1

        # 等待速率限制
        wait_time = self._wait_for_rate_limit()
        if wait_time > 0:
            logger.info(f"⏳ 等待速率限制: {wait_time:.1f}秒")

        try:
            # 构建专家提示
            expert_role = self.domain_experts.get(topic.lower(), "知识专家")

            prompt = f"""你是一位{expert_role}，请基于以下当前知识，扩展关于"{topic}"的知识网络。

当前知识状态：
{json.dumps(current_knowledge, ensure_ascii=False, indent=2)}

请从以下{modality}模态角度提供扩展知识：

1. 核心概念深化
2. 实际应用案例
3. 相关技术连接
4. 研究发展趋势
5. 技术挑战与解决方案
6. 学习路径建议
7. 相关主题推荐

请提供结构化的JSON响应，包含上述所有方面。"""

            # 调用Gemini 2.5 Flash API
            self._record_call()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )

            # 解析响应
            if response and response.text:
                try:
                    expanded_knowledge = json.loads(response.text.strip())
                    logger.info(f"✅ Gemini API扩展成功: {topic}")

                    # 缓存结果
                    self.memory_cache[cache_key] = expanded_knowledge
                    self._save_to_disk_cache(cache_key, expanded_knowledge)

                    return expanded_knowledge

                except json.JSONDecodeError as e:
                    logger.warning(f"❌ JSON解析失败: {e}")
                    self.stats['errors'] += 1
                    return self._local_knowledge_expansion(topic, current_knowledge, modality)
            else:
                logger.warning("❌ Gemini API无响应")
                self.stats['errors'] += 1
                return self._local_knowledge_expansion(topic, current_knowledge, modality)

        except Exception as e:
            logger.error(f"❌ Gemini API调用失败: {e}")
            self.stats['errors'] += 1
            return self._local_knowledge_expansion(topic, current_knowledge, modality)

    def _local_knowledge_expansion(self, topic: str, current_knowledge: Dict[str, Any],
                                  modality: str = "text") -> Dict[str, Any]:
        """本地知识扩展（当API不可用时使用）"""
        logger.info(f"📚 使用本地知识扩展{topic} ({modality}模态)")

        # 基于主题和模态的本地扩展逻辑
        base_expansion = {
            "核心概念深化": f"{topic}是{modality}领域的重要概念",
            "实际应用案例": f"{topic}在{modality}处理中有广泛应用",
            "相关技术连接": f"{topic}与其他{modality}技术密切相关",
            "研究发展趋势": f"{topic}在{modality}领域正在快速发展",
            "技术挑战与解决方案": f"{topic}面临{modality}处理的挑战",
            "学习路径建议": f"建议系统性学习{topic}在{modality}方面的知识",
            "相关主题推荐": [f"{modality}处理", "AI技术", "机器学习"]
        }

        # 模态特定的扩展
        if modality == "code":
            base_expansion.update({
                "核心概念深化": f"{topic}涉及代码生成、分析和优化技术",
                "实际应用案例": "代码补全、bug检测、重构等",
                "相关技术连接": "编译器技术、静态分析、AST处理",
                "研究发展趋势": "朝着多语言支持、大模型集成方向发展"
            })
        elif modality == "math":
            base_expansion.update({
                "核心概念深化": f"{topic}包含数学推理和证明技术",
                "实际应用案例": "定理证明、数学问题求解、公式推导",
                "相关技术连接": "符号计算、逻辑推理、数学建模",
                "研究发展趋势": "朝着自动化数学发现方向发展"
            })

        return base_expansion

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        if current_time - self.stats['last_reset'] > 3600:  # 每小时重置
            self.stats.update({
                'api_calls': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'last_reset': current_time
            })

        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_queue_size': len(self.call_history),
            'memory_cache_size': len(self.memory_cache)
        }

class MultimodalAGITrainer:
    """多模态AGI训练器 - 集成所有功能"""

    def __init__(self):
        self.agi_system = None
        self.knowledge_expander = EnhancedGeminiKnowledgeExpander()

        # 多模态支持
        self.modalities = ["text", "code", "math", "reasoning", "technical"]
        self.modality_weights = {mod: 1.0 for mod in self.modalities}

        # 训练统计
        self.training_stats = {
            'start_time': datetime.now(),
            'total_steps': 0,
            'knowledge_expansions': 0,
            'api_calls': 0,
            'learning_metrics': [],
            'modality_usage': {mod: 0 for mod in self.modalities}
        }

        # 智能调度
        self.expansion_interval = 30  # 每30步进行一次知识扩展
        self.last_expansion_step = 0
        self.adaptive_expansion = True  # 自适应扩展频率

        # 性能监控
        self.performance_history = deque(maxlen=100)

        logger.info("🚀 多模态AGI训练器初始化完成")

    def initialize_system(self):
        """初始化AGI系统"""
        logger.info("🔧 初始化多模态AGI系统...")

        # 加载学习资料
        learning_materials = self._load_learning_materials()

        # 创建AGI系统
        self.agi_system = OptimizedAutonomousAGI(
            input_dim=256,
            action_dim=64,
            learning_materials=learning_materials
        )

        logger.info("✅ 多模态AGI系统初始化完成")

    def _load_learning_materials(self) -> Dict[str, Any]:
        """加载学习资料"""
        try:
            with open('agi_learning_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"📚 已加载学习资料：{len(data.get('domains', []))}个领域")
                return data
        except Exception as e:
            logger.warning(f"❌ 学习资料加载失败: {e}")
            return {"domains": [], "learning_tasks": []}

    def _select_modality(self) -> str:
        """智能选择模态"""
        # 基于当前系统状态和学习需求选择模态
        if not hasattr(self.agi_system, 'consciousness_engine'):
            return "text"

        consciousness = self._get_consciousness_level() if hasattr(self.agi_system, 'consciousness_engine') else {}

        # 根据意识水平选择模态
        if consciousness['integrated_information'] > 0.5:
            # 高意识水平，适合复杂模态
            weights = {"reasoning": 0.3, "technical": 0.3, "code": 0.2, "math": 0.1, "text": 0.1}
        elif consciousness['metacognitive_awareness'] > 0.4:
            # 较高元认知，适合技术模态
            weights = {"technical": 0.4, "code": 0.3, "reasoning": 0.2, "text": 0.1, "math": 0.0}
        else:
            # 基础水平，从文本开始
            weights = {"text": 0.5, "reasoning": 0.3, "technical": 0.1, "code": 0.05, "math": 0.05}

        # 归一化权重
        total = sum(weights.values())
        normalized_weights = {k: v/total for k, v in weights.items()}

        # 按权重选择
        modalities = list(normalized_weights.keys())
        weights_list = list(normalized_weights.values())

        selected = np.random.choice(modalities, p=weights_list)
        self.training_stats['modality_usage'][selected] += 1

        return selected

    def _perform_knowledge_expansion_sync(self, step: int):
        """同步执行知识扩展（用于非异步上下文）"""
        try:
            # 使用线程池执行器来避免事件循环冲突
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_expansion, step)
                future.result(timeout=30)  # 30秒超时
        except Exception as e:
            logger.warning(f"知识扩展失败: {e}")

    def _run_async_expansion(self, step: int):
        """在新的线程中运行异步知识扩展"""
        try:
            # 创建新的异步环境
            import nest_asyncio
            nest_asyncio.apply()

            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 运行异步任务
            loop.run_until_complete(self._perform_knowledge_expansion(step))
            loop.close()
        except Exception as e:
            logger.warning(f"异步知识扩展失败: {e}")

    async def _perform_knowledge_expansion(self, step: int):
        """执行知识扩展"""
        if step - self.last_expansion_step < self.expansion_interval:
            return

        # 选择要扩展的主题
        available_topics = []
        if hasattr(self.agi_system, 'learning_materials'):
            for domain in self.agi_system.learning_materials.get('domains', []):
                available_topics.extend(domain.get('topics', []))

        if not available_topics:
            available_topics = ["artificial_intelligence", "machine_learning", "deepseek_technologies"]

        # 选择当前最相关的主题
        current_goals = []
        if hasattr(self.agi_system, 'goal_system') and self.agi_system.goal_system:
            active_goals = getattr(self.agi_system.goal_system, 'active_goals', [])
            for goal in active_goals:
                if isinstance(goal, dict):
                    # 如果goal是字典，获取description字段
                    current_goals.append(goal.get('description', str(goal)))
                elif hasattr(goal, 'description'):
                    # 如果goal是对象，获取description属性
                    current_goals.append(goal.description)
                else:
                    # 其他情况，转为字符串
                    current_goals.append(str(goal))

        # 简单的相关性匹配
        topic_scores = {}
        for topic in available_topics:
            score = 0
            for goal in current_goals:
                if topic.lower() in goal.lower():
                    score += 1
            topic_scores[topic] = score

        # 选择最高分的主题，如果都没有则随机选择
        if topic_scores:
            selected_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        else:
            selected_topic = np.random.choice(available_topics)

    def _get_consciousness_level(self) -> Dict[str, float]:
        """获取当前意识水平"""
        if not hasattr(self.agi_system, 'consciousness_engine') or not self.agi_system.consciousness_engine:
            return {}

        try:
            # 调用forward方法获取意识指标
            consciousness_metrics, _ = self.agi_system.consciousness_engine(self.agi_system.current_state)
            return {
                'integrated_information': consciousness_metrics.integrated_information,
                'neural_complexity': consciousness_metrics.neural_complexity,
                'self_model_accuracy': consciousness_metrics.self_model_accuracy,
                'metacognitive_awareness': consciousness_metrics.metacognitive_awareness,
                'emotional_valence': consciousness_metrics.emotional_valence,
                'temporal_binding': consciousness_metrics.temporal_binding
            }
        except Exception as e:
            logger.warning(f"获取意识水平失败: {e}")
            return {}

    def _select_modality(self) -> str:
        """获取当前知识状态"""
        # 从AGI系统的学习资料中提取相关知识
        current_state = {}

        if hasattr(self.agi_system, 'learning_materials'):
            materials = self.agi_system.learning_materials
            for domain in materials.get('domains', []):
                if topic.lower() in domain.get('name', '').lower():
                    current_state.update({
                        'existing_concepts': domain.get('concepts', []),
                        'current_level': domain.get('difficulty', 'beginner'),
                        'learned_topics': domain.get('topics', [])
                    })
                    break

        return current_state

    def _integrate_expanded_knowledge(self, topic: str, expanded_knowledge: Dict[str, Any], modality: str):
        """整合扩展的知识到AGI系统"""
        try:
            # 更新学习资料
            if hasattr(self.agi_system, 'learning_materials'):
                materials = self.agi_system.learning_materials

                # 查找或创建领域
                domain_found = False
                for domain in materials.get('domains', []):
                    if topic.lower() in domain.get('name', '').lower():
                        # 更新现有领域
                        domain.setdefault('expanded_knowledge', {}).update({
                            modality: expanded_knowledge
                        })
                        domain_found = True
                        break

                if not domain_found:
                    # 创建新领域
                    new_domain = {
                        'name': topic,
                        'topics': [topic],
                        'concepts': list(expanded_knowledge.keys()),
                        'difficulty': 'intermediate',
                        'expanded_knowledge': {modality: expanded_knowledge}
                    }
                    materials['domains'].append(new_domain)

                # 保存更新
                with open('agi_learning_data_expanded.json', 'w', encoding='utf-8') as f:
                    json.dump(materials, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 知识整合完成: {topic} ({modality})")

        except Exception as e:
            logger.error(f"❌ 知识整合失败: {e}")

    async def run_training_loop(self, max_steps: int = 1000):
        """运行训练循环"""
        logger.info(f"🏃 开始多模态AGI训练，目标步数：{max_steps}")

        try:
            for step in range(max_steps):
                self.training_stats['total_steps'] = step + 1

                # 执行一步训练
                if self.agi_system:
                    step_result = self.agi_system.step()
                    # 记录步骤结果
                    if step_result:
                        self.performance_history.append(step_result)

                # 定期执行知识扩展
                self._perform_knowledge_expansion_sync(step)

                # 记录学习指标
                if self.agi_system and hasattr(self.agi_system, 'get_learning_metrics'):
                    metrics = self.agi_system.get_learning_metrics()
                    self.training_stats['learning_metrics'].append(metrics)

                # 保存训练状态
                if step % 50 == 0:
                    self._save_training_state()

                # 显示进度
                if step % 10 == 0:
                    self._log_progress(step)

                # 小延迟避免过度占用CPU
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("⏹️ 训练被用户中断")
        except Exception as e:
            logger.error(f"❌ 训练过程中出错: {e}")
        finally:
            # 生成最终报告
            self._generate_final_report()

    def _log_progress(self, step: int):
        """记录训练进度"""
        expander_stats = self.knowledge_expander.get_stats()

        progress_info = {
            'step': step + 1,
            'expansions': self.training_stats['knowledge_expansions'],
            'api_calls': expander_stats['api_calls'],
            'cache_hit_rate': f"{expander_stats['hit_rate']:.2%}",
            'modality_usage': self.training_stats['modality_usage']
        }

        logger.info(f"📊 步骤 {step + 1}: {progress_info}")

    def _save_training_state(self):
        """保存训练状态"""
        state = {
            'training_stats': self.training_stats,
            'system_status': self._get_system_status(),
            'timestamp': datetime.now().isoformat()
        }

        with open('multimodal_agi_training_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

        logger.info("💾 训练状态已保存")

    def _get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.agi_system:
            return {}

        try:
            return {
                'step_count': getattr(self.agi_system, 'step_count', 0),
                'runtime': getattr(self.agi_system, 'runtime', 0),
                'consciousness_level': self._get_consciousness_level(),
                'goal_status': self.agi_system.goal_system.get_status() if hasattr(self.agi_system, 'goal_system') else {},
                'learning_status': self.agi_system.learning_engine.get_status() if hasattr(self.agi_system, 'learning_engine') else {}
            }
        except Exception as e:
            logger.warning(f"获取系统状态失败: {e}")
            return {}

    def _generate_final_report(self):
        """生成最终训练报告"""
        report = {
            'training_duration': str(datetime.now() - self.training_stats['start_time']),
            'total_steps': self.training_stats['total_steps'],
            'knowledge_expansions': self.training_stats['knowledge_expansions'],
            'modality_distribution': self.training_stats['modality_usage'],
            'expander_stats': self.knowledge_expander.get_stats(),
            'final_system_status': self._get_system_status(),
            'completion_time': datetime.now().isoformat()
        }

        with open('multimodal_agi_training_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info("📋 最终训练报告已生成")

async def main():
    """主函数"""
    logger.info("🚀 AGI多模态全能力训练系统启动")
    logger.info("=" * 50)

    # 创建训练器
    trainer = MultimodalAGITrainer()

    # 初始化系统
    trainer.initialize_system()

    # 运行训练
    await trainer.run_training_loop(max_steps=500)

    logger.info("=" * 50)
    logger.info("🎯 AGI多模态全能力训练系统结束")

if __name__ == "__main__":
    asyncio.run(main())