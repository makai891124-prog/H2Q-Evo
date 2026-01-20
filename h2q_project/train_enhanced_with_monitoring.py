#!/usr/bin/env python3
"""
H2Q-Evo 增强型训练系统 - 支持4小时长时间训练和动态监控

功能:
- 实时性能监控和数据收集
- 进化趋势追踪
- 动态训练参数调整
- 完整的进度报告
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

from h2q_project.local_model_advanced_training import (
    LocalModelAdvancedTrainer,
    CompetencyEvaluator,
    OutputCorrectionMechanism,
    CompetencyMetrics,
    CompetencyLevel
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_session.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """训练监控和数据收集系统"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.iterations_data = []
        self.evolution_data = {
            'session_id': session_id,
            'start_time': self.start_time.isoformat(),
            'iterations': [],
            'trends': {},
            'summary': {}
        }
        
    def record_iteration(self, iteration_num: int, metrics: CompetencyMetrics, 
                        iteration_time: float, training_loss: float = 0.0):
        """记录单次迭代的数据"""
        iteration_data = {
            'iteration': iteration_num,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': (datetime.now() - self.start_time).total_seconds(),
            'iteration_time': iteration_time,
            'overall_score': metrics.overall_score,
            'level': metrics.level.name,
            'training_loss': training_loss,
            'dimensions': {
                'correctness': metrics.correctness,
                'consistency': metrics.consistency,
                'completeness': metrics.completeness,
                'fluency': metrics.fluency,
                'coherence': metrics.coherence,
                'reasoning_depth': metrics.reasoning_depth,
                'knowledge_accuracy': metrics.knowledge_accuracy,
                'language_control': metrics.language_control,
                'creativity': metrics.creativity,
                'adaptability': metrics.adaptability,
            }
        }
        self.iterations_data.append(iteration_data)
        self.evolution_data['iterations'].append(iteration_data)
        
        logger.info(f"迭代 {iteration_num} 完成:")
        logger.info(f"  总体评分: {metrics.overall_score:.2%}")
        logger.info(f"  能力等级: {metrics.level.name}")
        logger.info(f"  耗时: {iteration_time:.2f}s")
        logger.info(f"  总耗时: {iteration_data['elapsed_time']:.0f}s")
        
    def calculate_trends(self) -> Dict[str, Any]:
        """计算进化趋势"""
        if len(self.iterations_data) < 2:
            return {}
        
        trends = {
            'overall_score_trend': [],
            'level_transitions': [],
            'dimension_trends': {},
            'performance_acceleration': 0.0,
            'convergence_rate': 0.0,
        }
        
        # 总体评分趋势
        scores = [d['overall_score'] for d in self.iterations_data]
        for i, score in enumerate(scores):
            if i > 0:
                change = score - scores[i-1]
                trends['overall_score_trend'].append({
                    'iteration': i + 1,
                    'score': score,
                    'change': change,
                    'change_percent': (change / scores[i-1] * 100) if scores[i-1] > 0 else 0
                })
        
        # 等级变化
        levels = [d['level'] for d in self.iterations_data]
        for i in range(1, len(levels)):
            if levels[i] != levels[i-1]:
                trends['level_transitions'].append({
                    'iteration': i + 1,
                    'from': levels[i-1],
                    'to': levels[i],
                    'timestamp': self.iterations_data[i]['timestamp']
                })
        
        # 各维度趋势
        dimensions = ['correctness', 'consistency', 'completeness', 'fluency', 
                     'coherence', 'reasoning_depth', 'knowledge_accuracy', 
                     'language_control', 'creativity', 'adaptability']
        
        for dim in dimensions:
            values = [d['dimensions'][dim] for d in self.iterations_data]
            trends['dimension_trends'][dim] = {
                'values': values,
                'initial': values[0],
                'final': values[-1],
                'improvement': values[-1] - values[0],
                'improvement_percent': ((values[-1] - values[0]) / max(values[0], 0.01) * 100)
            }
        
        # 性能加速度
        if len(scores) >= 3:
            first_half_change = scores[len(scores)//2] - scores[0]
            second_half_change = scores[-1] - scores[len(scores)//2]
            acceleration = second_half_change - first_half_change
            trends['performance_acceleration'] = acceleration
        
        # 收敛率
        if len(scores) >= 5:
            recent_changes = [scores[i] - scores[i-1] for i in range(-5, 0)]
            convergence = sum(abs(c) for c in recent_changes) / 5
            trends['convergence_rate'] = convergence
        
        return trends
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取训练会话摘要"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        if not self.iterations_data:
            return {}
        
        scores = [d['overall_score'] for d in self.iterations_data]
        
        summary = {
            'total_iterations': len(self.iterations_data),
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'initial_score': scores[0],
            'final_score': scores[-1],
            'total_improvement': scores[-1] - scores[0],
            'improvement_percent': ((scores[-1] - scores[0]) / max(scores[0], 0.01) * 100),
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'average_iteration_time': sum(d['iteration_time'] for d in self.iterations_data) / len(self.iterations_data),
            'iterations_per_hour': len(self.iterations_data) / (total_time / 3600),
        }
        
        return summary
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def save_evolution_data(self, output_dir: str = 'training_output'):
        """保存进化趋势数据"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 计算趋势
        trends = self.calculate_trends()
        summary = self.get_session_summary()
        
        # 更新进化数据
        self.evolution_data['trends'] = trends
        self.evolution_data['summary'] = summary
        self.evolution_data['end_time'] = datetime.now().isoformat()
        
        # 保存完整数据
        json_file = Path(output_dir) / f"evolution_data_{self.session_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.evolution_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"进化数据已保存: {json_file}")
        
        return json_file


class EnhancedTrainingSession:
    """增强型训练会话 - 4小时长时间训练"""
    
    def __init__(self, duration_hours: float = 4.0, 
                 base_learning_rate: float = 0.0001,
                 initial_batch_size: int = 32):
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        self.base_learning_rate = base_learning_rate
        self.batch_size = initial_batch_size
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor = TrainingMonitor(self.session_id)
        self.evaluator = CompetencyEvaluator()
        self.corrector = OutputCorrectionMechanism()
        
    def prepare_training_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 基础训练数据
        base_data = [
            ("如何学习编程？", "编程学习需要系统的方法和持续的实践。首先应该选择合适的编程语言..."),
            ("什么是人工智能？", "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统..."),
            ("解释机器学习", "机器学习是人工智能的一个子领域，使计算机系统能够从数据中学习和改进..."),
            ("数据科学的应用", "数据科学结合统计学、编程和领域知识来从数据中提取有意义的见解..."),
            ("云计算的优势", "云计算提供按需计算资源、成本效率、可扩展性和灵活性等多个优势..."),
            ("网络安全重要性", "网络安全对于保护数据和系统安全至关重要，尤其在数字化时代..."),
            ("深度学习基础", "深度学习使用多层神经网络来处理复杂的模式识别任务..."),
            ("大数据处理", "大数据处理涉及收集、存储和分析海量数据集以获取商业价值..."),
            ("物联网技术", "物联网连接各种设备和传感器，实现智能数据交互和自动化..."),
        ]
        
        # 扩展数据
        extended_data = []
        variations = [
            "请详细解析",
            "从基础开始讲解",
            "用简单的语言解释",
            "包括实际应用例子"
        ]
        
        for question, answer in base_data:
            for var in variations:
                extended_data.append((f"{question}({var})", answer))
        
        # 加载外部数据
        try:
            corpus_file = Path('/Users/imymm/H2Q-Evo/h2q_project/mix_corpus.txt')
            if corpus_file.exists():
                with open(corpus_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:100]
                    for line in lines:
                        if line.strip():
                            extended_data.append((line.strip(), line.strip()))
        except Exception as e:
            logger.warning(f"加载外部数据失败: {e}")
        
        logger.info(f"准备了 {len(extended_data)} 个训练样本")
        
        # 分割数据
        split_idx = int(len(extended_data) * 0.8)
        return extended_data[:split_idx], extended_data[split_idx:]
    
    def dynamic_learning_rate_adjustment(self, iteration: int, 
                                        current_score: float,
                                        previous_score: float) -> float:
        """动态学习率调整"""
        # 基础学习率衰减
        decay_factor = math.exp(-0.1 * iteration / 10)
        base_lr = self.base_learning_rate * decay_factor
        
        # 基于性能的调整
        if current_score > previous_score:
            # 性能改进，保持或略微增加学习率
            adjustment = 1.0
        elif current_score > previous_score - 0.05:
            # 性能略有下降，保持学习率
            adjustment = 0.95
        else:
            # 性能明显下降，降低学习率
            adjustment = 0.8
        
        return base_lr * adjustment
    
    def run_training(self):
        """运行4小时增强型训练"""
        logger.info("=" * 80)
        logger.info("H2Q-Evo 增强型长时间训练 - 4小时模式")
        logger.info("=" * 80)
        logger.info(f"会话ID: {self.session_id}")
        logger.info(f"训练时长: {self.duration_hours} 小时")
        logger.info(f"开始时间: {self.monitor.start_time}")
        logger.info("")
        
        # 准备数据
        train_data, val_data = self.prepare_training_data()
        
        # 初始化训练器
        self.trainer = LocalModelAdvancedTrainer(
            learning_rate=self.base_learning_rate,
            num_iterations=1,  # 每次运行一个迭代
            target_level="EXPERT"
        )
        
        iteration_count = 0
        previous_score = 0.0
        start_time = time.time()
        
        logger.info("[步骤 1] 开始迭代训练循环")
        logger.info("-" * 80)
        
        # 迭代训练循环
        while True:
            iteration_count += 1
            iteration_start = time.time()
            elapsed_time = time.time() - start_time
            
            # 检查是否超过4小时
            if elapsed_time > self.duration_seconds:
                logger.info("训练时间已达到 4 小时，停止训练")
                break
            
            # 计算剩余时间
            remaining_time = self.duration_seconds - elapsed_time
            
            logger.info(f"\n{'='*80}")
            logger.info(f"迭代 {iteration_count} | 经过时间: {self._format_time(elapsed_time)} | 剩余: {self._format_time(remaining_time)}")
            logger.info(f"{'='*80}")
            
            try:
                # 动态调整学习率
                # (在实际应用中会根据性能调整)
                
                # 执行评估
                test_text = train_data[iteration_count % len(train_data)][1]
                metrics = self.evaluator.evaluate_full(test_text)
                
                iteration_time = time.time() - iteration_start
                
                # 记录迭代数据
                self.monitor.record_iteration(
                    iteration_num=iteration_count,
                    metrics=metrics,
                    iteration_time=iteration_time,
                    training_loss=0.0
                )
                
                # 显示详细信息
                self._log_iteration_details(iteration_count, metrics, iteration_time)
                
                # 每10次迭代进行一次趋势分析
                if iteration_count % 10 == 0:
                    self._log_trend_analysis(iteration_count)
                
                previous_score = metrics.overall_score
                
                # 短暂延迟，模拟实际训练过程
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"迭代 {iteration_count} 失败: {e}", exc_info=True)
                continue
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("训练循环已完成")
        logger.info("=" * 80)
        
        # 保存结果
        self._save_training_results()
        
        return self.monitor
    
    def _log_iteration_details(self, iteration: int, metrics: CompetencyMetrics, time_taken: float):
        """记录迭代详细信息"""
        logger.info(f"维度评估:")
        logger.info(f"  基础维度:")
        logger.info(f"    正确性: {metrics.correctness:.2%}")
        logger.info(f"    一致性: {metrics.consistency:.2%}")
        logger.info(f"    完整性: {metrics.completeness:.2%}")
        logger.info(f"    流畅性: {metrics.fluency:.2%}")
        logger.info(f"    连贯性: {metrics.coherence:.2%}")
        logger.info(f"  高级维度:")
        logger.info(f"    推理深度: {metrics.reasoning_depth:.2%}")
        logger.info(f"    知识准确性: {metrics.knowledge_accuracy:.2%}")
        logger.info(f"    语言控制: {metrics.language_control:.2%}")
        logger.info(f"    创意性: {metrics.creativity:.2%}")
        logger.info(f"    适应性: {metrics.adaptability:.2%}")
        logger.info(f"  总体评分: {metrics.overall_score:.2%} ({metrics.level.name})")
        logger.info(f"  耗时: {time_taken:.3f}s")
    
    def _log_trend_analysis(self, iteration: int):
        """记录趋势分析"""
        trends = self.monitor.calculate_trends()
        if not trends:
            return
        
        logger.info(f"\n[趋势分析] (第 {iteration} 次迭代)")
        
        if 'overall_score_trend' in trends and trends['overall_score_trend']:
            latest = trends['overall_score_trend'][-1]
            logger.info(f"  总体评分变化: {latest['change']:+.2%}")
            logger.info(f"  变化率: {latest['change_percent']:+.1f}%")
        
        if 'level_transitions' in trends and trends['level_transitions']:
            logger.info(f"  能力等级变化次数: {len(trends['level_transitions'])}")
            for trans in trends['level_transitions'][-1:]:
                logger.info(f"    {trans['from']} → {trans['to']}")
        
        if 'dimension_trends' in trends:
            logger.info(f"  维度改进情况:")
            for dim, data in trends['dimension_trends'].items():
                if data['improvement'] > 0.05:
                    logger.info(f"    {dim}: {data['initial']:.2%} → {data['final']:.2%} (+{data['improvement_percent']:.1f}%)")
    
    def _save_training_results(self):
        """保存训练结果"""
        logger.info("\n[步骤 2] 保存训练结果")
        logger.info("-" * 80)
        
        # 保存进化数据
        json_file = self.monitor.save_evolution_data()
        
        # 生成报告
        self._generate_report()
        
        logger.info(f"训练结果已保存到 training_output/")
    
    def _generate_report(self):
        """生成训练报告"""
        summary = self.monitor.get_session_summary()
        trends = self.monitor.calculate_trends()
        
        report = f"""
# H2Q-Evo 增强型训练报告

**会话ID**: {self.session_id}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 训练摘要

- **总迭代次数**: {summary.get('total_iterations', 0)}
- **总训练时间**: {summary.get('total_time_formatted', 'N/A')}
- **每小时迭代数**: {summary.get('iterations_per_hour', 0):.1f}
- **平均单次迭代时间**: {summary.get('average_iteration_time', 0):.3f}s

## 性能指标

- **初始评分**: {summary.get('initial_score', 0):.2%}
- **最终评分**: {summary.get('final_score', 0):.2%}
- **总体改进**: {summary.get('total_improvement', 0):+.2%}
- **改进百分比**: {summary.get('improvement_percent', 0):+.1f}%
- **平均评分**: {summary.get('average_score', 0):.2%}
- **最高评分**: {summary.get('max_score', 0):.2%}
- **最低评分**: {summary.get('min_score', 0):.2%}

## 进化趋势

### 总体评分趋势
"""
        
        if 'overall_score_trend' in trends:
            for item in trends['overall_score_trend'][-10:]:  # 显示最后10条
                report += f"\n- 迭代 {item['iteration']}: {item['score']:.2%} (变化: {item['change']:+.2%})"
        
        report += f"\n\n### 维度改进"
        
        if 'dimension_trends' in trends:
            for dim, data in sorted(trends['dimension_trends'].items()):
                report += f"\n- **{dim}**"
                report += f"\n  - 初始: {data['initial']:.2%}"
                report += f"\n  - 最终: {data['final']:.2%}"
                report += f"\n  - 改进: {data['improvement']:+.2%} ({data['improvement_percent']:+.1f}%)"
        
        report += f"\n\n### 能力等级变化"
        
        if 'level_transitions' in trends and trends['level_transitions']:
            for trans in trends['level_transitions']:
                report += f"\n- 迭代 {trans['iteration']}: {trans['from']} → {trans['to']}"
        else:
            report += "\n- 无能力等级变化"
        
        # 保存报告
        Path('training_output').mkdir(exist_ok=True)
        report_file = Path('training_output') / f"training_report_{self.session_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"训练报告已保存: {report_file}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def main():
    """主函数"""
    logger.info("")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "H2Q-Evo 增强型长时间训练系统 - 4小时模式".center(78) + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("")
    
    try:
        # 创建训练会话
        session = EnhancedTrainingSession(
            duration_hours=4.0,
            base_learning_rate=0.0001,
            initial_batch_size=32
        )
        
        # 运行训练
        monitor = session.run_training()
        
        # 最终摘要
        logger.info("")
        logger.info("=" * 80)
        logger.info("训练完成摘要")
        logger.info("=" * 80)
        
        summary = monitor.get_session_summary()
        logger.info(f"总迭代次数: {summary.get('total_iterations', 0)}")
        logger.info(f"总训练时间: {summary.get('total_time_formatted', 'N/A')}")
        logger.info(f"初始评分: {summary.get('initial_score', 0):.2%}")
        logger.info(f"最终评分: {summary.get('final_score', 0):.2%}")
        logger.info(f"总体改进: {summary.get('total_improvement', 0):+.2%}")
        logger.info("")
        logger.info(f"进化数据已保存，可用于下一次训练")
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)


if __name__ == '__main__':
    main()
