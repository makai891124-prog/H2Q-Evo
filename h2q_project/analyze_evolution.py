#!/usr/bin/env python3
"""
进化趋势分析和可视化工具

功能:
- 分析训练会话的进化趋势
- 生成进化报告
- 为下一次训练提供基础数据
- 可视化性能变化
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

sys.path.insert(0, '/Users/imymm/H2Q-Evo')


class EvolutionAnalyzer:
    """进化趋势分析器"""
    
    def __init__(self, evolution_file: str):
        """初始化分析器"""
        self.evolution_file = Path(evolution_file)
        if not self.evolution_file.exists():
            raise FileNotFoundError(f"文件不存在: {evolution_file}")
        
        with open(self.evolution_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            'session_id': self.data.get('session_id'),
            'start_time': self.data.get('start_time'),
            'end_time': self.data.get('end_time'),
            'iterations': len(self.data.get('iterations', [])),
        }
    
    def analyze_score_progression(self) -> Dict[str, Any]:
        """分析评分进度"""
        iterations = self.data.get('iterations', [])
        if not iterations:
            return {}
        
        scores = [it['overall_score'] for it in iterations]
        
        analysis = {
            'initial_score': scores[0],
            'final_score': scores[-1],
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'total_change': scores[-1] - scores[0],
            'change_percent': ((scores[-1] - scores[0]) / max(scores[0], 0.01) * 100),
            'score_range': max(scores) - min(scores),
        }
        
        # 计算标准差
        if len(scores) > 1:
            analysis['std_dev'] = statistics.stdev(scores)
        
        return analysis
    
    def analyze_dimension_evolution(self) -> Dict[str, Dict[str, Any]]:
        """分析各维度的进化"""
        iterations = self.data.get('iterations', [])
        if not iterations:
            return {}
        
        dimensions = [
            'correctness', 'consistency', 'completeness', 'fluency', 'coherence',
            'reasoning_depth', 'knowledge_accuracy', 'language_control', 
            'creativity', 'adaptability'
        ]
        
        analysis = {}
        
        for dim in dimensions:
            values = [it['dimensions'].get(dim, 0) for it in iterations]
            
            if not values:
                continue
            
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            
            analysis[dim] = {
                'initial': values[0],
                'final': values[-1],
                'improvement': values[-1] - values[0],
                'improvement_percent': ((values[-1] - values[0]) / max(values[0], 0.01) * 100),
                'min': min(values),
                'max': max(values),
                'avg': statistics.mean(values),
                'total_changes': len([c for c in changes if abs(c) > 0.01]),
                'avg_change': statistics.mean(changes) if changes else 0,
            }
        
        return analysis
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """识别瓶颈维度 (需要改进的领域)"""
        dim_analysis = self.analyze_dimension_evolution()
        
        bottlenecks = []
        for dim, stats in dim_analysis.items():
            if stats['final'] < 0.5:  # 低于 50% 为瓶颈
                bottlenecks.append({
                    'dimension': dim,
                    'current_score': stats['final'],
                    'improvement_needed': 0.7 - stats['final'],  # 目标 70%
                    'priority': 'high' if stats['final'] < 0.3 else 'medium'
                })
        
        # 按优先级排序
        return sorted(bottlenecks, key=lambda x: (
            x['priority'] == 'low',
            x['current_score']
        ))
    
    def identify_strengths(self) -> List[Dict[str, Any]]:
        """识别优势维度 (已经很好的领域)"""
        dim_analysis = self.analyze_dimension_evolution()
        
        strengths = []
        for dim, stats in dim_analysis.items():
            if stats['final'] > 0.75:  # 高于 75% 为优势
                strengths.append({
                    'dimension': dim,
                    'current_score': stats['final'],
                    'improvement_potential': 1.0 - stats['final'],
                })
        
        # 按分数排序
        return sorted(strengths, key=lambda x: x['current_score'], reverse=True)
    
    def estimate_convergence(self) -> Dict[str, Any]:
        """估计收敛趋势"""
        iterations = self.data.get('iterations', [])
        if len(iterations) < 5:
            return {'status': '数据不足', 'message': '需要至少 5 次迭代'}
        
        scores = [it['overall_score'] for it in iterations]
        
        # 计算最后 N 次迭代的变化
        recent_changes = [abs(scores[i] - scores[i-1]) for i in range(-5, 0)]
        avg_recent_change = statistics.mean(recent_changes)
        
        # 计算所有变化的平均值
        all_changes = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
        avg_all_change = statistics.mean(all_changes)
        
        convergence_status = 'converging' if avg_recent_change < avg_all_change * 0.5 else 'exploring'
        
        return {
            'status': convergence_status,
            'recent_avg_change': avg_recent_change,
            'overall_avg_change': avg_all_change,
            'convergence_rate': (1 - (avg_recent_change / max(avg_all_change, 0.01))) * 100,
        }
    
    def generate_next_training_recommendations(self) -> Dict[str, Any]:
        """为下一次训练生成建议"""
        score_prog = self.analyze_score_progression()
        bottlenecks = self.identify_bottlenecks()
        strengths = self.identify_strengths()
        convergence = self.estimate_convergence()
        
        recommendations = {
            'overall_status': 'ready' if score_prog['final_score'] > 0.5 else 'needs_improvement',
            'training_duration': 4.0 if score_prog['final_score'] < 0.6 else 2.0,
            'learning_rate': 0.0001 if convergence.get('status') == 'exploring' else 0.00005,
            'focus_areas': [b['dimension'] for b in bottlenecks[:3]],
            'maintain_areas': [s['dimension'] for s in strengths[:3]],
            'priority_actions': [],
        }
        
        # 生成优先级操作
        for bottleneck in bottlenecks[:3]:
            recommendations['priority_actions'].append({
                'action': f'improve {bottleneck["dimension"]}',
                'priority': bottleneck['priority'],
                'target_score': 0.7,
                'current_score': bottleneck['current_score'],
            })
        
        return recommendations
    
    def generate_full_report(self) -> str:
        """生成完整分析报告"""
        session_info = self.get_session_info()
        score_prog = self.analyze_score_progression()
        dim_analysis = self.analyze_dimension_evolution()
        bottlenecks = self.identify_bottlenecks()
        strengths = self.identify_strengths()
        convergence = self.estimate_convergence()
        recommendations = self.generate_next_training_recommendations()
        
        report = f"""
# H2Q-Evo 进化趋势分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 会话信息

- **会话ID**: {session_info['session_id']}
- **开始时间**: {session_info['start_time']}
- **结束时间**: {session_info['end_time']}
- **迭代次数**: {session_info['iterations']}

## 性能进度分析

### 评分统计

- **初始评分**: {score_prog['initial_score']:.2%}
- **最终评分**: {score_prog['final_score']:.2%}
- **总体改进**: {score_prog['total_change']:+.2%}
- **改进百分比**: {score_prog['change_percent']:+.1f}%
- **平均评分**: {score_prog['avg_score']:.2%}
- **中位数**: {score_prog['median_score']:.2%}
- **最小值**: {score_prog['min_score']:.2%}
- **最大值**: {score_prog['max_score']:.2%}
- **分数范围**: {score_prog['score_range']:.2%}
- **标准差**: {score_prog.get('std_dev', 0):.2%}

## 维度进化分析

### 维度性能表 (按改进幅度排序)

| 维度 | 初始 | 最终 | 改进 | 改进% | 状态 |
|-----|------|------|------|-------|------|
"""
        
        # 按改进幅度排序
        sorted_dims = sorted(
            dim_analysis.items(),
            key=lambda x: x[1]['improvement'],
            reverse=True
        )
        
        for dim, stats in sorted_dims:
            improvement = stats['improvement']
            status = "📈 提升" if improvement > 0.05 else "📉 下降" if improvement < -0.05 else "➡️ 稳定"
            report += f"\n| {dim} | {stats['initial']:.1%} | {stats['final']:.1%} | {improvement:+.1%} | {stats['improvement_percent']:+.1f}% | {status} |"
        
        report += f"\n\n## 优势领域 (高于 75%)\n\n"
        
        for strength in strengths:
            report += f"- **{strength['dimension']}**: {strength['current_score']:.2%}"
            report += f" (提升空间: {strength['improvement_potential']:.2%})\n"
        
        if not strengths:
            report += "- 暂无高分维度\n"
        
        report += f"\n## 瓶颈领域 (低于 50%)\n\n"
        
        for bottleneck in bottlenecks:
            report += f"- **{bottleneck['dimension']}**: {bottleneck['current_score']:.2%}"
            report += f" (需要改进: {bottleneck['improvement_needed']:.2%}) - 优先级: {bottleneck['priority'].upper()}\n"
        
        if not bottlenecks:
            report += "- 所有维度表现良好\n"
        
        report += f"\n## 收敛趋势分析\n\n"
        report += f"- **状态**: {convergence.get('status', 'N/A')}\n"
        report += f"- **最近平均变化**: {convergence.get('recent_avg_change', 0):.4f}\n"
        report += f"- **整体平均变化**: {convergence.get('overall_avg_change', 0):.4f}\n"
        report += f"- **收敛率**: {convergence.get('convergence_rate', 0):.1f}%\n"
        
        report += f"\n## 下一次训练建议\n\n"
        report += f"- **整体状态**: {recommendations['overall_status']}\n"
        report += f"- **建议训练时长**: {recommendations['training_duration']} 小时\n"
        report += f"- **建议学习率**: {recommendations['learning_rate']}\n"
        report += f"- **重点改进领域**:\n"
        
        for area in recommendations['focus_areas']:
            report += f"  - {area}\n"
        
        report += f"- **维持领域**:\n"
        
        for area in recommendations['maintain_areas']:
            report += f"  - {area}\n"
        
        report += f"\n### 优先级操作清单\n\n"
        
        for i, action in enumerate(recommendations['priority_actions'], 1):
            report += f"{i}. **{action['action']}** (优先级: {action['priority'].upper()})\n"
            report += f"   - 当前分数: {action['current_score']:.2%}\n"
            report += f"   - 目标分数: {action['target_score']:.2%}\n"
        
        report += f"\n---\n\n*报告由 H2Q-Evo 进化趋势分析工具生成*\n"
        
        return report
    
    def save_report(self, output_file: str = None) -> Path:
        """保存报告"""
        if not output_file:
            session_id = self.data.get('session_id', 'unknown')
            output_file = f"evolution_analysis_{session_id}.md"
        
        output_path = Path('training_output') / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        report = self.generate_full_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return output_path


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 analyze_evolution.py <evolution_file.json>")
        print("\n示例:")
        print("  python3 analyze_evolution.py training_output/evolution_data_20260120_110000.json")
        sys.exit(1)
    
    evolution_file = sys.argv[1]
    
    try:
        analyzer = EvolutionAnalyzer(evolution_file)
        
        # 生成报告
        report_path = analyzer.save_report()
        
        print("\n" + "="*80)
        print("进化趋势分析报告已生成".center(80))
        print("="*80)
        print(f"\n报告位置: {report_path}")
        print("\n报告摘要:")
        print("-"*80)
        
        # 显示摘要
        session_info = analyzer.get_session_info()
        score_prog = analyzer.analyze_score_progression()
        bottlenecks = analyzer.identify_bottlenecks()
        strengths = analyzer.identify_strengths()
        recommendations = analyzer.generate_next_training_recommendations()
        
        print(f"\n会话ID: {session_info['session_id']}")
        print(f"迭代次数: {session_info['iterations']}")
        print(f"\n性能变化:")
        print(f"  初始评分: {score_prog['initial_score']:.2%}")
        print(f"  最终评分: {score_prog['final_score']:.2%}")
        print(f"  总体改进: {score_prog['total_change']:+.2%} ({score_prog['change_percent']:+.1f}%)")
        
        print(f"\n优势领域 ({len(strengths)}):")
        for strength in strengths[:3]:
            print(f"  • {strength['dimension']}: {strength['current_score']:.2%}")
        
        print(f"\n瓶颈领域 ({len(bottlenecks)}):")
        for bottleneck in bottlenecks[:3]:
            print(f"  • {bottleneck['dimension']}: {bottleneck['current_score']:.2%}")
        
        print(f"\n下一次训练建议:")
        print(f"  • 训练时长: {recommendations['training_duration']} 小时")
        print(f"  • 学习率: {recommendations['learning_rate']}")
        print(f"  • 重点改进: {', '.join(recommendations['focus_areas'][:2])}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
