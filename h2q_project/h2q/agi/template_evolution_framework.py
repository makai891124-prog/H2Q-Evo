#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板化进化框架：为AGI系统提供通用的自我进化模式
支持多种演化策略、自适应学习和性能优化
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """进化阶段枚举"""
    INITIALIZATION = "初始化"
    PROBLEM_GENERATION = "问题生成"
    SOLUTION_ATTEMPT = "解决尝试"
    EXTERNAL_VERIFICATION = "外部验证"
    HONESTY_VERIFICATION = "诚实验证"
    IMPROVEMENT = "改进"
    INTEGRATION = "集成"
    EVALUATION = "评估"
    COMPLETION = "完成"


@dataclass
class EvolutionStep:
    """进化步骤记录"""
    step_id: str
    phase: EvolutionPhase
    timestamp: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    gemini_feedback: Optional[Dict[str, Any]] = None
    m24_verification: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['phase'] = self.phase.value
        return data


@dataclass
class EvolutionTemplate:
    """进化模板配置"""
    name: str
    description: str
    phases: List[EvolutionPhase]
    max_iterations: int
    convergence_threshold: float
    use_external_feedback: bool = True
    use_honesty_verification: bool = True


class TemplateEvolutionFramework:
    """
    模板化的进化框架
    提供通用的进化流程，支持自定义策略
    """
    
    def __init__(self, output_dir: str = "./evolution_results"):
        """初始化框架"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.evolution_history = []
        self.current_iteration = 0
        self.performance_metrics = {
            'initial': None,
            'best': None,
            'current': None,
            'history': []
        }
        
        # 组件引用
        self.gemini_integration = None
        self.m24_protocol = None
        self.ensemble_system = None
        
    def register_gemini_integration(self, integration):
        """注册Gemini集成模块"""
        self.gemini_integration = integration
        logger.info("✓ Gemini集成已注册")
    
    def register_m24_protocol(self, protocol):
        """注册M24诚实协议"""
        self.m24_protocol = protocol
        logger.info("✓ M24诚实协议已注册")
    
    def register_ensemble_system(self, system):
        """注册多模型系统"""
        self.ensemble_system = system
        logger.info("✓ 多模型系统已注册")
    
    def create_template(self, name: str, **kwargs) -> EvolutionTemplate:
        """创建进化模板"""
        return EvolutionTemplate(
            name=name,
            description=kwargs.get('description', ''),
            phases=kwargs.get('phases', list(EvolutionPhase)),
            max_iterations=kwargs.get('max_iterations', 10),
            convergence_threshold=kwargs.get('convergence_threshold', 0.95),
            use_external_feedback=kwargs.get('use_external_feedback', True),
            use_honesty_verification=kwargs.get('use_honesty_verification', True)
        )
    
    def run_evolution_cycle(self, 
                           template: EvolutionTemplate,
                           initial_state: Dict[str, Any],
                           problem_generator: Callable,
                           solver: Callable) -> Dict[str, Any]:
        """
        运行完整的进化周期
        
        Args:
            template: 进化模板
            initial_state: 初始状态
            problem_generator: 问题生成函数
            solver: 问题解决函数
            
        Returns:
            进化结果
        """
        logger.info(f"🚀 开始进化周期: {template.name}")
        
        cycle_id = str(uuid.uuid4())[:8]
        evolution_log = {
            'cycle_id': cycle_id,
            'template': template.name,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metrics': {}
        }
        
        current_state = initial_state.copy()
        self.current_iteration = 0
        
        try:
            for iteration in range(template.max_iterations):
                self.current_iteration = iteration
                logger.info(f"\n【迭代 {iteration + 1}/{template.max_iterations}】")
                
                # 执行进化阶段
                for phase in template.phases:
                    step = self._execute_phase(
                        phase=phase,
                        current_state=current_state,
                        problem_generator=problem_generator,
                        solver=solver,
                        template=template,
                        iteration=iteration
                    )
                    
                    if step:
                        evolution_log['steps'].append(step.to_dict())
                        
                        # 更新状态
                        if step.output_data:
                            current_state.update(step.output_data)
                
                # 检查收敛
                if self._check_convergence(template.convergence_threshold):
                    logger.info(f"✓ 系统已收敛，在迭代 {iteration + 1} 停止")
                    break
                
                # 更新性能指标
                self._update_metrics(current_state, iteration)
        
        except Exception as e:
            logger.error(f"✗ 进化周期异常: {e}")
            evolution_log['error'] = str(e)
        
        # 保存进化日志
        evolution_log['end_time'] = datetime.now().isoformat()
        evolution_log['total_iterations'] = self.current_iteration + 1
        evolution_log['metrics'] = self.performance_metrics
        
        self._save_evolution_log(evolution_log, cycle_id)
        
        return {
            'cycle_id': cycle_id,
            'success': True,
            'final_state': current_state,
            'evolution_log': evolution_log,
            'metrics': self.performance_metrics
        }
    
    def _execute_phase(self,
                       phase: EvolutionPhase,
                       current_state: Dict,
                       problem_generator: Callable,
                       solver: Callable,
                       template: EvolutionTemplate,
                       iteration: int) -> Optional[EvolutionStep]:
        """执行单个进化阶段"""
        
        step_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        try:
            if phase == EvolutionPhase.INITIALIZATION:
                return self._phase_initialization(step_id, timestamp, current_state)
            
            elif phase == EvolutionPhase.PROBLEM_GENERATION:
                return self._phase_problem_generation(step_id, timestamp, current_state, problem_generator)
            
            elif phase == EvolutionPhase.SOLUTION_ATTEMPT:
                return self._phase_solution_attempt(step_id, timestamp, current_state, solver)
            
            elif phase == EvolutionPhase.EXTERNAL_VERIFICATION:
                if template.use_external_feedback and self.gemini_integration:
                    return self._phase_external_verification(step_id, timestamp, current_state)
            
            elif phase == EvolutionPhase.HONESTY_VERIFICATION:
                if template.use_honesty_verification and self.m24_protocol:
                    return self._phase_honesty_verification(step_id, timestamp, current_state)
            
            elif phase == EvolutionPhase.IMPROVEMENT:
                return self._phase_improvement(step_id, timestamp, current_state)
            
            elif phase == EvolutionPhase.INTEGRATION:
                return self._phase_integration(step_id, timestamp, current_state)
            
            elif phase == EvolutionPhase.EVALUATION:
                return self._phase_evaluation(step_id, timestamp, current_state)
            
            return None
        
        except Exception as e:
            logger.error(f"✗ 阶段 {phase.value} 执行失败: {e}")
            return EvolutionStep(
                step_id=step_id,
                phase=phase,
                timestamp=timestamp,
                input_data=current_state,
                output_data={},
                errors=[str(e)]
            )
    
    def _phase_initialization(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """初始化阶段"""
        logger.info(f"  【初始化】")
        return EvolutionStep(
            step_id=step_id,
            phase=EvolutionPhase.INITIALIZATION,
            timestamp=timestamp,
            input_data=state,
            output_data={'initialized': True, 'state_size': len(state)}
        )
    
    def _phase_problem_generation(self, step_id: str, timestamp: str, 
                                 state: Dict, generator: Callable) -> EvolutionStep:
        """问题生成阶段"""
        logger.info(f"  【自动问题生成】")
        
        try:
            problems = generator(state)
            
            return EvolutionStep(
                step_id=step_id,
                phase=EvolutionPhase.PROBLEM_GENERATION,
                timestamp=timestamp,
                input_data=state,
                output_data={'problems': problems, 'count': len(problems)},
                metrics={'problem_diversity': self._calculate_diversity(problems)}
            )
        except Exception as e:
            logger.error(f"问题生成失败: {e}")
            return None
    
    def _phase_solution_attempt(self, step_id: str, timestamp: str,
                               state: Dict, solver: Callable) -> EvolutionStep:
        """解决尝试阶段"""
        logger.info(f"  【尝试解决问题】")
        
        try:
            if 'problems' not in state:
                return None
            
            solutions = []
            for problem in state['problems']:
                solution = solver(problem)
                solutions.append(solution)
            
            return EvolutionStep(
                step_id=step_id,
                phase=EvolutionPhase.SOLUTION_ATTEMPT,
                timestamp=timestamp,
                input_data=state,
                output_data={'solutions': solutions, 'count': len(solutions)},
                metrics={'success_rate': sum(1 for s in solutions if s.get('success')) / len(solutions)}
            )
        except Exception as e:
            logger.error(f"解决尝试失败: {e}")
            return None
    
    def _phase_external_verification(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """外部验证阶段 (Gemini)"""
        logger.info(f"  【Gemini外部验证】")
        
        if not self.gemini_integration:
            return None
        
        try:
            solutions = state.get('solutions', [])
            verification_results = []
            
            for i, solution in enumerate(solutions):
                # 向Gemini请求验证
                feedback = self.gemini_integration.analyze_decision(
                    decision=solution,
                    reasoning=json.dumps(solution.get('reasoning', {}), ensure_ascii=False)
                )
                verification_results.append(feedback)
            
            return EvolutionStep(
                step_id=step_id,
                phase=EvolutionPhase.EXTERNAL_VERIFICATION,
                timestamp=timestamp,
                input_data=state,
                output_data={'verification_results': verification_results},
                gemini_feedback=verification_results[0] if verification_results else None,
                metrics={'verification_coverage': len(verification_results) / max(1, len(solutions))}
            )
        except Exception as e:
            logger.error(f"外部验证失败: {e}")
            return None
    
    def _phase_honesty_verification(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """诚实验证阶段 (M24)"""
        logger.info(f"  【M24诚实验证】")
        
        if not self.m24_protocol:
            return None
        
        try:
            solutions = state.get('solutions', [])
            m24_results = []
            
            for solution in solutions:
                # M24协议验证
                result = self.m24_protocol.audit_decision(
                    decision=solution,
                    context=state
                )
                m24_results.append(result)
            
            # 统计诚实性
            honest_count = sum(1 for r in m24_results if r.get('honesty_level', '').startswith('PROVEN'))
            honesty_score = honest_count / max(1, len(m24_results))
            
            return EvolutionStep(
                step_id=step_id,
                phase=EvolutionPhase.HONESTY_VERIFICATION,
                timestamp=timestamp,
                input_data=state,
                output_data={'m24_results': m24_results, 'count': len(m24_results)},
                m24_verification=m24_results[0] if m24_results else None,
                metrics={'honesty_score': honesty_score, 'fraud_free': honesty_score > 0.8}
            )
        except Exception as e:
            logger.error(f"诚实验证失败: {e}")
            return None
    
    def _phase_improvement(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """改进阶段"""
        logger.info(f"  【基于反馈改进】")
        
        improvements = {}
        
        # 基于Gemini反馈的改进
        if 'verification_results' in state:
            improvements['gemini_improvements'] = self._extract_improvements(state['verification_results'])
        
        # 基于M24反馈的改进
        if 'm24_results' in state:
            improvements['honesty_improvements'] = self._extract_honesty_improvements(state['m24_results'])
        
        return EvolutionStep(
            step_id=step_id,
            phase=EvolutionPhase.IMPROVEMENT,
            timestamp=timestamp,
            input_data=state,
            output_data=improvements,
            metrics={'improvement_suggestions': sum(len(v) if isinstance(v, list) else 1 for v in improvements.values())}
        )
    
    def _phase_integration(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """集成阶段"""
        logger.info(f"  【集成改进】")
        
        integrated_state = state.copy()
        
        # 集成改进
        if 'gemini_improvements' in state:
            integrated_state['gemini_integrated'] = True
        
        if 'honesty_improvements' in state:
            integrated_state['m24_integrated'] = True
        
        return EvolutionStep(
            step_id=step_id,
            phase=EvolutionPhase.INTEGRATION,
            timestamp=timestamp,
            input_data=state,
            output_data=integrated_state,
            metrics={'integration_status': 'success'}
        )
    
    def _phase_evaluation(self, step_id: str, timestamp: str, state: Dict) -> EvolutionStep:
        """评估阶段"""
        logger.info(f"  【性能评估】")
        
        metrics = {
            'problem_count': len(state.get('problems', [])),
            'solution_count': len(state.get('solutions', [])),
            'verification_status': 'completed',
            'improvements_applied': 'gemini_integrated' in state and 'm24_integrated' in state
        }
        
        return EvolutionStep(
            step_id=step_id,
            phase=EvolutionPhase.EVALUATION,
            timestamp=timestamp,
            input_data=state,
            output_data=metrics,
            metrics=metrics
        )
    
    def _calculate_diversity(self, items: List[Any]) -> float:
        """计算问题多样性"""
        if not items:
            return 0.0
        # 简化计算：项目数量越多多样性越高
        return min(1.0, len(items) / 10.0)
    
    def _extract_improvements(self, results: List[Dict]) -> List[Dict]:
        """从验证结果提取改进建议"""
        improvements = []
        for result in results:
            if isinstance(result, dict) and 'analysis' in result:
                analysis = result['analysis']
                if isinstance(analysis, dict) and '改进建议' in analysis:
                    improvements.append(analysis['改进建议'])
        return improvements
    
    def _extract_honesty_improvements(self, results: List[Dict]) -> List[Dict]:
        """从M24结果提取诚实性改进"""
        improvements = []
        for result in results:
            if isinstance(result, dict):
                if result.get('honesty_level', '').startswith('UNCERTAIN'):
                    improvements.append({
                        'type': 'honesty',
                        'reason': '诚实性评分不足',
                        'action': '需要增强验证'
                    })
        return improvements
    
    def _check_convergence(self, threshold: float) -> bool:
        """检查系统是否已收敛"""
        if len(self.performance_metrics['history']) < 2:
            return False
        
        recent = self.performance_metrics['history'][-1]
        previous = self.performance_metrics['history'][-2]
        
        if isinstance(recent, dict) and isinstance(previous, dict):
            recent_score = recent.get('overall_score', 0)
            previous_score = previous.get('overall_score', 0)
            
            # 改进小于阈值则认为收敛
            improvement = abs(recent_score - previous_score) / max(0.1, previous_score)
            return improvement < (1 - threshold)
        
        return False
    
    def _update_metrics(self, state: Dict, iteration: int):
        """更新性能指标"""
        metrics = {
            'iteration': iteration,
            'overall_score': self._calculate_overall_score(state),
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_metrics['history'].append(metrics)
        
        if self.performance_metrics['initial'] is None:
            self.performance_metrics['initial'] = metrics
        
        if self.performance_metrics['best'] is None or metrics['overall_score'] > self.performance_metrics['best']['overall_score']:
            self.performance_metrics['best'] = metrics
        
        self.performance_metrics['current'] = metrics
    
    def _calculate_overall_score(self, state: Dict) -> float:
        """计算总体性能分数"""
        score = 0.0
        weight_sum = 0.0
        
        # 问题生成得分
        if 'problems' in state:
            score += len(state['problems']) * 0.3
            weight_sum += 0.3
        
        # 解决方案得分
        if 'solutions' in state:
            success_rate = sum(1 for s in state['solutions'] if s.get('success')) / max(1, len(state['solutions']))
            score += success_rate * 0.3
            weight_sum += 0.3
        
        # 诚实性得分
        if 'm24_results' in state:
            honesty_score = sum(1 for r in state['m24_results'] if r.get('honesty_level', '').startswith('PROVEN')) / max(1, len(state['m24_results']))
            score += honesty_score * 0.4
            weight_sum += 0.4
        
        return score / max(0.1, weight_sum)
    
    def _save_evolution_log(self, log: Dict, cycle_id: str):
        """保存进化日志"""
        log_file = self.output_dir / f"evolution_{cycle_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 进化日志已保存: {log_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    framework = TemplateEvolutionFramework()
    
    # 创建进化模板
    template = framework.create_template(
        name="基础进化模板",
        description="用于演示的基础进化流程",
        phases=[
            EvolutionPhase.INITIALIZATION,
            EvolutionPhase.PROBLEM_GENERATION,
            EvolutionPhase.SOLUTION_ATTEMPT,
            EvolutionPhase.IMPROVEMENT,
            EvolutionPhase.EVALUATION
        ],
        max_iterations=3,
        convergence_threshold=0.9
    )
    
    print(f"✓ 进化模板已创建: {template.name}")
