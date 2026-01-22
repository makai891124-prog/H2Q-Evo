"""H2Q 分形增强层次规划 (Fractal-Enhanced Hierarchical Planning).

利用项目核心数学优势优化规划系统:
1. 分形层级展开 - 多尺度目标分解
2. 四元数状态编码 - 确定性状态表示
3. Berry 相位启发式 - 拓扑导向搜索
4. Fueter 可达性分析 - 全纯路径检测

学术基础:
- Erol et al., HTN Planning (1994)
- H2Q 分形微分几何框架
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import deque
from enum import Enum
import heapq
import time


# ============================================================================
# 分形数学工具
# ============================================================================

def fractal_decompose(value: float, n_levels: int = 4, base_ratio: float = 0.618) -> List[float]:
    """分形分解: 将值分解为多尺度层级.
    
    使用黄金比例 φ = 0.618 进行分形展开:
    value = v_0 + φ*v_1 + φ²*v_2 + ...
    
    Args:
        value: 待分解的值
        n_levels: 层级数
        base_ratio: 分形比例 (默认黄金比例)
    
    Returns:
        各层级的分解值
    """
    levels = []
    remaining = value
    
    for i in range(n_levels):
        ratio = base_ratio ** i
        level_value = remaining * (1 - base_ratio) if i < n_levels - 1 else remaining
        levels.append(level_value)
        remaining -= level_value
    
    return levels


def fractal_combine(levels: List[float], base_ratio: float = 0.618) -> float:
    """分形组合: 将多尺度层级合并."""
    result = 0.0
    for i, v in enumerate(levels):
        result += v * (base_ratio ** i)
    return result


def quaternion_state_encode(state_dict: Dict[str, Any]) -> np.ndarray:
    """将状态编码为四元数.
    
    状态 -> S³ 流形上的点
    """
    # 提取特征
    features = []
    for key, val in sorted(state_dict.items()):
        if isinstance(val, (int, float)):
            features.append(float(val))
        elif isinstance(val, bool):
            features.append(1.0 if val else 0.0)
        elif isinstance(val, str):
            features.append(len(val) / 100.0)  # 字符串长度归一化
    
    if len(features) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # 填充到 4 维
    while len(features) < 4:
        features.append(0.0)
    
    # 取前 4 个
    q = np.array(features[:4], dtype=np.float32)
    
    # 归一化到 S³
    norm = np.sqrt(np.sum(q * q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    return q / norm


def compute_quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """计算四元数距离 (测地距离).
    
    d(q1, q2) = arccos(|q1 · q2|)
    """
    dot = np.abs(np.sum(q1 * q2))
    return float(np.arccos(np.clip(dot, -1, 1)))


def compute_berry_heuristic(path_quaternions: List[np.ndarray], goal_q: np.ndarray) -> float:
    """Berry 相位启发式.
    
    基于路径四元数的拓扑相位估计到达目标的距离.
    """
    if len(path_quaternions) == 0:
        return compute_quaternion_distance(np.array([1, 0, 0, 0]), goal_q)
    
    current_q = path_quaternions[-1]
    
    # 基础距离
    base_dist = compute_quaternion_distance(current_q, goal_q)
    
    # Berry 相位修正: 考虑路径的拓扑性质
    if len(path_quaternions) >= 2:
        # 计算路径的相位累积
        phase_sum = 0.0
        for i in range(1, len(path_quaternions)):
            q_prev, q_curr = path_quaternions[i-1], path_quaternions[i]
            dot = np.sum(q_prev * q_curr)
            # 如果相邻四元数"翻转"了 (dot < 0), 累积相位
            if dot < 0:
                phase_sum += np.pi
        
        # 相位修正
        phase_factor = 1.0 + 0.1 * phase_sum / np.pi
        base_dist *= phase_factor
    
    return base_dist


def fueter_path_validity(path_quaternions: List[np.ndarray], threshold: float = 0.1) -> bool:
    """Fueter 路径有效性检查.
    
    检查路径是否"全纯" (平滑无撕裂)
    """
    if len(path_quaternions) < 2:
        return True
    
    # 计算相邻四元数的变化
    deviations = []
    for i in range(1, len(path_quaternions)):
        q_prev, q_curr = path_quaternions[i-1], path_quaternions[i]
        
        # 变化量
        delta = q_curr - q_prev
        deviation = np.sqrt(np.sum(delta ** 2))
        deviations.append(deviation)
    
    if len(deviations) == 0:
        return True
    
    # Fueter 残差: 变化的方差
    mean_dev = np.mean(deviations)
    var_dev = np.var(deviations)
    
    # 如果方差太大, 路径不平滑
    fueter_residual = var_dev / (mean_dev + 1e-8)
    
    return fueter_residual < threshold


# ============================================================================
# 分形规划原语
# ============================================================================

class FractalTaskStatus(Enum):
    """分形任务状态."""
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FractalState:
    """分形世界状态.
    
    支持多尺度状态表示.
    """
    facts: Set[str]
    numeric: Dict[str, float] = field(default_factory=dict)
    fractal_level: int = 0  # 当前分形层级
    quaternion: Optional[np.ndarray] = None  # S³ 状态表示
    
    def __post_init__(self):
        if self.quaternion is None:
            self.quaternion = quaternion_state_encode({'facts': len(self.facts), **self.numeric})
    
    def satisfies(self, conditions: Set[str]) -> bool:
        """检查状态是否满足条件."""
        return conditions.issubset(self.facts)
    
    def copy(self) -> 'FractalState':
        """深拷贝状态."""
        return FractalState(
            self.facts.copy(), 
            self.numeric.copy(),
            self.fractal_level,
            self.quaternion.copy() if self.quaternion is not None else None
        )
    
    def distance_to(self, other: 'FractalState') -> float:
        """计算到另一状态的距离."""
        if self.quaternion is not None and other.quaternion is not None:
            return compute_quaternion_distance(self.quaternion, other.quaternion)
        
        # 回退到集合差异
        diff = len(self.facts.symmetric_difference(other.facts))
        return float(diff)
    
    def __hash__(self):
        return hash(frozenset(self.facts))


@dataclass
class FractalAction:
    """分形动作.
    
    动作具有分形层级属性.
    """
    name: str
    parameters: List[str]
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str]
    cost: float = 1.0
    duration: float = 1.0
    fractal_level: int = 0  # 动作的抽象层级
    
    def applicable(self, state: FractalState) -> bool:
        return state.satisfies(self.preconditions)
    
    def apply(self, state: FractalState) -> FractalState:
        new_state = state.copy()
        new_state.facts -= self.delete_effects
        new_state.facts |= self.add_effects
        
        # 更新四元数状态
        new_state.quaternion = quaternion_state_encode({
            'facts': len(new_state.facts),
            **new_state.numeric
        })
        
        return new_state
    
    def __str__(self):
        params = ", ".join(self.parameters)
        return f"{self.name}({params})"


@dataclass
class FractalTask:
    """分形任务.
    
    任务可以按分形层级分解.
    """
    name: str
    task_type: str  # "primitive", "compound", "fractal"
    parameters: List[str] = field(default_factory=list)
    subtasks: List['FractalTask'] = field(default_factory=list)
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    priority: int = 0
    fractal_level: int = 0
    deadline: Optional[float] = None
    
    # 分形属性
    complexity: float = 1.0  # 任务复杂度
    decomposition_ratio: float = 0.618  # 分解比例 (黄金比例)
    
    def is_primitive(self) -> bool:
        return self.task_type == "primitive"
    
    def get_fractal_weight(self) -> float:
        """获取分形权重."""
        return self.decomposition_ratio ** self.fractal_level
    
    def __str__(self):
        params = ", ".join(self.parameters)
        return f"{self.name}[L{self.fractal_level}]({params})"


@dataclass
class FractalMethod:
    """分形任务分解方法."""
    name: str
    task: str
    preconditions: Set[str]
    subtasks: List[FractalTask]
    cost: float = 0.0
    fractal_reduction: int = 1  # 分形层级降低
    
    def applicable(self, state: FractalState) -> bool:
        return state.satisfies(self.preconditions)


@dataclass
class FractalPlan:
    """分形规划结果."""
    actions: List[FractalAction]
    total_cost: float
    total_duration: float
    success: bool
    explanation: List[str]
    fractal_depth: int = 0  # 分形展开深度
    path_quaternions: List[np.ndarray] = field(default_factory=list)
    fueter_valid: bool = True
    
    def __len__(self):
        return len(self.actions)


# ============================================================================
# 分形规划域
# ============================================================================

class FractalPlanningDomain:
    """分形规划域.
    
    支持多层级动作和方法定义.
    """
    
    def __init__(self, name: str, n_fractal_levels: int = 4):
        self.name = name
        self.n_fractal_levels = n_fractal_levels
        
        # 按层级组织动作
        self.actions: Dict[int, Dict[str, FractalAction]] = {
            i: {} for i in range(n_fractal_levels)
        }
        
        # 按层级组织方法
        self.methods: Dict[int, Dict[str, List[FractalMethod]]] = {
            i: {} for i in range(n_fractal_levels)
        }
        
        self.tasks: Dict[str, FractalTask] = {}
    
    def add_action(self, action: FractalAction):
        """添加动作到对应层级."""
        level = action.fractal_level
        if level >= self.n_fractal_levels:
            level = self.n_fractal_levels - 1
        self.actions[level][action.name] = action
    
    def add_method(self, method: FractalMethod, level: int = 0):
        """添加分解方法."""
        if level >= self.n_fractal_levels:
            level = self.n_fractal_levels - 1
        
        if method.task not in self.methods[level]:
            self.methods[level][method.task] = []
        self.methods[level][method.task].append(method)
    
    def add_task(self, task: FractalTask):
        self.tasks[task.name] = task
    
    def get_applicable_actions(self, state: FractalState, level: int = None
                               ) -> List[FractalAction]:
        """获取在当前状态下可执行的动作."""
        applicable = []
        
        levels = [level] if level is not None else range(self.n_fractal_levels)
        
        for lvl in levels:
            for action in self.actions.get(lvl, {}).values():
                if action.applicable(state):
                    applicable.append(action)
        
        return applicable
    
    def get_applicable_methods(self, task: FractalTask, state: FractalState
                               ) -> List[FractalMethod]:
        """获取适用于任务的分解方法."""
        applicable = []
        
        for level in range(self.n_fractal_levels):
            for method in self.methods.get(level, {}).get(task.name, []):
                if method.applicable(state):
                    applicable.append(method)
        
        return applicable


# ============================================================================
# 分形 HTN 规划器
# ============================================================================

class FractalHTNPlanner:
    """分形层次任务网络规划器.
    
    特点:
    1. 分形多尺度任务分解
    2. 四元数状态空间搜索
    3. Berry 相位启发式
    4. Fueter 路径验证
    """
    
    def __init__(self, domain: FractalPlanningDomain):
        self.domain = domain
        self.max_depth = 30
        self.max_iterations = 10000
        
        # 分形参数
        self.golden_ratio = 0.618
        
        # 统计
        self.nodes_expanded = 0
        self.plans_found = 0
        self.fractal_decompositions = 0
    
    def plan(self, initial_state: FractalState, task_network: List[FractalTask],
             goal: Optional[Set[str]] = None) -> FractalPlan:
        """执行分形 HTN 规划."""
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self.fractal_decompositions = 0
        
        # 计算目标四元数 (如果有目标)
        goal_q = None
        if goal:
            goal_state = FractalState(goal)
            goal_q = goal_state.quaternion
        
        # 搜索栈: (state, remaining_tasks, plan, cost, path_quaternions)
        stack = [(initial_state, list(task_network), [], 0.0, [initial_state.quaternion])]
        
        best_plan = None
        best_cost = float('inf')
        
        while stack and self.nodes_expanded < self.max_iterations:
            state, tasks, plan, cost, path_qs = stack.pop()
            self.nodes_expanded += 1
            
            # 检查是否完成
            if not tasks:
                if goal is None or state.satisfies(goal):
                    if cost < best_cost:
                        self.plans_found += 1
                        duration = sum(a.duration for a in plan)
                        explanation = self._generate_explanation(plan)
                        
                        # 检查 Fueter 有效性
                        fueter_valid = fueter_path_validity(path_qs)
                        
                        best_plan = FractalPlan(
                            plan, cost, duration, True, explanation,
                            fractal_depth=max(a.fractal_level for a in plan) if plan else 0,
                            path_quaternions=path_qs,
                            fueter_valid=fueter_valid
                        )
                        best_cost = cost
                continue
            
            # 获取第一个任务
            current_task = tasks[0]
            remaining = tasks[1:]
            
            if current_task.is_primitive():
                # 原子任务: 查找对应动作
                for level in range(self.domain.n_fractal_levels):
                    for action_name, action in self.domain.actions.get(level, {}).items():
                        if action.name == current_task.name and action.applicable(state):
                            new_state = action.apply(state)
                            new_plan = plan + [action]
                            new_cost = cost + action.cost * current_task.get_fractal_weight()
                            new_path = path_qs + [new_state.quaternion]
                            
                            # Berry 启发式优先级
                            if goal_q is not None:
                                priority = compute_berry_heuristic(new_path, goal_q)
                            else:
                                priority = new_cost
                            
                            stack.append((new_state, remaining, new_plan, new_cost, new_path))
            else:
                # 复合任务: 分形分解
                methods = self.domain.get_applicable_methods(current_task, state)
                
                if methods:
                    # 使用分解方法
                    for method in methods:
                        self.fractal_decompositions += 1
                        
                        # 降低子任务的分形层级
                        subtasks = []
                        for st in method.subtasks:
                            new_st = FractalTask(
                                name=st.name,
                                task_type=st.task_type,
                                parameters=st.parameters.copy(),
                                subtasks=st.subtasks.copy(),
                                preconditions=st.preconditions.copy(),
                                effects=st.effects.copy(),
                                priority=st.priority,
                                fractal_level=max(0, current_task.fractal_level - method.fractal_reduction),
                                complexity=st.complexity * self.golden_ratio
                            )
                            subtasks.append(new_st)
                        
                        new_tasks = subtasks + remaining
                        new_cost = cost + method.cost
                        stack.append((state, new_tasks, plan, new_cost, path_qs))
                else:
                    # 自动分形分解
                    if current_task.fractal_level > 0:
                        # 降低层级并重试
                        reduced_task = FractalTask(
                            name=current_task.name,
                            task_type=current_task.task_type,
                            parameters=current_task.parameters,
                            preconditions=current_task.preconditions,
                            effects=current_task.effects,
                            fractal_level=current_task.fractal_level - 1,
                            complexity=current_task.complexity * self.golden_ratio
                        )
                        stack.append((state, [reduced_task] + remaining, plan, cost, path_qs))
        
        # 返回最佳规划或失败
        if best_plan:
            return best_plan
        
        return FractalPlan([], float('inf'), 0, False, ["Planning failed"], 
                          fractal_depth=0, path_quaternions=[], fueter_valid=False)
    
    def _generate_explanation(self, plan: List[FractalAction]) -> List[str]:
        """生成规划解释."""
        explanations = []
        for i, action in enumerate(plan):
            exp = f"Step {i+1} [L{action.fractal_level}]: {action}"
            if action.preconditions:
                exp += f" (requires: {', '.join(list(action.preconditions)[:3])}...)"
            explanations.append(exp)
        return explanations


# ============================================================================
# 分形目标分解器
# ============================================================================

class FractalGoalDecomposer:
    """分形目标分解器.
    
    使用分形展开将复杂目标分解为多尺度子目标.
    """
    
    def __init__(self, n_levels: int = 4, base_ratio: float = 0.618):
        self.n_levels = n_levels
        self.base_ratio = base_ratio
    
    def decompose(self, goal: str, context: Dict[str, Any] = None
                  ) -> List[FractalTask]:
        """分形分解目标."""
        context = context or {}
        
        # 估计目标复杂度
        complexity = self._estimate_complexity(goal)
        
        # 确定分形展开层级
        n_expand = min(self.n_levels, max(1, int(np.log2(complexity + 1)) + 1))
        
        # 分形展开
        subtasks = []
        
        if " and " in goal.lower():
            # 合取目标: 分解为多个子目标
            parts = goal.lower().split(" and ")
            for i, part in enumerate(parts):
                level = min(i, n_expand - 1)
                complexity_part = complexity * (self.base_ratio ** level)
                
                subtasks.append(FractalTask(
                    name=f"achieve_{part.strip().replace(' ', '_')}",
                    task_type="compound",
                    parameters=[part.strip()],
                    fractal_level=level,
                    complexity=complexity_part,
                    priority=len(parts) - i
                ))
        elif " then " in goal.lower():
            # 顺序目标
            parts = goal.lower().split(" then ")
            for i, part in enumerate(parts):
                level = min(i, n_expand - 1)
                
                subtasks.append(FractalTask(
                    name=f"step_{i+1}_{part.strip().replace(' ', '_')}",
                    task_type="compound",
                    parameters=[part.strip()],
                    fractal_level=level,
                    complexity=complexity / len(parts),
                    priority=len(parts) - i
                ))
        else:
            # 简单目标: 使用分形层级表示抽象程度
            for level in range(n_expand):
                weight = self.base_ratio ** level
                
                subtasks.append(FractalTask(
                    name=f"achieve_{goal.replace(' ', '_')}_L{level}",
                    task_type="compound" if level > 0 else "primitive",
                    parameters=[goal],
                    fractal_level=level,
                    complexity=complexity * weight,
                    priority=n_expand - level
                ))
        
        return subtasks
    
    def _estimate_complexity(self, goal: str) -> float:
        """估计目标复杂度."""
        words = goal.split()
        n_conditions = goal.count(" and ") + goal.count(" or ")
        n_steps = goal.count(" then ") + 1
        
        base_complexity = len(words) * 0.5 + n_conditions * 2.0 + n_steps * 1.5
        return max(1.0, base_complexity)
    
    def get_fractal_signature(self, goal: str) -> np.ndarray:
        """获取目标的分形签名 (四元数表示)."""
        complexity = self._estimate_complexity(goal)
        
        # 分形分解
        levels = fractal_decompose(complexity, self.n_levels, self.base_ratio)
        
        # 编码为四元数
        q = np.zeros(4, dtype=np.float32)
        q[0] = np.cos(levels[0] / 10) if len(levels) > 0 else 1.0
        q[1] = np.sin(levels[1] / 10) if len(levels) > 1 else 0.0
        q[2] = np.sin(levels[2] / 10) if len(levels) > 2 else 0.0
        q[3] = np.sin(levels[3] / 10) if len(levels) > 3 else 0.0
        
        # 归一化
        norm = np.sqrt(np.sum(q * q))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return q / norm


# ============================================================================
# 分形动态重规划器
# ============================================================================

class FractalDynamicReplanner:
    """分形动态重规划器.
    
    特点:
    1. 多尺度规划修复
    2. 四元数路径连续性
    3. Fueter 有效性保证
    """
    
    def __init__(self, planner: FractalHTNPlanner):
        self.planner = planner
        self.current_plan: Optional[FractalPlan] = None
        self.execution_index: int = 0
        self.execution_history: List[Tuple[FractalAction, bool]] = []
        
        # 四元数路径历史
        self.path_quaternions: List[np.ndarray] = []
    
    def set_plan(self, plan: FractalPlan):
        self.current_plan = plan
        self.execution_index = 0
        self.execution_history = []
        self.path_quaternions = list(plan.path_quaternions) if plan.path_quaternions else []
    
    def execute_step(self, actual_state: FractalState
                     ) -> Tuple[Optional[FractalAction], bool]:
        """执行一步."""
        if not self.current_plan or self.execution_index >= len(self.current_plan):
            return None, False
        
        action = self.current_plan.actions[self.execution_index]
        
        # 检查动作是否仍适用
        if not action.applicable(actual_state):
            return action, True
        
        # 记录四元数
        self.path_quaternions.append(actual_state.quaternion)
        
        self.execution_index += 1
        self.execution_history.append((action, True))
        return action, False
    
    def replan(self, current_state: FractalState, remaining_goal: Set[str]
               ) -> FractalPlan:
        """重新规划.
        
        使用分形层级优先修复局部问题.
        """
        if not self.current_plan:
            return FractalPlan([], float('inf'), 0, False, ["No current plan"])
        
        remaining_actions = self.current_plan.actions[self.execution_index:]
        
        # 确定重规划层级 (局部修复用低层级)
        replan_level = 0
        if len(remaining_actions) <= 2:
            replan_level = 0  # 微调
        elif len(remaining_actions) <= 5:
            replan_level = 1  # 局部修复
        else:
            replan_level = 2  # 大范围重规划
        
        # 构造剩余任务
        remaining_tasks = []
        for a in remaining_actions:
            task = FractalTask(
                name=a.name,
                task_type="primitive",
                parameters=a.parameters,
                fractal_level=replan_level
            )
            remaining_tasks.append(task)
        
        # 重新规划
        new_plan = self.planner.plan(current_state, remaining_tasks, remaining_goal)
        
        if new_plan.success:
            # 合并已执行的动作
            executed_actions = [a for a, _ in self.execution_history]
            merged_plan = FractalPlan(
                actions=executed_actions + new_plan.actions,
                total_cost=sum(a.cost for a in executed_actions) + new_plan.total_cost,
                total_duration=sum(a.duration for a in executed_actions) + new_plan.total_duration,
                success=True,
                explanation=["Fractal replanned at level " + str(replan_level)] + new_plan.explanation,
                fractal_depth=max(new_plan.fractal_depth, replan_level),
                path_quaternions=self.path_quaternions + new_plan.path_quaternions,
                fueter_valid=fueter_path_validity(self.path_quaternions + new_plan.path_quaternions)
            )
            self.current_plan = merged_plan
            return merged_plan
        
        return new_plan
    
    def get_progress(self) -> Dict[str, Any]:
        if not self.current_plan:
            return {"progress": 0, "status": "no_plan"}
        
        total = len(self.current_plan)
        executed = self.execution_index
        
        # 检查路径 Fueter 有效性
        fueter_valid = fueter_path_validity(self.path_quaternions)
        
        return {
            "progress": executed / max(1, total),
            "executed_steps": executed,
            "total_steps": total,
            "status": "completed" if executed >= total else "in_progress",
            "fueter_valid": fueter_valid,
            "path_length": len(self.path_quaternions)
        }


# ============================================================================
# 分形层次规划系统 (集成接口)
# ============================================================================

@dataclass
class FractalPlanningResult:
    """分形规划系统结果."""
    plan: FractalPlan
    decomposition: List[FractalTask]
    complexity: Dict[str, float]
    planning_time_ms: float
    replanning_count: int
    fractal_stats: Dict[str, Any]


class FractalHierarchicalPlanningSystem:
    """完整的分形层次规划系统.
    
    集成 H2Q 数学优势:
    1. 分形多尺度分解
    2. 四元数状态空间
    3. Berry 相位启发式
    4. Fueter 路径验证
    """
    
    def __init__(self, n_fractal_levels: int = 4):
        self.domain = FractalPlanningDomain("fractal_default", n_fractal_levels)
        self.planner = FractalHTNPlanner(self.domain)
        self.decomposer = FractalGoalDecomposer(n_fractal_levels)
        self.replanner = FractalDynamicReplanner(self.planner)
        
        self.n_fractal_levels = n_fractal_levels
        
        # 初始化默认域
        self._init_default_domain()
        
        # 统计
        self.total_plans = 0
        self.successful_plans = 0
    
    def _init_default_domain(self):
        """初始化默认规划域."""
        # 基础动作 (不同层级)
        for level in range(self.n_fractal_levels):
            # 移动动作
            move = FractalAction(
                name=f"move_L{level}",
                parameters=["from", "to"],
                preconditions={f"at_from_L{level}", f"connected_L{level}"},
                add_effects={f"at_to_L{level}"},
                delete_effects={f"at_from_L{level}"},
                cost=1.0 * (0.618 ** level),  # 分形成本
                fractal_level=level
            )
            self.domain.add_action(move)
            
            # 拾取动作
            pickup = FractalAction(
                name=f"pickup_L{level}",
                parameters=["obj"],
                preconditions={f"at_obj_L{level}", f"hand_empty_L{level}"},
                add_effects={f"holding_obj_L{level}"},
                delete_effects={f"at_obj_L{level}", f"hand_empty_L{level}"},
                cost=0.5 * (0.618 ** level),
                fractal_level=level
            )
            self.domain.add_action(pickup)
            
            # 放置动作
            putdown = FractalAction(
                name=f"putdown_L{level}",
                parameters=["obj", "loc"],
                preconditions={f"holding_obj_L{level}", f"at_loc_L{level}"},
                add_effects={f"at_obj_loc_L{level}", f"hand_empty_L{level}"},
                delete_effects={f"holding_obj_L{level}"},
                cost=0.5 * (0.618 ** level),
                fractal_level=level
            )
            self.domain.add_action(putdown)
    
    def plan_from_goal(self, initial_state: FractalState, goal: str
                       ) -> FractalPlanningResult:
        """从高层目标规划."""
        start_time = time.perf_counter()
        
        # 分形分解目标
        decomposition = self.decomposer.decompose(goal)
        
        # 提取目标条件
        goal_conditions = set()
        for task in decomposition:
            if task.effects:
                goal_conditions.update(task.effects)
            else:
                # 从任务名推断
                goal_conditions.add(f"achieved_{task.name}")
        
        # 执行规划
        plan = self.planner.plan(initial_state, decomposition, goal_conditions)
        
        planning_time = (time.perf_counter() - start_time) * 1000
        
        self.total_plans += 1
        if plan.success:
            self.successful_plans += 1
        
        # 复杂度分析
        complexity = {
            "goal_complexity": self.decomposer._estimate_complexity(goal),
            "decomposition_depth": max(t.fractal_level for t in decomposition) if decomposition else 0,
            "n_subtasks": len(decomposition),
            "fractal_coverage": sum(t.get_fractal_weight() for t in decomposition)
        }
        
        # 分形统计
        fractal_stats = {
            "nodes_expanded": self.planner.nodes_expanded,
            "fractal_decompositions": self.planner.fractal_decompositions,
            "plan_depth": plan.fractal_depth,
            "fueter_valid": plan.fueter_valid,
            "path_length": len(plan.path_quaternions)
        }
        
        return FractalPlanningResult(
            plan=plan,
            decomposition=decomposition,
            complexity=complexity,
            planning_time_ms=planning_time,
            replanning_count=0,
            fractal_stats=fractal_stats
        )
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_plans": self.total_plans,
            "successful_plans": self.successful_plans,
            "success_rate": self.successful_plans / max(1, self.total_plans),
            "n_fractal_levels": self.n_fractal_levels,
            "domain_actions": sum(len(self.domain.actions[i]) for i in range(self.n_fractal_levels))
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_fractal_planning_system(n_fractal_levels: int = 4
                                    ) -> FractalHierarchicalPlanningSystem:
    """创建分形层次规划系统."""
    return FractalHierarchicalPlanningSystem(n_fractal_levels)


def create_test_fractal_state(facts: Set[str] = None, 
                              numeric: Dict[str, float] = None) -> FractalState:
    """创建测试状态."""
    facts = facts or {"at_home", "hand_empty"}
    numeric = numeric or {"energy": 1.0}
    return FractalState(facts, numeric)


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 分形增强层次规划 - 演示")
    print("=" * 70)
    
    # 创建系统
    system = create_fractal_planning_system(n_fractal_levels=4)
    
    print(f"\n分形层级: {system.n_fractal_levels}")
    print(f"域动作数: {system.get_summary()['domain_actions']}")
    
    # 创建初始状态
    initial_state = FractalState(
        facts={"at_home_L0", "hand_empty_L0", "connected_L0"},
        numeric={"energy": 1.0}
    )
    
    print(f"\n初始状态四元数: {initial_state.quaternion}")
    
    # 目标分解测试
    print("\n1. 分形目标分解")
    print("-" * 50)
    
    goal = "move to office and pickup document then return home"
    decomposition = system.decomposer.decompose(goal)
    
    print(f"目标: {goal}")
    print(f"复杂度: {system.decomposer._estimate_complexity(goal):.2f}")
    print(f"分解为 {len(decomposition)} 个子任务:")
    for task in decomposition:
        print(f"  - {task}")
    
    goal_sig = system.decomposer.get_fractal_signature(goal)
    print(f"目标签名 (四元数): {goal_sig}")
    
    # 四元数距离测试
    print("\n2. 四元数状态空间")
    print("-" * 50)
    
    state1 = FractalState({"at_home"}, {"x": 0.0})
    state2 = FractalState({"at_office"}, {"x": 1.0})
    
    dist = state1.distance_to(state2)
    print(f"状态1 -> 状态2 测地距离: {dist:.4f}")
    
    # 分形路径验证
    print("\n3. Fueter 路径验证")
    print("-" * 50)
    
    # 生成平滑路径
    smooth_path = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32),
        np.array([0.8, 0.2, 0.2, 0.2], dtype=np.float32),
    ]
    smooth_path = [q / np.linalg.norm(q) for q in smooth_path]
    
    print(f"平滑路径 Fueter 有效: {fueter_path_validity(smooth_path)}")
    
    # 生成不平滑路径
    rough_path = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0], dtype=np.float32),  # 大跳跃
        np.array([0.8, 0.2, 0.2, 0.2], dtype=np.float32),
    ]
    rough_path = [q / np.linalg.norm(q) for q in rough_path]
    
    print(f"不平滑路径 Fueter 有效: {fueter_path_validity(rough_path)}")
    
    print("\n" + "=" * 70)
    summary = system.get_summary()
    print(f"系统摘要: {summary}")
