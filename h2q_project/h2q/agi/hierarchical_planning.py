"""H2Q 层次化规划系统 (Hierarchical Planning System).

实现AGI核心能力：
1. 层次任务网络 (HTN)
2. 目标分解与子目标生成
3. 动态重规划
4. 不确定性处理

参考文献:
- Erol et al., "HTN Planning" (1994)
- Kaelbling & Lozano-Pérez, "Hierarchical Task and Motion Planning" (2011)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import deque
from enum import Enum
import heapq
import time
import copy


# ============================================================================
# 规划原语
# ============================================================================

class TaskStatus(Enum):
    """任务状态."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class State:
    """世界状态."""
    facts: Set[str]  # 状态谓词集合
    numeric: Dict[str, float] = field(default_factory=dict)
    
    def satisfies(self, conditions: Set[str]) -> bool:
        """检查状态是否满足条件."""
        return conditions.issubset(self.facts)
    
    def copy(self) -> 'State':
        """深拷贝状态."""
        return State(self.facts.copy(), self.numeric.copy())
    
    def __hash__(self):
        return hash(frozenset(self.facts))


@dataclass
class Action:
    """原子动作."""
    name: str
    parameters: List[str]
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str]
    cost: float = 1.0
    duration: float = 1.0
    
    def applicable(self, state: State) -> bool:
        """检查动作是否可在当前状态执行."""
        return state.satisfies(self.preconditions)
    
    def apply(self, state: State) -> State:
        """应用动作到状态."""
        new_state = state.copy()
        new_state.facts -= self.delete_effects
        new_state.facts |= self.add_effects
        return new_state
    
    def __str__(self):
        params = ", ".join(self.parameters)
        return f"{self.name}({params})"


@dataclass
class Task:
    """任务 (可以是原子或复合)."""
    name: str
    task_type: str  # "primitive" or "compound"
    parameters: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    priority: int = 0
    deadline: Optional[float] = None
    
    def is_primitive(self) -> bool:
        return self.task_type == "primitive"
    
    def __str__(self):
        params = ", ".join(self.parameters)
        return f"{self.name}({params})"


@dataclass
class Method:
    """任务分解方法."""
    name: str
    task: str  # 适用的任务名
    preconditions: Set[str]
    subtasks: List[Task]
    cost: float = 0.0
    
    def applicable(self, state: State) -> bool:
        return state.satisfies(self.preconditions)


@dataclass
class Plan:
    """规划结果."""
    actions: List[Action]
    total_cost: float
    total_duration: float
    success: bool
    explanation: List[str]
    
    def __len__(self):
        return len(self.actions)


# ============================================================================
# 规划域
# ============================================================================

class PlanningDomain:
    """规划域: 定义动作和方法."""
    
    def __init__(self, name: str):
        self.name = name
        self.actions: Dict[str, Action] = {}
        self.methods: Dict[str, List[Method]] = {}  # task_name -> methods
        self.tasks: Dict[str, Task] = {}
    
    def add_action(self, action: Action):
        """添加原子动作."""
        self.actions[action.name] = action
    
    def add_method(self, method: Method):
        """添加分解方法."""
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)
    
    def add_task(self, task: Task):
        """添加任务定义."""
        self.tasks[task.name] = task
    
    def get_applicable_methods(self, task: Task, state: State) -> List[Method]:
        """获取在当前状态下适用于任务的方法."""
        methods = self.methods.get(task.name, [])
        return [m for m in methods if m.applicable(state)]
    
    def get_action(self, name: str) -> Optional[Action]:
        """获取动作."""
        return self.actions.get(name)


# ============================================================================
# HTN 规划器
# ============================================================================

class HTNPlanner:
    """层次任务网络规划器."""
    
    def __init__(self, domain: PlanningDomain):
        self.domain = domain
        self.max_depth = 20
        self.max_iterations = 10000
        
        # 统计
        self.nodes_expanded = 0
        self.plans_found = 0
    
    def plan(self, initial_state: State, task_network: List[Task],
             goal: Optional[Set[str]] = None) -> Plan:
        """执行 HTN 规划.
        
        Args:
            initial_state: 初始状态
            task_network: 初始任务网络
            goal: 可选的目标条件
        """
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        
        # 搜索栈: (state, remaining_tasks, plan_so_far, cost)
        stack = [(initial_state, list(task_network), [], 0.0)]
        
        while stack and self.nodes_expanded < self.max_iterations:
            state, tasks, plan, cost = stack.pop()
            self.nodes_expanded += 1
            
            # 检查是否完成
            if not tasks:
                # 验证目标
                if goal is None or state.satisfies(goal):
                    self.plans_found += 1
                    duration = sum(a.duration for a in plan)
                    explanation = self._generate_explanation(plan)
                    return Plan(plan, cost, duration, True, explanation)
                continue
            
            # 获取第一个任务
            current_task = tasks[0]
            remaining = tasks[1:]
            
            if current_task.is_primitive():
                # 原子任务: 直接执行
                action = self.domain.get_action(current_task.name)
                if action and action.applicable(state):
                    new_state = action.apply(state)
                    new_plan = plan + [action]
                    new_cost = cost + action.cost
                    stack.append((new_state, remaining, new_plan, new_cost))
            else:
                # 复合任务: 分解
                methods = self.domain.get_applicable_methods(current_task, state)
                
                for method in methods:
                    new_tasks = list(method.subtasks) + remaining
                    new_cost = cost + method.cost
                    stack.append((state, new_tasks, plan, new_cost))
        
        # 规划失败
        return Plan([], float('inf'), 0, False, ["Planning failed"])
    
    def _generate_explanation(self, plan: List[Action]) -> List[str]:
        """生成规划解释."""
        explanations = []
        for i, action in enumerate(plan):
            exp = f"Step {i+1}: {action}"
            if action.preconditions:
                exp += f" (requires: {', '.join(action.preconditions)})"
            explanations.append(exp)
        return explanations


# ============================================================================
# 目标分解器
# ============================================================================

class GoalDecomposer:
    """目标分解器: 将高层目标分解为子目标."""
    
    def __init__(self):
        self.decomposition_rules: Dict[str, Callable] = {}
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认分解规则."""
        # 序列分解
        self.decomposition_rules["sequence"] = lambda g, deps: [
            {"goal": sub, "type": "achieve"} for sub in deps
        ]
        
        # 并行分解
        self.decomposition_rules["parallel"] = lambda g, deps: [
            {"goal": sub, "type": "achieve", "parallel": True} for sub in deps
        ]
        
        # 条件分解
        self.decomposition_rules["conditional"] = lambda g, cond, then_g, else_g: [
            {"goal": then_g if cond else else_g, "type": "achieve"}
        ]
    
    def decompose(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """分解目标为子任务."""
        # 简化实现: 基于模式匹配
        subtasks = []
        
        # 识别目标模式
        if "and" in goal.lower():
            # 合取目标: 分解为多个子目标
            parts = goal.lower().split("and")
            for part in parts:
                subtasks.append(Task(
                    name=f"achieve_{part.strip().replace(' ', '_')}",
                    task_type="compound",
                    parameters=[part.strip()],
                    priority=1
                ))
        elif "then" in goal.lower():
            # 顺序目标
            parts = goal.lower().split("then")
            for i, part in enumerate(parts):
                subtasks.append(Task(
                    name=f"step_{i+1}_{part.strip().replace(' ', '_')}",
                    task_type="compound",
                    parameters=[part.strip()],
                    priority=len(parts) - i  # 先执行的优先级高
                ))
        else:
            # 原子目标
            subtasks.append(Task(
                name=f"achieve_{goal.replace(' ', '_')}",
                task_type="primitive",
                parameters=[goal]
            ))
        
        return subtasks
    
    def estimate_complexity(self, goal: str) -> Dict[str, float]:
        """估计目标复杂度."""
        # 简单启发式
        words = goal.split()
        n_conditions = goal.count("and") + goal.count("or")
        n_steps = goal.count("then") + 1
        
        return {
            "word_count": len(words),
            "condition_count": n_conditions,
            "step_count": n_steps,
            "estimated_difficulty": len(words) * 0.1 + n_conditions * 0.3 + n_steps * 0.2
        }


# ============================================================================
# 动态重规划器
# ============================================================================

class DynamicReplanner:
    """动态重规划器: 处理执行过程中的变化."""
    
    def __init__(self, planner: HTNPlanner):
        self.planner = planner
        self.current_plan: Optional[Plan] = None
        self.execution_index: int = 0
        self.execution_history: List[Tuple[Action, bool]] = []
    
    def set_plan(self, plan: Plan):
        """设置当前规划."""
        self.current_plan = plan
        self.execution_index = 0
        self.execution_history = []
    
    def execute_step(self, actual_state: State) -> Tuple[Optional[Action], bool]:
        """执行一步并检查是否需要重规划.
        
        Returns:
            (action, needs_replanning)
        """
        if not self.current_plan or self.execution_index >= len(self.current_plan):
            return None, False
        
        action = self.current_plan.actions[self.execution_index]
        
        # 检查动作是否仍然适用
        if not action.applicable(actual_state):
            # 需要重规划
            return action, True
        
        self.execution_index += 1
        self.execution_history.append((action, True))
        return action, False
    
    def replan(self, current_state: State, remaining_goal: Set[str]) -> Plan:
        """重新规划."""
        # 获取剩余任务
        if not self.current_plan:
            return Plan([], float('inf'), 0, False, ["No current plan"])
        
        remaining_actions = self.current_plan.actions[self.execution_index:]
        
        # 将剩余动作转换为任务
        remaining_tasks = [
            Task(name=a.name, task_type="primitive", parameters=a.parameters)
            for a in remaining_actions
        ]
        
        # 重新规划
        new_plan = self.planner.plan(current_state, remaining_tasks, remaining_goal)
        
        if new_plan.success:
            # 合并已执行的动作和新规划
            executed_actions = [a for a, _ in self.execution_history]
            merged_plan = Plan(
                actions=executed_actions + new_plan.actions,
                total_cost=sum(a.cost for a in executed_actions) + new_plan.total_cost,
                total_duration=sum(a.duration for a in executed_actions) + new_plan.total_duration,
                success=True,
                explanation=["Replanned successfully"] + new_plan.explanation
            )
            self.current_plan = merged_plan
            return merged_plan
        
        return new_plan
    
    def get_progress(self) -> Dict[str, Any]:
        """获取执行进度."""
        if not self.current_plan:
            return {"progress": 0, "status": "no_plan"}
        
        total = len(self.current_plan)
        executed = self.execution_index
        
        return {
            "progress": executed / max(1, total),
            "executed_steps": executed,
            "total_steps": total,
            "status": "completed" if executed >= total else "in_progress"
        }


# ============================================================================
# 层次化规划系统
# ============================================================================

@dataclass
class PlanningResult:
    """规划系统结果."""
    plan: Plan
    decomposition: List[Task]
    complexity: Dict[str, float]
    planning_time_ms: float
    replanning_count: int


class HierarchicalPlanningSystem:
    """完整的层次化规划系统."""
    
    def __init__(self):
        self.domain = PlanningDomain("default")
        self.planner = HTNPlanner(self.domain)
        self.decomposer = GoalDecomposer()
        self.replanner = DynamicReplanner(self.planner)
        
        # 初始化默认域
        self._init_default_domain()
        
        # 统计
        self.total_plans = 0
        self.successful_plans = 0
    
    def _init_default_domain(self):
        """初始化默认规划域."""
        # 基础动作
        move = Action(
            name="move",
            parameters=["from", "to"],
            preconditions={"at_from", "connected_from_to"},
            add_effects={"at_to"},
            delete_effects={"at_from"},
            cost=1.0
        )
        self.domain.add_action(move)
        
        pick_up = Action(
            name="pick_up",
            parameters=["object", "location"],
            preconditions={"at_location", "object_at_location", "hand_empty"},
            add_effects={"holding_object"},
            delete_effects={"object_at_location", "hand_empty"},
            cost=1.0
        )
        self.domain.add_action(pick_up)
        
        put_down = Action(
            name="put_down",
            parameters=["object", "location"],
            preconditions={"at_location", "holding_object"},
            add_effects={"object_at_location", "hand_empty"},
            delete_effects={"holding_object"},
            cost=1.0
        )
        self.domain.add_action(put_down)
        
        # 复合任务和方法
        transport_task = Task(
            name="transport",
            task_type="compound",
            parameters=["object", "from", "to"]
        )
        self.domain.add_task(transport_task)
        
        transport_method = Method(
            name="transport_by_carrying",
            task="transport",
            preconditions=set(),
            subtasks=[
                Task("move", "primitive", ["start", "from"]),
                Task("pick_up", "primitive", ["object", "from"]),
                Task("move", "primitive", ["from", "to"]),
                Task("put_down", "primitive", ["object", "to"])
            ]
        )
        self.domain.add_method(transport_method)
    
    def add_action(self, name: str, preconditions: List[str], 
                   add_effects: List[str], delete_effects: List[str],
                   cost: float = 1.0, duration: float = 1.0):
        """添加动作到域."""
        action = Action(
            name=name,
            parameters=[],
            preconditions=set(preconditions),
            add_effects=set(add_effects),
            delete_effects=set(delete_effects),
            cost=cost,
            duration=duration
        )
        self.domain.add_action(action)
    
    def add_method(self, name: str, task: str, preconditions: List[str],
                   subtask_names: List[str], cost: float = 0.0):
        """添加方法到域."""
        subtasks = [
            Task(name=sn, task_type="primitive") for sn in subtask_names
        ]
        method = Method(
            name=name,
            task=task,
            preconditions=set(preconditions),
            subtasks=subtasks,
            cost=cost
        )
        self.domain.add_method(method)
    
    def plan_for_goal(self, goal: str, initial_facts: List[str],
                      goal_facts: Optional[List[str]] = None) -> PlanningResult:
        """为目标生成规划."""
        start_time = time.perf_counter()
        self.total_plans += 1
        
        # 1. 分解目标
        decomposition = self.decomposer.decompose(goal, {})
        complexity = self.decomposer.estimate_complexity(goal)
        
        # 2. 构建初始状态
        initial_state = State(set(initial_facts))
        
        # 3. 执行 HTN 规划
        goal_set = set(goal_facts) if goal_facts else None
        plan = self.planner.plan(initial_state, decomposition, goal_set)
        
        if plan.success:
            self.successful_plans += 1
        
        planning_time = (time.perf_counter() - start_time) * 1000
        
        return PlanningResult(
            plan=plan,
            decomposition=decomposition,
            complexity=complexity,
            planning_time_ms=planning_time,
            replanning_count=0
        )
    
    def execute_with_monitoring(self, plan: Plan, 
                                 state_observer: Callable[[], State],
                                 action_executor: Callable[[Action], bool]) -> Dict[str, Any]:
        """监控执行规划."""
        self.replanner.set_plan(plan)
        results = []
        replanning_count = 0
        
        while True:
            current_state = state_observer()
            action, needs_replan = self.replanner.execute_step(current_state)
            
            if action is None:
                break
            
            if needs_replan:
                # 触发重规划
                remaining_goal = plan.actions[-1].add_effects if plan.actions else set()
                new_plan = self.replanner.replan(current_state, remaining_goal)
                replanning_count += 1
                
                if not new_plan.success:
                    results.append({"action": str(action), "status": "failed", "reason": "replan_failed"})
                    break
                
                continue
            
            # 执行动作
            success = action_executor(action)
            results.append({"action": str(action), "status": "success" if success else "failed"})
            
            if not success:
                break
        
        return {
            "execution_results": results,
            "progress": self.replanner.get_progress(),
            "replanning_count": replanning_count,
            "final_status": "completed" if self.replanner.get_progress()["status"] == "completed" else "failed"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        return {
            "total_plans": self.total_plans,
            "successful_plans": self.successful_plans,
            "success_rate": self.successful_plans / max(1, self.total_plans),
            "domain_actions": len(self.domain.actions),
            "domain_methods": sum(len(m) for m in self.domain.methods.values()),
            "planner_nodes_expanded": self.planner.nodes_expanded
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_planning_system() -> HierarchicalPlanningSystem:
    """创建层次化规划系统."""
    return HierarchicalPlanningSystem()


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 层次化规划系统 - 演示")
    print("=" * 70)
    
    system = create_planning_system()
    
    # 添加更多动作
    system.add_action(
        "open_door",
        preconditions=["at_door", "has_key"],
        add_effects=["door_open"],
        delete_effects=["door_closed"]
    )
    
    system.add_action(
        "get_key",
        preconditions=["at_key_location", "hand_empty"],
        add_effects=["has_key"],
        delete_effects=["key_at_location", "hand_empty"]
    )
    
    # 测试规划
    print("\n1. 简单目标规划")
    print("-" * 50)
    
    result = system.plan_for_goal(
        goal="get key and open door",
        initial_facts=["at_key_location", "hand_empty", "key_at_location", "at_door", "door_closed"],
        goal_facts=["door_open"]
    )
    
    print(f"目标: get key and open door")
    print(f"规划成功: {result.plan.success}")
    print(f"动作数: {len(result.plan)}")
    print(f"总成本: {result.plan.total_cost}")
    print(f"规划时间: {result.planning_time_ms:.2f} ms")
    print(f"复杂度估计: {result.complexity}")
    
    if result.plan.success:
        print("\n规划步骤:")
        for exp in result.plan.explanation:
            print(f"  {exp}")
    
    print("\n" + "=" * 70)
    print("统计信息:")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
