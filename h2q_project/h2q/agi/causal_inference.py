"""H2Q 因果推断模块 (Causal Inference Module).

实现AGI核心能力：
1. 结构因果模型 (SCM)
2. 因果发现 (PC算法变种)
3. do-演算干预效应估计
4. 反事实推理

参考文献:
- Pearl, "Causality" (2009)
- Peters et al., "Elements of Causal Inference" (2017)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import time


# ============================================================================
# 因果图结构
# ============================================================================

@dataclass
class CausalNode:
    """因果图节点."""
    name: str
    node_type: str = "endogenous"  # endogenous (内生) / exogenous (外生)
    domain: Optional[List[Any]] = None  # 变量取值域
    observed: bool = True  # 是否可观测
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, CausalNode) and self.name == other.name


@dataclass
class CausalEdge:
    """因果边: cause → effect."""
    cause: CausalNode
    effect: CausalNode
    strength: float = 1.0  # 因果强度
    mechanism: Optional[str] = None  # 因果机制描述


class CausalGraph:
    """因果图 (DAG)."""
    
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # 子节点集
        self.parents: Dict[str, Set[str]] = defaultdict(set)     # 父节点集
    
    def add_node(self, name: str, **kwargs) -> CausalNode:
        """添加节点."""
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name, **kwargs)
        return self.nodes[name]
    
    def add_edge(self, cause: str, effect: str, strength: float = 1.0, 
                 mechanism: str = None):
        """添加因果边."""
        cause_node = self.add_node(cause)
        effect_node = self.add_node(effect)
        
        edge = CausalEdge(cause_node, effect_node, strength, mechanism)
        self.edges.append(edge)
        
        self.adjacency[cause].add(effect)
        self.parents[effect].add(cause)
    
    def get_parents(self, node: str) -> Set[str]:
        """获取节点的父节点."""
        return self.parents.get(node, set())
    
    def get_children(self, node: str) -> Set[str]:
        """获取节点的子节点."""
        return self.adjacency.get(node, set())
    
    def get_ancestors(self, node: str) -> Set[str]:
        """获取节点的所有祖先."""
        ancestors = set()
        to_visit = list(self.get_parents(node))
        
        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))
        
        return ancestors
    
    def get_descendants(self, node: str) -> Set[str]:
        """获取节点的所有后代."""
        descendants = set()
        to_visit = list(self.get_children(node))
        
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))
        
        return descendants
    
    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """检查 X 和 Y 是否在给定 Z 下 d-分离."""
        # 简化实现: 使用 Bayes Ball 算法的变种
        
        # 找出所有从 X 到 Y 的路径
        def find_paths(start: str, end: str, path: List[str]) -> List[List[str]]:
            if start == end:
                return [path + [end]]
            
            paths = []
            for neighbor in self.get_children(start) | self.get_parents(start):
                if neighbor not in path:
                    paths.extend(find_paths(neighbor, end, path + [start]))
            return paths
        
        paths = find_paths(x, y, [])
        
        # 检查每条路径是否被阻断
        for path in paths:
            blocked = False
            for i in range(1, len(path) - 1):
                node = path[i]
                prev_node = path[i-1]
                next_node = path[i+1]
                
                # 检查路径类型
                is_collider = (prev_node in self.get_parents(node) and 
                              next_node in self.get_parents(node))
                
                if is_collider:
                    # 碰撞节点: 需要节点或其后代在 Z 中
                    node_descendants = self.get_descendants(node) | {node}
                    if not (node_descendants & z):
                        blocked = True
                        break
                else:
                    # 非碰撞节点: 节点在 Z 中则阻断
                    if node in z:
                        blocked = True
                        break
            
            if not blocked:
                return False  # 存在未阻断的路径
        
        return True  # 所有路径都被阻断
    
    def topological_sort(self) -> List[str]:
        """拓扑排序."""
        in_degree = {n: len(self.get_parents(n)) for n in self.nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result


# ============================================================================
# 结构因果模型 (SCM)
# ============================================================================

@dataclass
class StructuralEquation:
    """结构方程: X = f(Pa(X), U_X)."""
    variable: str
    parents: List[str]
    function: callable  # f(parent_values, noise) -> value
    noise_distribution: str = "gaussian"
    noise_params: Dict[str, float] = field(default_factory=lambda: {"mean": 0, "std": 0.1})


class StructuralCausalModel:
    """结构因果模型."""
    
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.equations: Dict[str, StructuralEquation] = {}
        self.data_cache: Dict[str, np.ndarray] = {}
    
    def add_equation(self, variable: str, parents: List[str], 
                     function: callable, noise_dist: str = "gaussian",
                     noise_params: Dict[str, float] = None):
        """添加结构方程."""
        self.equations[variable] = StructuralEquation(
            variable=variable,
            parents=parents,
            function=function,
            noise_distribution=noise_dist,
            noise_params=noise_params or {"mean": 0, "std": 0.1}
        )
    
    def sample_noise(self, variable: str, n_samples: int) -> np.ndarray:
        """采样噪声."""
        eq = self.equations.get(variable)
        if not eq:
            return np.random.randn(n_samples) * 0.1
        
        params = eq.noise_params
        if eq.noise_distribution == "gaussian":
            return np.random.normal(params.get("mean", 0), 
                                   params.get("std", 0.1), n_samples)
        elif eq.noise_distribution == "uniform":
            return np.random.uniform(params.get("low", -1), 
                                    params.get("high", 1), n_samples)
        else:
            return np.random.randn(n_samples) * 0.1
    
    def sample(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """从 SCM 采样观测数据."""
        data = {}
        
        # 按拓扑顺序采样
        order = self.graph.topological_sort()
        
        for var in order:
            eq = self.equations.get(var)
            noise = self.sample_noise(var, n_samples)
            
            if eq and eq.parents:
                parent_values = {p: data[p] for p in eq.parents if p in data}
                data[var] = eq.function(parent_values, noise)
            else:
                # 外生变量
                data[var] = noise
        
        self.data_cache = data
        return data
    
    def intervene(self, interventions: Dict[str, float], 
                  n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """执行 do-干预: do(X=x).
        
        干预切断 X 的父节点边，将 X 固定为 x。
        """
        data = {}
        order = self.graph.topological_sort()
        
        for var in order:
            if var in interventions:
                # 干预变量: 固定值
                data[var] = np.full(n_samples, interventions[var])
            else:
                eq = self.equations.get(var)
                noise = self.sample_noise(var, n_samples)
                
                if eq and eq.parents:
                    parent_values = {p: data[p] for p in eq.parents if p in data}
                    data[var] = eq.function(parent_values, noise)
                else:
                    data[var] = noise
        
        return data
    
    def counterfactual(self, observation: Dict[str, float], 
                       intervention: Dict[str, float],
                       target: str) -> float:
        """反事实推理.
        
        问题: 给定观测 O，如果我们 do(X=x)，Y 会是什么？
        
        步骤:
        1. Abduction: 从观测推断噪声
        2. Action: 应用干预
        3. Prediction: 使用推断的噪声预测
        """
        # 步骤 1: 推断噪声 (简化: 使用观测值减去预期)
        inferred_noise = {}
        order = self.graph.topological_sort()
        
        for var in order:
            if var in observation:
                eq = self.equations.get(var)
                if eq and eq.parents:
                    parent_values = {p: observation.get(p, 0) for p in eq.parents}
                    expected = eq.function(parent_values, np.array([0]))[0]
                    inferred_noise[var] = observation[var] - expected
                else:
                    inferred_noise[var] = observation[var]
        
        # 步骤 2 & 3: 应用干预并使用推断的噪声
        counterfactual_values = {}
        
        for var in order:
            if var in intervention:
                counterfactual_values[var] = intervention[var]
            else:
                eq = self.equations.get(var)
                noise = np.array([inferred_noise.get(var, 0)])
                
                if eq and eq.parents:
                    parent_values = {p: counterfactual_values.get(p, 0) 
                                    for p in eq.parents}
                    counterfactual_values[var] = eq.function(parent_values, noise)[0]
                else:
                    counterfactual_values[var] = noise[0]
        
        return counterfactual_values.get(target, 0)


# ============================================================================
# 因果发现 (简化 PC 算法)
# ============================================================================

class CausalDiscovery:
    """因果发现算法."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # 独立性检验显著性水平
    
    def discover(self, data: Dict[str, np.ndarray]) -> CausalGraph:
        """从数据发现因果结构 (简化版 PC 算法)."""
        variables = list(data.keys())
        n_vars = len(variables)
        
        # 初始化完全图
        graph = CausalGraph()
        for var in variables:
            graph.add_node(var)
        
        # 添加所有可能的边 (无向)
        undirected_edges = set()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                undirected_edges.add((variables[i], variables[j]))
        
        # 阶段 1: 删除边 (条件独立性检验)
        for d in range(n_vars):  # 条件集大小
            edges_to_remove = set()
            
            for edge in undirected_edges:
                x, y = edge
                other_vars = [v for v in variables if v not in edge]
                
                # 检查所有大小为 d 的条件集
                from itertools import combinations
                for cond_set in combinations(other_vars, min(d, len(other_vars))):
                    if self._conditional_independence_test(data, x, y, list(cond_set)):
                        edges_to_remove.add(edge)
                        break
            
            undirected_edges -= edges_to_remove
        
        # 阶段 2: 方向定向 (简化: 使用时间顺序启发式)
        # 假设变量按名称排序代表时间顺序
        sorted_vars = sorted(variables)
        
        for x, y in undirected_edges:
            if sorted_vars.index(x) < sorted_vars.index(y):
                graph.add_edge(x, y)
            else:
                graph.add_edge(y, x)
        
        return graph
    
    def _conditional_independence_test(self, data: Dict[str, np.ndarray],
                                       x: str, y: str, 
                                       cond: List[str]) -> bool:
        """条件独立性检验 (使用偏相关)."""
        if not cond:
            # 无条件: 使用皮尔逊相关
            corr = np.corrcoef(data[x], data[y])[0, 1]
            n = len(data[x])
            # Fisher z 变换
            z = 0.5 * np.log((1 + corr + 1e-10) / (1 - corr + 1e-10))
            se = 1 / np.sqrt(n - 3)
            p_value = 2 * (1 - self._norm_cdf(abs(z) / se))
            return p_value > self.alpha
        else:
            # 条件独立: 使用偏相关 (简化)
            # 残差法
            residuals_x = self._residualize(data[x], [data[c] for c in cond])
            residuals_y = self._residualize(data[y], [data[c] for c in cond])
            
            corr = np.corrcoef(residuals_x, residuals_y)[0, 1]
            n = len(data[x])
            z = 0.5 * np.log((1 + corr + 1e-10) / (1 - corr + 1e-10))
            se = 1 / np.sqrt(n - len(cond) - 3)
            p_value = 2 * (1 - self._norm_cdf(abs(z) / se))
            return p_value > self.alpha
    
    def _residualize(self, y: np.ndarray, X: List[np.ndarray]) -> np.ndarray:
        """通过回归获取残差."""
        if not X:
            return y
        
        X_matrix = np.column_stack(X)
        X_matrix = np.column_stack([np.ones(len(y)), X_matrix])
        
        try:
            beta = np.linalg.lstsq(X_matrix, y, rcond=None)[0]
            residuals = y - X_matrix @ beta
        except:
            residuals = y
        
        return residuals
    
    def _norm_cdf(self, x: float) -> float:
        """标准正态分布 CDF (近似)."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


# ============================================================================
# 因果推断引擎
# ============================================================================

@dataclass
class CausalQuery:
    """因果查询."""
    query_type: str  # "ate", "cate", "counterfactual", "mediation"
    treatment: str
    outcome: str
    conditioning: Optional[Dict[str, Any]] = None
    intervention_value: Optional[float] = None


@dataclass
class CausalResult:
    """因果推断结果."""
    query: CausalQuery
    estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    details: Dict[str, Any]
    latency_ms: float


class CausalInferenceEngine:
    """因果推断引擎."""
    
    def __init__(self):
        self.scm: Optional[StructuralCausalModel] = None
        self.discovery = CausalDiscovery()
        
        # 统计
        self.total_queries = 0
    
    def set_causal_model(self, scm: StructuralCausalModel):
        """设置因果模型."""
        self.scm = scm
    
    def discover_from_data(self, data: Dict[str, np.ndarray]) -> CausalGraph:
        """从数据发现因果结构."""
        return self.discovery.discover(data)
    
    def estimate_ate(self, treatment: str, outcome: str,
                     treatment_value: float = 1.0,
                     control_value: float = 0.0,
                     n_samples: int = 5000) -> CausalResult:
        """估计平均处理效应 (ATE).
        
        ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        """
        start_time = time.perf_counter()
        self.total_queries += 1
        
        if not self.scm:
            return CausalResult(
                query=CausalQuery("ate", treatment, outcome),
                estimate=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                method="none",
                details={"error": "No causal model set"},
                latency_ms=0
            )
        
        # 干预采样
        data_treated = self.scm.intervene({treatment: treatment_value}, n_samples)
        data_control = self.scm.intervene({treatment: control_value}, n_samples)
        
        # 计算 ATE
        y_treated = data_treated[outcome]
        y_control = data_control[outcome]
        
        ate = np.mean(y_treated) - np.mean(y_control)
        
        # 置信区间 (bootstrap)
        bootstrap_ates = []
        for _ in range(100):
            idx_t = np.random.choice(n_samples, n_samples, replace=True)
            idx_c = np.random.choice(n_samples, n_samples, replace=True)
            boot_ate = np.mean(y_treated[idx_t]) - np.mean(y_control[idx_c])
            bootstrap_ates.append(boot_ate)
        
        ci_low = np.percentile(bootstrap_ates, 2.5)
        ci_high = np.percentile(bootstrap_ates, 97.5)
        
        # p-value (简化: t-test)
        from scipy import stats
        try:
            _, p_value = stats.ttest_ind(y_treated, y_control)
        except:
            p_value = 0.0
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return CausalResult(
            query=CausalQuery("ate", treatment, outcome, 
                            intervention_value=treatment_value),
            estimate=float(ate),
            confidence_interval=(float(ci_low), float(ci_high)),
            p_value=float(p_value),
            method="do-calculus",
            details={
                "n_samples": n_samples,
                "mean_treated": float(np.mean(y_treated)),
                "mean_control": float(np.mean(y_control)),
                "std_treated": float(np.std(y_treated)),
                "std_control": float(np.std(y_control)),
            },
            latency_ms=latency
        )
    
    def estimate_counterfactual(self, observation: Dict[str, float],
                                intervention: Dict[str, float],
                                target: str) -> CausalResult:
        """估计反事实."""
        start_time = time.perf_counter()
        self.total_queries += 1
        
        if not self.scm:
            return CausalResult(
                query=CausalQuery("counterfactual", list(intervention.keys())[0], target),
                estimate=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=0.0,
                method="none",
                details={"error": "No causal model set"},
                latency_ms=0
            )
        
        # 反事实计算
        cf_value = self.scm.counterfactual(observation, intervention, target)
        
        # 不确定性估计 (简化)
        ci_width = 0.1 * abs(cf_value) + 0.01
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return CausalResult(
            query=CausalQuery("counterfactual", 
                            list(intervention.keys())[0], target,
                            conditioning=observation),
            estimate=float(cf_value),
            confidence_interval=(float(cf_value - ci_width), 
                               float(cf_value + ci_width)),
            p_value=0.05,  # 反事实无传统 p-value
            method="twin_network",
            details={
                "observation": observation,
                "intervention": intervention,
                "factual_outcome": observation.get(target, None),
            },
            latency_ms=latency
        )
    
    def analyze_mediation(self, treatment: str, mediator: str, outcome: str,
                          n_samples: int = 5000) -> Dict[str, CausalResult]:
        """中介分析.
        
        分解总效应为:
        - 直接效应 (DE): T → Y
        - 间接效应 (IE): T → M → Y
        """
        # 总效应
        total_effect = self.estimate_ate(treatment, outcome, n_samples=n_samples)
        
        # 直接效应 (控制中介变量)
        # 这里需要更复杂的实现...
        
        # 间接效应 = 总效应 - 直接效应
        
        return {
            "total_effect": total_effect,
            # "direct_effect": direct_effect,
            # "indirect_effect": indirect_effect,
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_causal_inference_engine() -> CausalInferenceEngine:
    """创建因果推断引擎并初始化示例模型."""
    engine = CausalInferenceEngine()
    
    # 创建示例因果图
    graph = CausalGraph()
    graph.add_edge("education", "income", mechanism="skills")
    graph.add_edge("income", "health", mechanism="healthcare_access")
    graph.add_edge("education", "health", mechanism="health_literacy")
    graph.add_edge("age", "income")
    graph.add_edge("age", "health")
    
    # 创建 SCM
    scm = StructuralCausalModel(graph)
    
    # 定义结构方程
    scm.add_equation("age", [], lambda p, n: n * 10 + 40)
    scm.add_equation("education", [], lambda p, n: np.clip(n * 2 + 12, 0, 20))
    scm.add_equation("income", ["age", "education"],
                     lambda p, n: 20000 + p["age"] * 500 + p["education"] * 3000 + n * 5000)
    scm.add_equation("health", ["income", "education", "age"],
                     lambda p, n: np.clip(
                         50 + p["income"] / 10000 + p["education"] * 2 - p["age"] * 0.3 + n * 10,
                         0, 100))
    
    engine.set_causal_model(scm)
    
    return engine


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 因果推断引擎 - 演示")
    print("=" * 70)
    
    engine = create_causal_inference_engine()
    
    # 测试 ATE 估计
    print("\n1. 平均处理效应 (ATE) 估计")
    print("-" * 50)
    
    ate_result = engine.estimate_ate("education", "income", 
                                     treatment_value=16, control_value=12)
    print(f"问题: 教育从12年增加到16年对收入的因果效应")
    print(f"ATE 估计: ${ate_result.estimate:,.0f}")
    print(f"95% CI: (${ate_result.confidence_interval[0]:,.0f}, ${ate_result.confidence_interval[1]:,.0f})")
    print(f"p-value: {ate_result.p_value:.4f}")
    print(f"延迟: {ate_result.latency_ms:.2f} ms")
    
    # 测试反事实
    print("\n2. 反事实推理")
    print("-" * 50)
    
    observation = {"age": 30, "education": 12, "income": 50000, "health": 70}
    intervention = {"education": 16}
    
    cf_result = engine.estimate_counterfactual(observation, intervention, "income")
    print(f"观测: 30岁, 12年教育, 收入$50000, 健康70")
    print(f"反事实问题: 如果教育是16年, 收入会是多少?")
    print(f"反事实估计: ${cf_result.estimate:,.0f}")
    print(f"实际收入变化: ${cf_result.estimate - observation['income']:+,.0f}")
    
    print("\n" + "=" * 70)
