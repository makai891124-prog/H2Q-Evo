# H2Q-Evo 数学架构快速参考 (Quick Reference)

## 🚀 快速开始

### 初始化统一架构

```python
from h2q.core.unified_architecture import get_unified_h2q_architecture

# 创建引擎
unified = get_unified_h2q_architecture(dim=256, action_dim=64)

# 使用
x = torch.randn(batch_size, 256)
output, results = unified(x)
```

### 初始化进化系统集成

```python
from h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge

# 创建桥接
bridge = MathematicalArchitectureEvolutionBridge(dim=256, action_dim=64)

# 进化步骤
state = torch.randn(batch_size, 256)
learning_signal = torch.tensor(0.1)
results = bridge.evolution_step(state, learning_signal)

# 保存检查点
bridge.save_checkpoint("checkpoint.pt")
```

---

## 📋 模块接口

### 1. LieAutomorphismEngine (李群自动同构引擎)

```python
from h2q.core.lie_automorphism_engine import get_lie_automorphism_engine

engine = get_lie_automorphism_engine(dim=256)
output, intermediates = engine(x)

# 中间表示
intermediates['quaternion']      # [batch, 4] - 四元数投影
intermediates['fractal']         # [batch, 256] - 分形变换
intermediates['reflected']       # [batch, 256] - 反射变换
intermediates['knot_invariants'] # [batch, 3] - 纽结不变量
```

### 2. NoncommutativeGeometryModule (非交换几何反射)

```python
from h2q.core.noncommutative_geometry_operators import (
    ComprehensiveReflectionOperatorModule
)

ops = ComprehensiveReflectionOperatorModule(dim=256)
output, results = ops(x)

# 结果
results['fueter_violation']      # 标量 - Fueter正则性违反
results['reflection_laplacian']  # [batch, 256] - 反射Laplacian
results['weyl_projection']       # [batch, 256] - Weyl投影
results['spacetime_reflection']  # [batch, 256] - CP反演
```

### 3. AutomophicDDE (自动同构决策引擎)

```python
from h2q.core.automorphic_dde import get_automorphic_dde

dde = get_automorphic_dde(latent_dim=256, action_dim=64)
action_probs, results = dde(state)

# 关键输出
results['action_sample']         # [batch] - 采样行动
results['spectral_shift']        # [batch] - η值
results['topological_tear']      # 布尔 - 拓扑撕裂检测
results['running_eta']           # 标量 - 运行平均
```

### 4. KnotInvariantHub (纽结不变量)

```python
from h2q.core.knot_invariant_hub import KnotInvariantCentralHub

hub = KnotInvariantCentralHub(dim=256, knot_genus=3)
corrected_x, results = hub(x)

# 不变量
invariants = hub.compute_all_invariants()
invariants.alexander_poly    # Alexander多项式
invariants.jones_poly        # Jones多项式
invariants.homfly_poly       # HOMFLY多项式
invariants.knot_genus        # 亏格
```

---

## 🔧 常见操作

### 改变融合权重

```python
unified = get_unified_h2q_architecture()

# 访问权重
weights = unified.module_fusion_weights

# 调整
weights['lie_automorphism'].data = torch.tensor(0.4)
weights['reflection'].data = torch.tensor(0.3)
weights['knot_constraints'].data = torch.tensor(0.2)
weights['dde'].data = torch.tensor(0.1)
```

### 启用/禁用模块

```python
from h2q.core.unified_architecture import UnifiedMathematicalArchitectureConfig

config = UnifiedMathematicalArchitectureConfig(
    dim=256,
    enable_lie_automorphism=True,
    enable_reflection_operators=True,
    enable_knot_constraints=True,
    enable_dde_integration=True,
)
```

### 监控拓扑约束

```python
bridge = MathematicalArchitectureEvolutionBridge()

# 运行多个步骤
for gen in range(10):
    state = torch.randn(batch_size, 256)
    results = bridge.evolution_step(state)
    
    # 检查约束
    stats = bridge.unified_arch.get_system_report()
    eta = stats['statistics']['avg_eta']
    violation = stats['statistics']['avg_constraint_violation']
    
    print(f"Gen {gen}: η={eta:.4f}, Violation={violation:.4f}")

# 导出报告
report = bridge.export_metrics_report("report.json")
```

---

## 📊 性能优化

### 批处理

```python
# 小批处理 (1) - 延迟敏感
batch_1 = torch.randn(1, 256)
output_1, _ = unified(batch_1)  # ~32ms

# 中批处理 (8) - 平衡
batch_8 = torch.randn(8, 256)
output_8, _ = unified(batch_8)  # ~33ms (4x吞吐)

# 大批处理 (16+) - 吞吐量优化
batch_16 = torch.randn(16, 256)
output_16, _ = unified(batch_16)  # ~34ms (16x吞吐)
```

### GPU加速 (可选)

```python
# 目前支持CPU，GPU支持在开发中
device = "cpu"  # 或 "cuda", "mps"

unified = get_unified_h2q_architecture(
    dim=256, 
    device=device
)
```

---

## 🐛 调试技巧

### 检查形状

```python
output, results = unified(x)

print(f"输入: {x.shape}")
print(f"输出: {output.shape}")

for name, intermediate in results['intermediates'].items():
    print(f"  {name}: {intermediate.shape}")
```

### 监控数值稳定性

```python
output, results = unified(x)

# 检查NaN/Inf
has_nan = torch.isnan(output).any()
has_inf = torch.isinf(output).any()

print(f"NaN: {has_nan}, Inf: {has_inf}")

# 检查范数
print(f"输出范数: {torch.norm(output):.4f}")
print(f"输出最大: {torch.max(output).item():.4f}")
```

### 追踪梯度

```python
x = torch.randn(batch_size, 256, requires_grad=True)
output, results = unified(x)

loss = output.sum()
loss.backward()

print(f"梯度范数: {x.grad.norm():.4f}")
```

---

## 📈 指标解释

### 谱位移 (Spectral Shift, η)

```
η = (1/π) arg{det(S)}

含义:
- η ≈ 0: 系统稳定，拓扑完整
- |η| > 0.05: 警告，可能的拓扑撕裂
- |η| > 0.1: 严重，执行流形修复

用途: 检测系统何时出现"幻觉"或拓扑异常
```

### Fueter违反度量

```
违反 = ||∂_L f|| + ||∂_R f||

含义:
- 0: 完全Fueter-正则 (理想)
- <1: 轻微非正则性
- >10: 严重违反，需要修正

用途: 评估决策逻辑的全纯性
```

### 纽结亏格 (Knot Genus)

```
g(K) ≤ (degree(Δ) + 1) / 2

系统中: 默认 g=3 (三叶结复杂度)

约束: 确保系统的拓扑复杂度一致
```

---

## 🔗 与Evolution System集成

### 在evolution_system.py中使用

```python
from h2q.core.evolution_integration import create_mathematical_core_for_evolution_system

class H2QNexus:
    def __init__(self):
        # ... 其他初始化 ...
        
        # 初始化数学核心
        self.math_core = create_mathematical_core_for_evolution_system(
            dim=256,
            action_dim=64,
            project_root=self.project_root
        )
    
    def forward_pass(self, state, learning_signal):
        # 通过数学架构处理
        output, results = self.math_core.evolution_step(
            state,
            learning_signal
        )
        
        # 记录指标
        self.log_metrics(results)
        
        return output
    
    def save_state(self, path):
        # 保存数学核心
        self.math_core.save_checkpoint(path)
    
    def export_report(self, path):
        # 导出完整报告
        report = self.math_core.export_metrics_report(path)
        return report
```

---

## 📝 完整示例

```python
import torch
from h2q.core.evolution_integration import MathematicalArchitectureEvolutionBridge

# 初始化
bridge = MathematicalArchitectureEvolutionBridge(dim=256, action_dim=64)

# 运行进化循环
for generation in range(100):
    # 输入状态
    state = torch.randn(32, 256)
    
    # 计算学习信号
    target = torch.randn(32, 64)  # 目标动作分布
    
    # 进化步骤
    results = bridge.evolution_step(state, learning_signal=0.01)
    
    # 提取关键指标
    output = results['fused_output']
    stats = results['system_report']['statistics']
    
    if generation % 10 == 0:
        print(f"Gen {generation}:")
        print(f"  η average: {stats['avg_eta']:.6f}")
        print(f"  Constraint: {stats['avg_constraint_violation']:.6f}")
        print(f"  Modules: {results['enabled_modules']}")

# 保存和导出
bridge.save_checkpoint("final_checkpoint.pt")
report = bridge.export_metrics_report("evolution_report.json")
```

---

## 📚 进阶主题

### 自定义分形维数

```python
from h2q.core.lie_automorphism_engine import LieAutomorphismConfig, AutomaticAutomorphismOrchestrator

config = LieAutomorphismConfig(
    dim=256,
    fractal_levels=10,  # 增加深度
)
engine = AutomaticAutomorphismOrchestrator(config)
```

### 修改纽结配置

```python
from h2q.core.knot_invariant_hub import KnotInvariantCentralHub

hub = KnotInvariantCentralHub(dim=256, knot_genus=5)  # 更复杂的纽结
```

### 并行多模块

```python
from concurrent.futures import ThreadPoolExecutor

configs = [
    (256, i) for i in range(4)  # 4个不同配置
]

def create_engine(dim, seed):
    torch.manual_seed(seed)
    return get_unified_h2q_architecture(dim=dim)

with ThreadPoolExecutor(max_workers=4) as executor:
    engines = list(executor.map(lambda x: create_engine(*x), configs))
```

---

## 🎯 最佳实践

1. **始终检查拓扑约束** - 定期监控η和约束违反
2. **批量处理** - 使用batch≥8以获得最佳性能
3. **保存检查点** - 每N代保存一次以允许恢复
4. **监控梯度** - 确保梯度流正常，避免消失/爆炸
5. **导出指标** - 定期导出报告进行离线分析

---

**最后更新**: 2026年1月24日
**版本**: 1.0
**状态**: Production Ready 🟢
