# Precision-Gated Executor - 快速开始指南

## 概述

`PrecisionGatedExecutor` 实现了 **DAS Meta-Theory** (定向公理系统)，一个专为防止LLM幻觉而设计的架构框架。

## 核心特性

✅ **精度门控 (Precision Gating)** - 基于熵防止不确定的输出  
✅ **对偶验证 (Dualistic Verification)** - 命题对一致性检查  
✅ **拓扑闭包 (Topological Closure)** - 概率分布验证  
✅ **指标解耦 (Metric Decoupling)** - 离散逻辑 + 连续流形  
✅ **执行追踪 (Execution Tracing)** - 完整审计日志  

## 快速开始

### 1. 基础使用

```python
from h2q_project.local_executor import LocalExecutor

# 创建执行器（启用精度门控）
executor = LocalExecutor(enable_precision_gating=True)

# 执行任务
result = executor.execute(
    task="Calculate the square root of 144",
    strategy="auto",
)

# 检查结果
print(f"输出: {result['output']}")
print(f"置信度: {result['confidence']:.2f}")
print(f"状态: {result['state_manifold']}")
```

### 2. 理解执行状态

| 状态 | 熵范围 | 含义 | 执行方式 |
|------|--------|------|---------|
| **Particle** | < 0.25 | 高精度 | 直接输出 ✓ |
| **Coherence** | 0.25-0.65 | 平衡 | 标准验证 |
| **Wave** | > 0.65 | 低精度 | 思维链展开 |

```python
# 例1: 粒子态（高精度数学）
result1 = executor.execute("2 + 2 = ?")
# 状态: particle, 直接输出

# 例2: 波态（高熵推理）
result2 = executor.execute(
    "Should we prioritize AI safety or capabilities? Discuss tradeoffs."
)
# 状态: wave, 使用思维链

# 例3: 相干态（平衡查询）
result3 = executor.execute("What is the capital of France?")
# 状态: coherence, 标准验证
```

### 3. 熵度量解释

```python
# 获取详细的熵分解
print(f"逻辑熵: {result['entropy_metrics']['logical_entropy']:.4f}")
print(f"语义熵: {result['entropy_metrics']['semantic_entropy']:.4f}")
print(f"时间熵: {result['entropy_metrics']['temporal_entropy']:.4f}")
print(f"组合熵: {result['entropy_metrics']['combined_entropy']:.4f}")

# 含义:
# - 逻辑熵: 任务关键词的多样性 (Shannon 熵)
# - 语义熵: 四元数语义空间中的不确定性
# - 时间熵: 历史状态变化的速率
# - 组合熵: 加权平均
```

### 4. 对偶验证 (Axiom I)

```python
# 查看对偶命题验证
for i, prop in enumerate(result['dualistic_verification'], 1):
    print(f"\n命题对 {i}:")
    print(f"  论文: {prop['thesis']}")
    print(f"  反论文: {prop['antithesis']}")
    print(f"  论文置信度: {prop['thesis_confidence']:.4f}")
    print(f"  反论文置信度: {prop['antithesis_confidence']:.4f}")
    print(f"  拓扑闭包有效: {prop['closure_valid']}")
    print(f"  闭包间隙: {prop['closure_gap']:.6f}")

# 拓扑闭包解释:
# - 闭包间隙 ≈ 0: 概率分布一致 ✓
# - 闭包间隙 >> 0: 表示可能的幻觉或不一致 ✗
```

### 5. 执行追踪

```python
# 查看完整的执行路径
print("执行追踪:")
for step in result['execution_trace']:
    print(f"  {step}")

# 输出示例:
# STEP_1_ENTROPY_MEASUREMENT
# STEP_2_DUALISTIC_GENERATION
# STEP_3_CLOSURE_VERIFICATION
# STEP_4_EXECUTION_ROUTING
# ROUTE_CHAIN_OF_THOUGHT
# STEP_5_MANIFOLD_ENCODING
```

### 6. 统计信息

```python
# 获取执行历史统计
stats = executor.get_precision_gating_stats()

print(f"总执行数: {stats['total_executions']}")
print(f"状态分布: {stats['state_distribution']}")
print(f"平均熵: {stats['average_entropy']:.4f}")
print(f"平均置信度: {stats['average_confidence']:.4f}")
```

## 运行演示

```bash
cd /Users/imymm/H2Q-Evo/h2q_project

# 运行完整演示
python3 precision_gated_demo.py

# 输出会显示:
# - DAS Meta-Theory 概念说明
# - 5 个测试用例的执行结果
# - 熵分解和状态分布
# - 对偶验证详情
# - 统计汇总
```

## 核心概念

### 公理 III - 指标解耦

分离**离散逻辑**（图灵机）和**连续流形**（四元数）：

```python
# 离散层：任务分类 (图灵机操作)
is_negative = "not" in task.lower()
anchor_type = "negative" if is_negative else "affirmative"

# 连续层：四元数编码 (Hamilton代数)
q = semantic_anchors[anchor_type] * confidence
q_normalized = q / ||q||  # 单位四元数
```

### 精度门控因果性

基于熵的动态路由：

```
任务 → 测量熵
         ↓
    ┌────┴─────┐
    ↓          ↓
  高精度     低精度
  粒子态     波态
    ↓          ↓
  直接       思维链+
  输出       工具验证
```

**关键原理**: 只有低熵(高精度)的情况下才允许直接输出！

### 公理 I - 对偶生成

生成命题对验证一致性：

```python
# 对于每个任务，生成:
thesis = "任务可以被肯定地解决"
antithesis = "任务不能被肯定地解决"

# 验证拓扑闭包:
P(thesis) + P(antithesis) ≈ 1.0

# 如果闭包间隙太大 → 可能存在幻觉
```

## 防止幻觉的工作原理

### 问题：传统 LLM

```
输入 → LLM → Softmax坍缺 → 输出（可能是幻觉！）
```

LLM 即使在不确定的情况下也会输出令人信服的答案。

### 解决方案：DAS Meta-Theory

```
输入 → 熵测量 → 高熵？
            ↓
        不允许直接输出！
            ↓
      思维链展开 + 
      工具验证 +
      对偶检验
            ↓
      对偶命题检查
      (Thesis vs Antithesis)
            ↓
      拓扑闭包验证
      (P(A) + P(¬A) ≈ 1.0)
            ↓
      输出 + 完整验证记录
```

**效果**:
- ✓ 低熵任务：快速直接输出
- ✓ 高熵任务：强制推理验证
- ✓ 所有输出：拓扑闭包保证一致性

## 关键参数调整

```python
# 自定义熵阈值
executor.precision_gated_executor._precision_threshold = 0.35  # 默认 0.30

# 禁用思维链（性能优化）
executor.precision_gated_executor.enable_cot = False

# 禁用精度门控（回到标准模式）
executor = LocalExecutor(enable_precision_gating=False)

# 拓扑闭包容差
prop.verify_closure(tolerance=0.1)  # 默认 0.05
```

## 性能特性

| 组件 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 熵测量 | O(k log k) | O(k) |
| 四元数编码 | O(k) | O(1) |
| 对偶生成 | O(k) | O(1) |
| 逻辑验证 | O(n²) | O(n) |

其中 k = 关键词数量, n = 命题数量

## 调试提示

### 检查是否启用精度门控

```python
if executor.precision_gated_executor:
    print("✓ 精度门控已启用")
else:
    print("✗ 精度门控未启用")
```

### 查看原始熵值

```python
metrics = executor.precision_gated_executor._measure_entropy(task)
print(f"逻辑熵: {metrics.logical_entropy}")
print(f"语义熵: {metrics.semantic_entropy}")
print(f"时间熵: {metrics.temporal_entropy}")
print(f"组合熵: {metrics.combined_entropy}")
```

### 验证对偶命题

```python
props = executor.precision_gated_executor._generate_dualistic_propositions(task)
for prop in props:
    is_valid = prop.verify_closure()
    print(f"论文: {prop.thesis[:40]}... 有效: {is_valid}")
```

## 进阶用法

### 自定义语义锚点

```python
encoder = ContinuousManifoldEncoder()
encoder._semantic_anchors["custom"] = np.array([1.0, 0.5, 0.5, 0.5])
```

### 自定义逻辑规则

```python
verifier = DiscreteLogicVerifier()
verifier._contradiction_rules.append(("hypothesis", "null_hypothesis"))
```

### 集成外部验证工具

```python
result = executor.execute(task, strategy="auto")

# 如果状态是 wave，运行额外验证
if result['state_manifold'] == 'wave':
    # 调用外部工具（搜索、计算器等）
    verified_result = external_verifier(result['output'])
```

## 故障排除

### 问题：所有任务都是 wave 状态

**原因**: 熵测量的关键词可能过多
**解决**: 调整 `_extract_confidence()` 中的确定性标记

### 问题：拓扑闭包总是无效

**原因**: 信度计算可能不平衡
**解决**: 调整 `_extract_confidence()` 中的权重

### 问题：执行太慢

**原因**: 思维链展开对所有高熵任务
**解决**: `executor.precision_gated_executor.enable_cot = False`

## 参考资源

- 📖 [完整DAS Meta-Theory文档](./DAS_META_THEORY.md)
- 🧪 [单元测试](./test_precision_gated_executor.py)
- 📊 [演示脚本](./precision_gated_demo.py)
- 📝 [源代码](./precision_gated_executor.py)

## 支持和反馈

对于问题、建议或改进意见，请参考项目指南。
