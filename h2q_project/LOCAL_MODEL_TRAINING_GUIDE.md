# H2Q-Evo 本地大模型高级训练系统 - 快速开始指南

## 📋 系统概述

这个系统提供了完整的本地大模型继续训练能力，包括：

### 🎯 四大核心功能

1. **能力评估系统 (CompetencyEvaluator)**
   - 10+ 个维度的能力评估
   - 多层次能力等级判定
   - 与在线模型基准对标

2. **内容输出矫正 (OutputCorrectionMechanism)**
   - 自动错误检测（重复、不完整、矛盾）
   - 智能内容修正
   - 质量保证机制

3. **循环学习系统 (IterativeLearningSystem)**
   - 多轮迭代训练
   - 动态目标调整
   - 性能反馈循环

4. **高级训练管理器 (LocalModelAdvancedTrainer)**
   - 完整的训练流程管理
   - 实时性能监控
   - 报告生成和可视化

---

## 🚀 快速开始

### 1. 环境准备

```bash
cd /Users/imymm/H2Q-Evo/h2q_project

# 确保依赖已安装
pip install torch numpy -q

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. 运行演示评估

```bash
# 运行能力评估演示
python local_model_advanced_training.py
```

**预期输出**:
```
======================================
H2Q-Evo 本地大模型高级训练系统 - 演示
======================================

执行完整能力评估...

评估结果:
  正确性: 85.00%
  一致性: 82.00%
  完整性: 90.00%
  流畅性: 88.00%
  连贯性: 84.00%
  推理深度: 78.00%
  知识准确性: 82.00%
  语言控制: 80.00%
  创意性: 70.00%
  适应性: 75.00%

  总体评分: 80.90%
  能力等级: ADVANCED

改进建议:
  ... (改进建议列表)
```

### 3. 运行完整训练

```bash
# 启动完整的高级训练系统
python train_local_model_advanced.py
```

**运行时间**: 15-60 分钟（取决于数据量和硬件）

**输出文件**:
- `training_output/training_report.json` - 详细的 JSON 报告
- `training_output/training_report.md` - Markdown 格式报告
- `training_output/best_model_iteration_*.pt` - 最佳模型权重
- `local_model_training.log` - 完整的训练日志

---

## 📚 详细使用指南

### 评估单个输出

```python
from local_model_advanced_training import CompetencyEvaluator

# 初始化评估器
evaluator = CompetencyEvaluator(device='cpu')

# 评估文本
output = "人工智能是计算机科学的一个分支..."
reference = "人工智能（AI）是..."

metrics = evaluator.evaluate_full(
    output=output,
    reference=reference,
    expected_elements=["AI", "计算机", "智能"]
)

# 查看结果
print(f"总体评分: {metrics.overall_score:.2%}")
print(f"能力等级: {metrics.competency_level.name}")

# 获取改进建议
suggestions = evaluator.get_improvement_suggestions(metrics)
for suggestion in suggestions:
    print(f"  {suggestion}")
```

### 矫正输出内容

```python
from local_model_advanced_training import OutputCorrectionMechanism

# 初始化矫正器
corrector = OutputCorrectionMechanism()

# 检测错误
text = "这是一个测试。这是一个测试。这是一个测试。"
errors = corrector.detect_errors(text)
print(f"检测到 {len(errors)} 个错误")

# 矫正输出
corrected_text, corrections = corrector.correct_output(text, errors)
print(f"原文: {text}")
print(f"修正: {corrected_text}")
print(f"操作: {corrections}")
```

### 自定义训练循环

```python
from local_model_advanced_training import LocalModelAdvancedTrainer
import torch.nn as nn

# 你的模型
model = YourModel()

# 初始化训练器
trainer = LocalModelAdvancedTrainer(model, device='cuda')

# 准备数据
train_data = [("输入文本", "目标文本"), ...]
val_data = [("输入文本", "目标文本"), ...]

# 运行训练
history = trainer.train(
    training_data=train_data,
    validation_data=val_data,
    num_iterations=10,
    learning_rate=1e-4,
    batch_size=32
)

# 查看历史
for iteration in history:
    print(f"迭代 {iteration['iteration']}: "
          f"总体评分 {iteration['metrics']['overall_score']:.2%}")
```

---

## 🔍 能力评估维度详解

### 基础维度 (40% 权重)

| 维度 | 说明 | 评估方法 |
|------|------|--------|
| **正确性** | 答案的准确度 | 与参考答案对比、事实检查 |
| **一致性** | 多次输出的一致性 | 计算输出相似度 |
| **完整性** | 答案的完整程度 | 检查所需元素覆盖率 |
| **流畅性** | 表达的自然度 | 分析句式和词汇搭配 |
| **连贯性** | 逻辑关系的清晰度 | 检测逻辑连接词 |

### 高级维度 (60% 权重)

| 维度 | 说明 | 评估方法 |
|------|------|--------|
| **推理深度** | 分析深度 | 检测因果关系、条件分析 |
| **知识准确性** | 事实准确性 | 事实检查、知识库验证 |
| **语言控制** | 表达能力 | 词汇多样性、句式变化 |
| **创意性** | 创新程度 | 检测新颖表达和观点 |
| **适应性** | 灵活调整能力 | 上下文适配度检查 |

### 能力等级划分

```
总体评分          能力等级    描述
0-40%            BASIC      基础能力
40-60%           INTERMEDIATE 中级能力
60-80%           ADVANCED   高级能力
80-95%           EXPERT     专家级能力
95-100%          MASTERY    精通级能力 ⭐
```

---

## 🔧 配置和优化

### 调整评估权重

编辑 `CompetencyMetrics.__post_init__()`:

```python
# 基础指标权重: 40% -> 30%
basic_score = (
    self.correctness * 0.25 +
    self.consistency * 0.20 +
    # ... 其他
)

# 高级指标权重: 60% -> 70%
advanced_score = (
    self.reasoning_depth * 0.15 +
    # ... 其他
)

self.overall_score = basic_score * 0.3 + advanced_score * 0.7
```

### 添加自定义评估维度

```python
class CustomCompetencyMetrics(CompetencyMetrics):
    # 添加新维度
    domain_expertise: float    # 领域专业性
    safety_awareness: float    # 安全意识
    
    def __post_init__(self):
        # 在原有基础上加入新维度
        super().__post_init__()
        # 调整权重...
```

### 自定义矫正规则

```python
# 在 OutputCorrectionMechanism.__init__() 中
custom_rules = [
    {
        "name": "自定义规则",
        "pattern": r"你的正则表达式",
        "replacement": r"替换模式",
        "severity": "high"
    }
]
self.correction_rules.extend(custom_rules)
```

---

## 📊 输出示例

### 训练报告结构

```
training_output/
├── training_report.json          # 详细的 JSON 数据
├── training_report.md            # Markdown 格式报告
├── best_model_iteration_1.pt     # 最佳模型 1
├── best_model_iteration_5.pt     # 最佳模型 5
└── ...
```

### JSON 报告示例

```json
{
  "title": "H2Q-Evo 本地大模型高级训练报告",
  "timestamp": "2026-01-20T...",
  "summary": {
    "total_iterations": 10,
    "best_overall_score": 0.8231,
    "final_overall_score": 0.8015
  },
  "iterations": [
    {
      "iteration": 1,
      "timestamp": "...",
      "train_loss": 0.4521,
      "metrics": {
        "correctness": 0.85,
        "consistency": 0.82,
        "overall_score": 0.7821,
        "competency_level": "ADVANCED"
      },
      "improvements": [
        "⚠️ 正确性需要改进...",
        "⚠️ 推理深度需要改进..."
      ]
    }
  ]
}
```

---

## 💡 最佳实践

### 1. 数据准备

✅ **推荐做法**:
- 多样化数据源（文学、技术、对话等）
- 高质量参考答案
- 清晰的输入-输出对应

❌ **避免**:
- 数据不均衡（某类数据过多）
- 低质量或错误的参考答案
- 重复或相似的样本

### 2. 训练策略

✅ **推荐做法**:
- 从小数据集开始验证流程
- 逐步增加数据量和复杂度
- 定期检查评估结果
- 保存中间模型检查点

❌ **避免**:
- 一次性用大数据集
- 忽视评估指标
- 过度拟合
- 频繁改变目标

### 3. 性能优化

✅ **推荐做法**:
- 使用 GPU 加速（如可用）
- 批处理多个样本
- 使用混合精度训练
- 定期清理日志和检查点

❌ **避免**:
- 单样本训练
- CPU 上处理大数据
- 保存所有中间结果
- 忽视内存管理

---

## 🐛 故障排除

### 问题 1: 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 减小批处理大小
batch_size = 16  # 从 32 减少到 16

# 使用 CPU
device = 'cpu'

# 启用梯度检查点
model.gradient_checkpointing_enable()
```

### 问题 2: 评估分数不变

**症状**: 所有迭代的评估分数都相同

**解决方案**:
```python
# 检查模型是否真的在更新
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: requires_grad={param.requires_grad}")

# 增加学习率
learning_rate = 1e-3  # 从 1e-4 增加到 1e-3
```

### 问题 3: 评估过程很慢

**症状**: 每次评估需要很长时间

**解决方案**:
```python
# 减少验证数据量
val_data = val_data[:50]  # 只使用前 50 个样本

# 使用更简单的评估方法
# 跳过某些昂贵的计算

# 并行处理
num_workers = 4
```

---

## 📖 更多资源

### 相关文件
- `local_model_advanced_training.py` - 核心系统实现
- `train_local_model_advanced.py` - 完整训练脚本
- `local_model_training.log` - 训练日志

### 集成建议
- 与 H2Q 的 DiscreteDecisionEngine 集成
- 与生产验证系统集成
- 与监控和告警系统集成

---

## 🎓 学习路径

**初级**:
1. 运行演示评估
2. 理解能力维度
3. 尝试手动矫正

**中级**:
1. 运行完整训练
2. 自定义评估权重
3. 添加自定义数据

**高级**:
1. 修改系统架构
2. 实现自定义评估器
3. 生产环境部署

---

## ❓ 常见问题

**Q: 能否集成到现有的模型中?**  
A: 是的，通过 `LocalModelAdvancedTrainer` 可以包装任何 PyTorch 模型。

**Q: 评估结果准确吗?**  
A: 评估系统提供了启发式方法。对关键应用建议添加人类评审。

**Q: 能否离线运行?**  
A: 是的，所有处理都在本地进行，不需要网络连接。

**Q: 如何加速训练?**  
A: 使用 GPU、减小验证集大小、并行处理。

---

## 📞 反馈和支持

如有问题或建议，请:
1. 检查日志文件
2. 阅读相关源代码
3. 提交 GitHub Issue
4. 联系开发团队

---

**版本**: v1.0  
**最后更新**: 2026-01-20  
**状态**: ✅ 生产就绪
