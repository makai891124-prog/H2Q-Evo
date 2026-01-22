# H2Q AGI 监督学习系统 - 快速入门指南

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           终 极 目 标                                       ║
║                                                                            ║
║          训练本地可用的实时AGI系统                                          ║
╚════════════════════════════════════════════════════════════════════════════╝
```

## 系统概述

本系统实现了真实的神经网络学习框架，配合 Gemini 第三方验证，确保 AGI 系统是通过真正的学习获得能力，而非作弊手段。

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 真实学习框架 | `real_learning_framework.py` | 神经网络真实学习，带计算追踪 |
| Gemini 验证器 | `gemini_verifier.py` | 第三方幻觉/作弊检测 |
| 监督学习系统 | `supervised_learning_system.py` | 集成学习与验证 |

---

## 快速开始

### 1. 基本运行（无需 API Key）

```bash
cd /Users/imymm/H2Q-Evo

# 运行真实学习框架
PYTHONPATH=. python3 h2q_project/h2q/agi/real_learning_framework.py

# 运行监督学习系统（本地验证）
PYTHONPATH=h2q_project/h2q/agi python3 h2q_project/h2q/agi/supervised_learning_system.py
```

### 2. 启用 Gemini 第三方验证

#### 方法 A: 使用 Gemini API Key

1. 获取 API Key: https://aistudio.google.com/app/apikey
2. 设置环境变量:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

3. 运行验证:

```bash
PYTHONPATH=. python3 h2q_project/h2q/agi/gemini_verifier.py
```

#### 方法 B: 使用 Gemini Code Assist（VS Code）

1. 在 VS Code 中安装 "Gemini Code Assist" 扩展
2. 使用您的免费席位登录
3. 扩展会自动配置 API Key

#### 方法 C: 使用 Gemini CLI（推荐）

```bash
# 安装 Gemini CLI
npm install -g @anthropic-ai/gemini-cli

# 或使用 pip
pip install google-genai

# 配置
gemini auth login
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      H2Q 监督学习系统                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  训练数据生成     │───▶│  真实学习 AGI    │───▶│  学习证明输出    │      │
│  │  (即时计算)       │    │  (梯度下降)      │    │  (可验证)        │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│           │                       │                       │                │
│           ▼                       ▼                       ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                     Gemini 第三方验证                             │      │
│  │  • 幻觉检测  • 作弊检测  • 代码质量  • 学习验证  • 事实核查       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 真实学习保证

### 我们的承诺

1. **无预设答案**: 所有输出都通过神经网络计算得到
2. **无查找表**: 不存在根据输入名称返回预存结果的逻辑
3. **可追踪计算**: 每次推理都有完整的计算步骤记录
4. **真实梯度**: 学习通过梯度下降实现，可验证梯度流动
5. **第三方验证**: Gemini 作为独立验证者检查代码和学习过程

### 反作弊检测

系统会自动检测以下作弊模式：

```python
# ❌ 作弊模式 1: 查找表
answers = {"task1": 42, "task2": 100}
return answers[task_name]

# ❌ 作弊模式 2: 名称匹配
if "arithmetic" in task_name:
    return simple_add()

# ❌ 作弊模式 3: 硬编码
return 42

# ✓ 正确方式: 神经网络计算
output = self.model(encoded_input)
return self.decoder(output)
```

---

## 验证类型

### 1. 幻觉检测 (hallucination_check)

检查生成的代码/声明是否包含虚假信息：
- 不存在的 API/函数
- 错误的技术概念
- 夸大的能力声明
- 虚假的性能数据

### 2. 作弊检测 (cheating_detection)

检测代码是否使用作弊手段：
- 硬编码返回值
- 查找表
- 名称/类别匹配
- 预计算结果

### 3. 代码质量 (code_quality)

评估代码质量：
- 可读性
- 可维护性
- 错误处理
- 性能
- 安全性

### 4. 学习验证 (learning_verification)

验证神经网络是否真的在学习：
- 损失下降趋势
- 梯度健康状况
- 学习曲线合理性
- 过拟合风险

### 5. 事实核查 (fact_check)

验证技术声明的准确性：
- 声明是否准确
- 是否有夸大
- 证据是否支持

---

## 使用示例

### 训练真实学习模型

```python
from supervised_learning_system import SupervisedLearningManager, SupervisedLearningConfig

config = SupervisedLearningConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    use_gemini_verification=True  # 启用 Gemini 验证
)

manager = SupervisedLearningManager(config)
report = manager.train(num_epochs=100)

print(f"学习状态: {report['learning_proof']['status']}")
```

### 验证代码

```python
from gemini_verifier import GeminiVerifier

verifier = GeminiVerifier()

# 检测作弊
result = verifier.verify_cheating(code, "代码应该计算而非查找")
print(f"是否作弊: {not result.passed}")

# 验证学习
result = verifier.verify_learning(learning_proof)
print(f"学习真实性: {result.passed}")
```

### 实时监督

```python
from gemini_verifier import RealTimeSupervisionSystem

supervisor = RealTimeSupervisionSystem()

# 监督代码生成
is_ok, result = supervisor.supervise_code_generation(generated_code)
if not is_ok:
    print("警报: 检测到作弊!")

# 监督学习过程
is_ok, result = supervisor.supervise_learning(learning_proof)
```

---

## 输出文件

| 文件 | 位置 | 内容 |
|------|------|------|
| 模型权重 | `h2q_project/h2q/agi/real_learning_agi.pt` | 训练好的模型 |
| 检查点 | `h2q_project/h2q/agi/checkpoints/` | 训练检查点 |
| 验证报告 | `h2q_project/h2q/agi/gemini_verification_report.json` | Gemini 验证结果 |
| 训练报告 | `h2q_project/h2q/agi/checkpoints/training_report.json` | 训练摘要 |

---

## 常见问题

### Q: 没有 Gemini API Key 能用吗？

A: 可以。系统会自动切换到本地验证模式，使用基于规则的检测。

### Q: 如何确认模型真的在学习？

A: 查看学习证明：
```python
proof = model.get_learning_proof()
print(proof['status'])  # 应该是 'learning_verified'
print(proof['loss_trend'])  # 应该是负数（损失下降）
```

### Q: 如何检测作弊代码？

A: 使用本地验证器：
```python
from supervised_learning_system import LocalVerifier
is_clean, issues = LocalVerifier.verify_no_cheating(code)
```

---

## 下一步

1. 获取 Gemini API Key 启用第三方验证
2. 运行完整的监督学习训练
3. 检查验证报告，确保学习真实性
4. 部署验证通过的模型

---

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           终 极 目 标                                       ║
║                                                                            ║
║          训练本地可用的实时AGI系统                                          ║
║                                                                            ║
║  本系统确保 AGI 通过真实学习获得能力，而非作弊手段。                         ║
╚════════════════════════════════════════════════════════════════════════════╝
```

文档更新: 2026-01-22
