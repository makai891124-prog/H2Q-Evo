# Contributing to H2Q-Evo

首先，感谢您对 H2Q-Evo 项目的兴趣！我们非常欢迎各种形式的贡献，帮助人类攀登最终 AGI 的高峰。

**First, thank you for your interest in H2Q-Evo! We welcome all contributions to build AGI together.**

---

## 📋 目录 (Table of Contents)

- [行为准则](#行为准则-code-of-conduct)
- [如何贡献](#如何贡献-how-to-contribute)
- [开发设置](#开发设置-development-setup)
- [编码规范](#编码规范-coding-standards)
- [提交指南](#提交指南-submission-guidelines)
- [报告 bug](#报告-bug-bug-reports)
- [建议功能](#建议功能-feature-requests)

---

## 行为准则 (Code of Conduct)

### 我们的承诺 (Our Commitment)

为了营造一个开放和欢迎的环境，我们（作为贡献者和维护者）承诺：

在我们的项目和社区中参与所有人员，无论其年龄、体型、残疾、种族、性别认同与表达、经验水平、国籍、个人外表、宗教或性认同与性取向，都享受一个无骚扰的体验。

### 我们的标准 (Our Standards)

建立积极环境的行为示例包括：

- 使用欢迎和包容的语言
- 尊重持不同观点和经历
- 接受建设性批评
- 关注对社区最有利的事情
- 表现对其他社区成员的同情

不可接受的行为示例包括：

- 使用带有性内涵的语言或意象
- 人身攻击、侮辱性评论
- 骚扰或骚扰性评论
- 未经同意发布他人隐私信息
- 其他可能合理地被视为不专业或不受欢迎的行为

---

## 如何贡献 (How to Contribute)

### 贡献类型 (Types of Contributions)

我们欢迎以下形式的贡献：

#### 🎯 核心算法 (Core Algorithm)

- **四元数优化**：改进 Fueter 微积分、流形学习算法
- **分形层级**：递归结构、内存管理优化
- **全纱流**：约束传播、幻觉检测改进

**技能要求**: 数学、PyTorch、深度学习

**预计时间**: 2-4 周

**示例 PR**: 
```python
# 改进四元数乘法的数值稳定性
# Improve numerical stability of quaternion multiplication
```

#### 🐛 Bug 修复 (Bug Fixes)

- 发现并修复 bug
- 改进错误消息
- 修复文档错误

**技能要求**: Python, debugging

**预计时间**: 1-3 天

**示例 PR**:
```
Fix: Incorrect dimension handling in quaternion projection layer
```

#### 📖 文档 (Documentation)

- 扩展现有文档
- 编写教程和示例
- 提供中文/英文翻译
- 改进 API 文档

**技能要求**: 技术写作、理解项目架构

**预计时间**: 1-2 周

**示例任务**:
- 编写"从零开始训练自定义模型"教程
- 添加更多 Jupyter 笔记本示例
- 改进 API 参考文档

#### 🧪 测试 (Testing)

- 编写单元测试
- 编写集成测试
- 性能基准测试
- 覆盖率改进

**技能要求**: Python, pytest, testing

**预计时间**: 1-2 周

**示例**:
```python
def test_quaternion_multiplication_commutativity():
    """Test that quaternion multiplication is not commutative"""
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(5, 6, 7, 8)
    assert (q1 * q2) != (q2 * q1)
```

#### 🚀 性能优化 (Performance)

- GPU/TPU CUDA 核心实现
- 分布式训练支持
- 内存使用优化
- 推理速度改进

**技能要求**: CUDA, PyTorch, 性能分析

**预计时间**: 2-4 周

**目标**: 
- 推理延迟 < 10 μs（vs 当前 23.68 μs）
- GPU 加速 > 50x

#### 🌍 应用 (Applications)

- 真实世界用例实现
- 与其他框架集成
- 演示项目
- 行业应用

**技能要求**: 特定领域知识

**预计时间**: 2-6 周

**示例**:
- 多模态学习集成（Vision + Language）
- 在线学习演示（持续适应）
- 边缘设备部署示例

---

## 开发设置 (Development Setup)

### 1. Fork 和 Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/H2Q-Evo.git
cd H2Q-Evo
git remote add upstream https://github.com/ORIGINAL_OWNER/H2Q-Evo.git
```

### 2. 创建虚拟环境 (Virtual Environment)

```bash
# Python 3.8+
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. 安装依赖 (Dependencies)

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

# 开发依赖包括:
# - pytest (单元测试)
# - black (代码格式化)
# - flake8 (代码检查)
# - mypy (类型检查)
# - sphinx (文档)
```

### 4. 创建功能分支 (Feature Branch)

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/bug-description
# 或
git checkout -b docs/documentation-update
```

### 5. 设置 Git 钩子 (Git Hooks) [可选]

```bash
# 自动运行测试和格式化
pip install pre-commit
pre-commit install
```

---

## 编码规范 (Coding Standards)

### Python 风格

遵循 **PEP 8** 标准，使用 `black` 和 `flake8`:

```bash
# 自动格式化
black h2q_project/

# 检查代码风格
flake8 h2q_project/

# 类型检查
mypy h2q_project/
```

### 命名约定 (Naming)

```python
# ✅ Good
def compute_quaternion_magnitude(q: Quaternion) -> float:
    """Calculate magnitude of a quaternion."""
    pass

class FractalHierarchyNode:
    """Node in a fractal hierarchy tree."""
    pass

# ❌ Avoid
def compute_quat_mag(q):
    pass

class FNode:
    pass
```

### 文档字符串 (Docstrings)

使用 Google 风格的 docstrings:

```python
def train_with_online_learning(
    model: HoloModel,
    data_stream: Iterator[Tensor],
    learning_rate: float = 1e-4,
) -> Dict[str, float]:
    """Train model with online learning using spectral shift tracking.

    Online learning allows the model to adapt continuously without
    catastrophic forgetting through incremental manifold updates.

    Args:
        model: The holomorphic model to train.
        data_stream: Iterator of input tensors from data stream.
        learning_rate: Learning rate for manifold adaptation.

    Returns:
        Dictionary with keys:
            - 'loss': Final training loss
            - 'eta': Spectral shift metric
            - 'iterations': Number of updates

    Raises:
        ValueError: If learning_rate is negative
        TypeError: If model is not HoloModel instance

    Example:
        >>> model = HoloModel(dim=32)
        >>> stream = iter(data_loader)
        >>> results = train_with_online_learning(model, stream)
        >>> print(results['eta'])
    """
    pass
```

### 类型提示 (Type Hints)

使用类型提示增加代码可读性：

```python
from typing import Dict, List, Optional, Tuple
import torch
from h2q.core.quaternion import Quaternion

def process_quaternion_batch(
    batch: torch.Tensor,
    apply_normalization: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process a batch of quaternions."""
    pass
```

### 测试 (Tests)

编写完整的单元测试和集成测试：

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_quaternion.py::test_multiplication

# 生成覆盖率报告
pytest --cov=h2q_project
```

### 测试结构

```
tests/
├── unit/
│   ├── test_quaternion.py
│   ├── test_fractal.py
│   └── test_memory.py
├── integration/
│   ├── test_training_pipeline.py
│   └── test_online_learning.py
└── fixtures/
    ├── sample_data.py
    └── mock_models.py
```

---

## 提交指南 (Submission Guidelines)

### Commit 消息 (Commit Messages)

遵循 **Conventional Commits** 格式:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能改进
- `test`: 测试更新
- `chore`: 构建过程、依赖更新

**示例**:

```
feat(quaternion): add SLERP interpolation for smooth rotation paths

Implement spherical linear interpolation for quaternions to enable
smooth animation and trajectory planning. This improves compatibility
with graphics pipelines and enables new use cases.

Closes #123
```

```
fix(memory): correct spectral swap buffer overflow

Fixed off-by-one error in spectral swap memory management that caused
buffer overflow when exceeding 2GB virtual memory. Added comprehensive
tests for edge cases.

Fixes #456
```

### Pull Request 模板

创建 PR 时，请使用此模板:

```markdown
## Description
简洁描述你的改动

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issues
Closes #123

## Changes Made
- 改动 1
- 改动 2
- 改动 3

## Testing
描述你的测试方法

## Checklist
- [ ] Tests pass locally
- [ ] No new warnings
- [ ] Documentation updated
- [ ] Commit messages follow convention
```

---

## 报告 Bug (Bug Reports)

### 创建 Bug Report

点击 [GitHub Issues](https://github.com/yourusername/H2Q-Evo/issues/new) 并选择 "Bug Report" 模板

**必须包含**:
1. **标题**: 清晰简洁的问题描述
2. **环境**: Python 版本、OS、PyTorch 版本等
3. **步骤**: 重现问题的具体步骤
4. **预期**: 预期的行为
5. **实际**: 实际的行为/错误消息
6. **日志**: 完整的错误跟踪

**示例**:

```markdown
## Bug: Quaternion normalization produces NaN values

### Environment
- Python: 3.10
- PyTorch: 2.0
- OS: macOS 13
- H2Q-Evo: v0.1

### Steps to Reproduce
1. Create quaternion with all zero components
2. Call normalize()
3. Observe NaN values

### Expected
Should return unit quaternion or raise ValueError

### Actual
```python
q = Quaternion(0, 0, 0, 0)
q_norm = q.normalize()  # Returns all NaN
```

### Error Log
```
RuntimeError: Cannot normalize zero quaternion
```
```

---

## 建议功能 (Feature Requests)

### 创建功能请求

点击 [GitHub Issues](https://github.com/yourusername/H2Q-Evo/issues/new) 并选择 "Feature Request" 模板

**应包含**:
1. **问题**: 要解决的问题
2. **解决方案**: 你的建议解决方案
3. **替代方案**: 考虑过的其他方案
4. **影响**: 对项目的潜在影响

**示例**:

```markdown
## Feature: GPU-accelerated quaternion operations

### Problem
目前四元数操作在 CPU 上运行，限制了处理速度

### Proposed Solution
实现 CUDA 核心进行四元数乘法、归一化等基本操作

### Alternative Solutions
- 使用 CuPy（性能可能较低）
- 等待 PyTorch 原生支持

### Potential Impact
- 推理速度可能提升 50-100x
- 训练速度提升 20-30x
- 开启 GPU 上的边缘部署
```

---

## 代码审查 (Code Review)

### 审查清单 (Review Checklist)

当审查 PR 时，请检查:

- ✅ 代码遵循风格指南
- ✅ 包含适当的单元测试
- ✅ 文档字符串已更新
- ✅ 没有性能退化
- ✅ 向后兼容（或明确说明破坏性更改）
- ✅ Commit 消息清晰明确

### 建设性反馈

提供有帮助的审查意见:

```markdown
# 好的反馈示例 ✅
考虑使用列表推导式来提高性能。
根据 PEP 8，函数长度应保持在 50 行以内。

# 不好的反馈示例 ❌
这太糟糕了
代码很烂
```

---

## CI/CD 流程

### 自动检查

每个 PR 将自动运行:

```bash
# 代码格式检查
black --check h2q_project/

# 代码质量检查
flake8 h2q_project/
mypy h2q_project/

# 测试套件
pytest --cov=h2q_project
pytest tests/unit
pytest tests/integration

# 文档构建
sphinx-build docs/ docs/_build/
```

所有检查必须通过才能合并。

---

## 获得帮助 (Getting Help)

### 资源

- **文档**: [README.md](./README.md)
- **讨论**: [GitHub Discussions](https://github.com/yourusername/H2Q-Evo/discussions)
- **问题**: [GitHub Issues](https://github.com/yourusername/H2Q-Evo/issues)

### 联系维护者

- 📧 Email: [your-email@example.com]
- 💬 GitHub: [@yourusername](https://github.com/yourusername)

---

## 许可证 (License)

通过贡献到此项目，您同意您的贡献将在 MIT 许可证下发布。

**By contributing to H2Q-Evo, you agree that your contributions will be licensed under its MIT License.**

---

## 致谢 (Acknowledgments)

感谢所有贡献者帮助我们构建更好的 AGI 框架！

*Thank you for helping us build the future of AGI!* 🚀

---

**最后更新** (Last Updated): 2026-01-19
