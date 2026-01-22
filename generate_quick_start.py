#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动指南 - 如何使用完整的自我进化系统
"""

QUICK_START_GUIDE = """
# 🚀 完整自我进化AGI系统 - 快速启动指南

## ⚡ 30秒快速开始

### 1. 环境准备
```bash
cd /Users/imymm/H2Q-Evo

# 安装依赖
pip install google-generativeai cryptography torch transformers

# 设置API密钥 (可选)
export GEMINI_API_KEY="your-api-key-here"
```

### 2. 运行演示
```bash
# 运行完整的自我进化循环演示
PYTHONPATH=. python3 h2q_project/h2q/agi/complete_evolution_demo.py
```

### 3. 查看结果
```bash
# 查看演示结果
cat complete_evolution_results/complete_demo_*.json | python3 -m json.tool

# 查看论证分析
cat complete_evolution_results/analysis/formal_arguments.json
```

---

## 📚 模块使用指南

### A. 使用 Gemini 集成

```python
from h2q_project.h2q.agi.gemini_cli_integration import GeminiCLIIntegration

# 初始化
gemini = GeminiCLIIntegration(api_key="your-key")

# 查询
result = gemini.query("什么是自我进化的AGI？")
print(result['response'])

# 分析决策
feedback = gemini.analyze_decision(
    decision={'answer': '...'},
    reasoning='...'
)
print(feedback['analysis'])
```

### B. 使用进化框架

```python
from h2q_project.h2q.agi.template_evolution_framework import (
    TemplateEvolutionFramework, EvolutionPhase
)

# 初始化框架
framework = TemplateEvolutionFramework()

# 创建模板
template = framework.create_template(
    name="我的进化实验",
    max_iterations=5,
    convergence_threshold=0.85
)

# 定义生成和求解函数
def problem_generator(state):
    return [{'question': 'test question'}]

def problem_solver(state):
    return [{'answer': 'test answer'}]

# 运行循环
result = framework.run_evolution_cycle(
    template=template,
    initial_state={},
    problem_generator=problem_generator,
    solver=problem_solver
)
```

### C. 使用自我进化循环

```python
from h2q_project.h2q.agi.self_evolution_loop import SelfEvolutionLoop

# 初始化
loop = SelfEvolutionLoop(gemini, m24_protocol, framework)

# 运行完整进化
result = loop.run_complete_evolution_cycle(
    num_iterations=3,
    num_problems_per_iteration=2
)

# 获取总结
summary = loop.get_evolution_summary()
print(f"总问题数: {summary['total_problems']}")
print(f"总解答数: {summary['total_solutions']}")
```

### D. 使用论证分析

```python
from h2q_project.h2q.agi.evolution_argumentation_analysis import EvolutionProcessAnalysis

# 初始化
analysis = EvolutionProcessAnalysis(gemini)

# 生成论证
arguments = analysis.generate_formal_argument_chain()
formalization = analysis.generate_process_formalization()

# 保存分析
analysis.save_complete_argumentation()
```

---

## 🎯 常见使用场景

### 场景1: 快速进化演示
```bash
# 运行完整演示 (包含所有功能)
python3 h2q_project/h2q/agi/complete_evolution_demo.py
```

### 场景2: 测试问题生成
```python
from h2q_project.h2q.agi.self_evolution_loop import AutomaticProblemGenerator

gen = AutomaticProblemGenerator()
problems = gen.generate_problems({}, num_problems=5)
for p in problems:
    print(f"Q: {p['question']}")
```

### 场景3: 本地完全自持循环
```python
# 无需Gemini，完全本地运行
loop.demonstrate_local_self_sufficiency()
```

### 场景4: 自定义进化策略
```python
# 创建自定义模板
my_template = framework.create_template(
    name="自定义策略",
    phases=[
        EvolutionPhase.INITIALIZATION,
        EvolutionPhase.PROBLEM_GENERATION,
        EvolutionPhase.SOLUTION_ATTEMPT,
        EvolutionPhase.EVALUATION
    ],
    max_iterations=10,
    use_external_feedback=False,  # 不使用Gemini
    use_honesty_verification=True  # 仅用M24
)
```

---

## 🔍 输出文件说明

### 演示结果目录
```
complete_evolution_results/
├── complete_demo_YYYYMMDD_HHMMSS.json    # 演示结果
├── analysis/
│   ├── formal_arguments.json              # 5个论证
│   └── process_formalization.json         # 形式化模型
└── evolution_XXXXXX.json                  # 进化日志
```

### JSON 结果结构

**演示结果文件：**
```json
{
  "start_time": "...",
  "phases": [
    {
      "phase": "论证生成",
      "argument_chain_sections": 5,
      "local_sufficiency_proven": "yes"
    },
    ...
  ],
  "end_time": "..."
}
```

**论证文件：**
```json
{
  "title": "自动进化AGI系统的形式化论证",
  "sections": [
    {
      "name": "问题自动生成的合理性",
      "claim": "...",
      "premises": [...],
      "evidence": [...],
      "conclusion": "✓ ..."
    },
    ...
  ]
}
```

---

## 📊 性能监控

### 查看系统统计

```python
# 获取Gemini调用统计
stats = gemini.get_call_statistics()
print(f"总调用数: {stats['total_calls']}")
print(f"成功率: {stats['success_rate']:.1%}")

# 获取进化总结
summary = loop.get_evolution_summary()
print(f"生成问题: {summary['total_problems']}")
print(f"生成解答: {summary['total_solutions']}")
```

### 监控进化过程

```python
# 检查性能指标
metrics = framework.performance_metrics
print(f"初始性能: {metrics['initial']['overall_score']:.2f}")
print(f"当前性能: {metrics['current']['overall_score']:.2f}")
print(f"最佳性能: {metrics['best']['overall_score']:.2f}")
```

---

## 🛠️ 故障排除

### 问题1: Gemini API 不可用
**症状:** `⚠️ GEMINI_API_KEY未设置`

**解决:**
```bash
export GEMINI_API_KEY="your-api-key"
# 或运行本地模式 (自动启用)
```

### 问题2: 缓存文件过期
**症状:** 重复调用Gemini

**解决:** 清理缓存
```bash
rm -rf gemini_cache/
# 缓存24小时后自动过期
```

### 问题3: 内存不足
**症状:** MemoryError

**解决:** 减少并发
```python
result = gemini.batch_query(prompts, max_workers=1)
```

---

## 📈 进阶使用

### 自定义问题领域

```python
class CustomProblemGenerator(AutomaticProblemGenerator):
    def generate_problems(self, state, num_problems):
        # 添加自定义问题
        return [
            {
                "domain": "物理",
                "question": "加速度是什么？",
                "difficulty": "简单"
            },
            ...
        ]
```

### 自定义求解策略

```python
class SmartProblemSolver(ProblemSolver):
    def solve(self, problem):
        # 实现更智能的求解
        if problem['domain'] == '数学':
            return self._solve_math(problem)
        elif problem['domain'] == '代码':
            return self._solve_code(problem)
        else:
            return super().solve(problem)
```

### 集成自有模型

```python
# 替换Gemini集成
my_gemini = CustomGeminiIntegration()
loop = SelfEvolutionLoop(my_gemini, m24_protocol, framework)
```

---

## 📚 文档和资源

### 完整文档
- `SELF_EVOLUTION_IMPLEMENTATION_GUIDE.md` - 详细实现指南
- `SELF_EVOLUTION_SYSTEM_FINAL_REPORT.json` - 最终报告

### 源代码
- `h2q_project/h2q/agi/gemini_cli_integration.py` - Gemini集成
- `h2q_project/h2q/agi/template_evolution_framework.py` - 进化框架
- `h2q_project/h2q/agi/self_evolution_loop.py` - 自我进化循环
- `h2q_project/h2q/agi/evolution_argumentation_analysis.py` - 论证分析
- `h2q_project/h2q/agi/complete_evolution_demo.py` - 完整演示

---

## 🤝 贡献指南

欢迎贡献！可以帮助：

1. **新的问题生成策略** - 实现更多领域的问题
2. **高效的求解算法** - 提升解答质量
3. **改进的验证机制** - 增强安全性
4. **性能优化** - 加快处理速度
5. **文档和示例** - 改善学习体验

---

## ❓ 常见问题

**Q: 我可以离线使用吗？**
A: 是的！系统具有完全的本地自持能力。不设置Gemini API时自动使用本地模式。

**Q: 性能如何？**
A: 完整的进化循环通常在5-30分钟内完成，取决于问题复杂度。

**Q: 能否用于生产环境？**
A: 可以。系统已实现生产级别的错误处理、日志和验证。

**Q: 如何扩展系统？**
A: 系统采用模块化设计，便于扩展。参考自定义使用部分。

**Q: 是否支持并发？**
A: 是的，Gemini集成支持并发查询。

---

## 📞 支持

- 查看日志: `h2q_project/h2q/agi/logs/`
- GitHub Issues: 报告问题
- 讨论区: 技术讨论

---

## 📄 许可证

MIT License - 开源可用

---

**版本:** 1.0.0
**最后更新:** 2026-01-22
**状态:** ✓ 生产就绪

祝您使用愉快！🎉
"""


if __name__ == "__main__":
    from pathlib import Path
    
    # 保存指南
    guide_path = Path("QUICK_START_GUIDE.md")
    guide_path.write_text(QUICK_START_GUIDE, encoding='utf-8')
    
    print(f"✓ 快速启动指南已生成: {guide_path}")
    print(f"  大小: {guide_path.stat().st_size / 1024:.1f} KB")
    print("\n快速启动指南已准备好！")
