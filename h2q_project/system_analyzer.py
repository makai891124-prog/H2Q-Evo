"""
H2Q-Evo 系统分析器 - 代码关系网络与健壮性分析
生成完整的依赖关系图和生产就绪验证报告
"""

import ast
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import importlib.util
import sys

@dataclass
class ComponentMetrics:
    """组件指标"""
    name: str
    file_path: str
    lines_of_code: int
    complexity: int
    dependencies: List[str]
    dependents: List[str]
    test_coverage: bool
    has_error_handling: bool
    has_validation: bool
    version_controlled: bool
    robustness_score: float

@dataclass
class SystemHealthReport:
    """系统健康报告"""
    total_components: int
    critical_components: List[str]
    dependency_graph: Dict[str, List[str]]
    circular_dependencies: List[Tuple[str, str]]
    untested_components: List[str]
    missing_error_handling: List[str]
    robustness_scores: Dict[str, float]
    production_readiness_score: float
    recommendations: List[str]

class CodeAnalyzer(ast.NodeVisitor):
    """AST 代码分析器"""
    
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.complexity = 0
        self.has_try_except = False
        self.has_assertions = False
        self.has_type_hints = False
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        # 检查类型提示
        if node.returns or any(arg.annotation for arg in node.args.args):
            self.has_type_hints = True
        self.generic_visit(node)
        
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.has_try_except = True
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Assert(self, node):
        self.has_assertions = True
        self.generic_visit(node)

class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.components = {}
        
    def analyze_file(self, file_path: Path) -> Optional[ComponentMetrics]:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 跳过空文件
            if not content.strip():
                return None
                
            tree = ast.parse(content)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            # 计算行数
            lines = content.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            
            # 计算鲁棒性评分
            robustness = self._calculate_robustness(
                analyzer.has_try_except,
                analyzer.has_assertions,
                analyzer.has_type_hints,
                analyzer.complexity,
                len(code_lines)
            )
            
            rel_path = file_path.relative_to(self.project_root)
            
            metrics = ComponentMetrics(
                name=str(rel_path),
                file_path=str(file_path),
                lines_of_code=len(code_lines),
                complexity=analyzer.complexity,
                dependencies=list(analyzer.imports),
                dependents=[],
                test_coverage=self._has_test_file(rel_path),
                has_error_handling=analyzer.has_try_except,
                has_validation=analyzer.has_assertions,
                version_controlled=True,  # 假设在 git 中
                robustness_score=robustness
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
            
    def _calculate_robustness(self, has_error: bool, has_validation: bool,
                             has_types: bool, complexity: int, loc: int) -> float:
        """计算鲁棒性评分 (0-100)"""
        score = 0.0
        
        # 错误处理 (30分)
        if has_error:
            score += 30
            
        # 输入验证 (25分)
        if has_validation:
            score += 25
            
        # 类型提示 (20分)
        if has_types:
            score += 20
            
        # 复杂度惩罚 (最多扣15分)
        if loc > 0:
            complexity_ratio = complexity / loc
            if complexity_ratio > 0.3:
                score -= 15
            elif complexity_ratio > 0.2:
                score -= 10
            elif complexity_ratio > 0.1:
                score -= 5
                
        # 代码量合理性 (10分)
        if 10 < loc < 500:
            score += 10
        elif 500 <= loc < 1000:
            score += 5
            
        # 基础分 (5分)
        score += 5
        
        return max(0, min(100, score))
        
    def _has_test_file(self, file_path: Path) -> bool:
        """检查是否有对应的测试文件"""
        test_dir = self.project_root / "tests"
        if not test_dir.exists():
            return False
            
        test_name = f"test_{file_path.stem}.py"
        return (test_dir / test_name).exists()
        
    def build_dependency_graph(self):
        """构建完整的依赖关系图"""
        # 扫描所有 Python 文件
        python_files = list(self.project_root.rglob("*.py"))
        
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            # 跳过测试文件和虚拟环境
            if 'test' in str(file_path) or 'venv' in str(file_path) or '.pyenv' in str(file_path):
                continue
                
            metrics = self.analyze_file(file_path)
            if metrics:
                self.components[metrics.name] = metrics
                
                # 构建依赖图
                for dep in metrics.dependencies:
                    self.dependency_graph[metrics.name].add(dep)
                    self.reverse_graph[dep].add(metrics.name)
                    
    def find_circular_dependencies(self) -> List[Tuple[str, str]]:
        """查找循环依赖"""
        circular = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path[:])
                elif neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) >= 2:
                        circular.append((cycle[0], cycle[-1]))
                        
            rec_stack.remove(node)
            
        for component in self.components:
            if component not in visited:
                dfs(component, [])
                
        return list(set(circular))
        
    def identify_critical_components(self) -> List[str]:
        """识别关键组件（被多个组件依赖）"""
        critical = []
        for component, dependents in self.reverse_graph.items():
            if len(dependents) >= 3:  # 被3个以上组件依赖
                critical.append(component)
        return sorted(critical, key=lambda x: len(self.reverse_graph[x]), reverse=True)
        
    def generate_health_report(self) -> SystemHealthReport:
        """生成系统健康报告"""
        circular_deps = self.find_circular_dependencies()
        critical_components = self.identify_critical_components()
        
        untested = [
            name for name, metrics in self.components.items()
            if not metrics.test_coverage and 'h2q/core' in name
        ]
        
        missing_error_handling = [
            name for name, metrics in self.components.items()
            if not metrics.has_error_handling and metrics.lines_of_code > 50
        ]
        
        robustness_scores = {
            name: metrics.robustness_score
            for name, metrics in self.components.items()
        }
        
        # 计算生产就绪分数
        avg_robustness = np.mean(list(robustness_scores.values())) if robustness_scores else 0
        test_coverage = sum(1 for m in self.components.values() if m.test_coverage) / max(len(self.components), 1)
        error_handling = sum(1 for m in self.components.values() if m.has_error_handling) / max(len(self.components), 1)
        
        production_score = (
            avg_robustness * 0.4 +
            test_coverage * 100 * 0.3 +
            error_handling * 100 * 0.2 -
            len(circular_deps) * 5 -
            len(untested) * 2
        )
        production_score = max(0, min(100, production_score))
        
        # 生成建议
        recommendations = []
        if circular_deps:
            recommendations.append(f"解决 {len(circular_deps)} 个循环依赖")
        if untested:
            recommendations.append(f"为 {len(untested)} 个核心组件添加测试")
        if missing_error_handling:
            recommendations.append(f"为 {len(missing_error_handling)} 个组件添加错误处理")
        if avg_robustness < 60:
            recommendations.append("提高整体代码鲁棒性（当前平均: {:.1f}/100）".format(avg_robustness))
            
        return SystemHealthReport(
            total_components=len(self.components),
            critical_components=critical_components[:10],
            dependency_graph={k: list(v) for k, v in self.dependency_graph.items()},
            circular_dependencies=circular_deps,
            untested_components=untested[:20],
            missing_error_handling=missing_error_handling[:20],
            robustness_scores=robustness_scores,
            production_readiness_score=production_score,
            recommendations=recommendations
        )

def generate_detailed_report(report: SystemHealthReport, output_path: Path):
    """生成详细的分析报告"""
    from datetime import datetime
    report_md = f"""# H2Q-Evo 系统健康与代码关系网络分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 总体概览

- **组件总数**: {report.total_components}
- **生产就绪度**: {report.production_readiness_score:.1f}/100
- **循环依赖数**: {len(report.circular_dependencies)}
- **未测试组件**: {len(report.untested_components)}
- **缺少错误处理**: {len(report.missing_error_handling)}

## 🎯 关键组件（高依赖度）

被最多组件依赖的核心模块：

"""
    
    for i, comp in enumerate(report.critical_components, 1):
        report_md += f"{i}. `{comp}`\n"
        
    report_md += f"""
## ⚠️ 循环依赖

检测到 {len(report.circular_dependencies)} 个循环依赖：

"""
    
    for src, dst in report.circular_dependencies[:10]:
        report_md += f"- `{src}` ⟷ `{dst}`\n"
        
    report_md += f"""
## 🧪 测试覆盖缺口

以下核心组件缺少测试：

"""
    
    for comp in report.untested_components[:15]:
        report_md += f"- `{comp}`\n"
        
    report_md += f"""
## 🛡️ 错误处理缺失

以下组件需要添加错误处理（>50 行代码）：

"""
    
    for comp in report.missing_error_handling[:15]:
        report_md += f"- `{comp}`\n"
        
    report_md += f"""
## 📈 鲁棒性评分

### 得分分布
"""
    
    # 计算评分分布
    scores = list(report.robustness_scores.values())
    if scores:
        excellent = sum(1 for s in scores if s >= 80)
        good = sum(1 for s in scores if 60 <= s < 80)
        fair = sum(1 for s in scores if 40 <= s < 60)
        poor = sum(1 for s in scores if s < 40)
        
        report_md += f"""
- 优秀 (≥80): {excellent} ({excellent/len(scores)*100:.1f}%)
- 良好 (60-79): {good} ({good/len(scores)*100:.1f}%)
- 一般 (40-59): {fair} ({fair/len(scores)*100:.1f}%)
- 较差 (<40): {poor} ({poor/len(scores)*100:.1f}%)

### 最佳实践组件 (评分≥80)

"""
        best_components = sorted(
            report.robustness_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for comp, score in best_components:
            if score >= 80:
                report_md += f"- `{comp}`: {score:.1f}/100\n"
                
        report_md += f"""
### 需要改进的组件 (评分<40)

"""
        worst_components = sorted(
            report.robustness_scores.items(),
            key=lambda x: x[1]
        )[:10]
        
        for comp, score in worst_components:
            if score < 40:
                report_md += f"- `{comp}`: {score:.1f}/100\n"
                
    report_md += f"""
## 🎯 改进建议

"""
    
    for i, rec in enumerate(report.recommendations, 1):
        report_md += f"{i}. {rec}\n"
        
    report_md += """
## 📊 依赖关系网络拓扑

### 核心依赖层级

```
DiscreteDecisionEngine (核心决策引擎)
  ├── SpectralShiftTracker (谱移跟踪器)
  ├── QuaternionicManifold (四元数流形)
  └── LatentConfig (配置管理)

AutonomousSystem (自主系统)
  ├── DiscreteDecisionEngine
  ├── TopologicalPhaseQuantizer
  └── ReversibleKernel

SpectralShiftTracker (谱移跟踪)
  └── SU(2) 流形投影

```

## 🔒 版本控制建议

### 核心算法版本快照

建议为以下核心组件创建版本快照：

1. **DiscreteDecisionEngine** - 决策引擎核心逻辑
2. **SpectralShiftTracker** - 谱移计算公式
3. **QuaternionicManifold** - 四元数流形操作
4. **ReversibleKernel** - 可逆核心函数
5. **AutonomousSystem** - 自主系统集成

### 推荐版本控制策略

```python
# 算法版本标记
ALGORITHM_VERSION = {
    "discrete_decision_engine": "2.1.0",
    "spectral_shift_tracker": "1.5.0",
    "quaternionic_manifold": "1.8.0",
    "reversible_kernel": "1.3.0",
    "autonomous_system": "2.0.0"
}

# API 兼容性标记
API_COMPATIBILITY = {
    "min_version": "2.0.0",
    "max_version": "3.0.0",
    "breaking_changes": []
}
```

## 📋 生产环境检查清单

- [ ] 所有核心组件有单元测试
- [ ] 所有 API 接口有集成测试
- [ ] 错误处理覆盖所有外部调用
- [ ] 输入验证防止无效数据
- [ ] 性能监控和日志记录
- [ ] 降级策略和熔断机制
- [ ] 健康检查端点
- [ ] 版本兼容性检查
- [ ] 文档完整且更新
- [ ] 部署回滚预案

## 🚀 下一步行动

### 高优先级

1. 解决所有循环依赖
2. 为核心组件添加错误处理
3. 补充缺失的单元测试
4. 实现算法版本控制

### 中优先级

5. 重构高复杂度组件
6. 添加类型提示和文档
7. 实现性能监控
8. 创建健康检查系统

### 低优先级

9. 优化代码结构
10. 改进日志记录
11. 增强可观测性
12. 完善文档和示例

---

*报告由 H2Q-Evo 系统分析器自动生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"✅ 详细报告已生成: {output_path}")

def main():
    """主函数"""
    try:
        import pandas as pd
    except ImportError:
        print("警告: pandas 未安装，使用基础功能")
        
    project_root = Path(__file__).parent
    
    print("🔍 开始分析 H2Q-Evo 系统...")
    print(f"📁 项目根目录: {project_root}")
    
    analyzer = DependencyAnalyzer(project_root)
    analyzer.build_dependency_graph()
    
    print(f"✅ 已分析 {len(analyzer.components)} 个组件")
    
    print("📊 生成系统健康报告...")
    report = analyzer.generate_health_report()
    
    print(f"""
╔════════════════════════════════════════════╗
║     H2Q-Evo 系统健康报告                   ║
╠════════════════════════════════════════════╣
║ 总组件数:        {report.total_components:4d}                  ║
║ 生产就绪度:      {report.production_readiness_score:5.1f}/100             ║
║ 关键组件数:      {len(report.critical_components):4d}                  ║
║ 循环依赖:        {len(report.circular_dependencies):4d}                  ║
║ 未测试组件:      {len(report.untested_components):4d}                  ║
║ 缺少错误处理:    {len(report.missing_error_handling):4d}                  ║
╚════════════════════════════════════════════╝
    """)
    
    # 保存报告
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    # JSON 报告
    json_path = output_dir / "system_health_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_components': report.total_components,
            'production_readiness_score': report.production_readiness_score,
            'critical_components': report.critical_components,
            'circular_dependencies': [list(cd) for cd in report.circular_dependencies],
            'untested_components': report.untested_components,
            'missing_error_handling': report.missing_error_handling,
            'robustness_scores': report.robustness_scores,
            'recommendations': report.recommendations
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 报告已保存: {json_path}")
    
    # Markdown 报告
    md_path = output_dir / "SYSTEM_HEALTH_REPORT.md"
    generate_detailed_report(report, md_path)
    
    # 保存依赖图
    graph_path = output_dir / "dependency_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(report.dependency_graph, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 依赖关系图已保存: {graph_path}")
    
    print("\n" + "="*50)
    print("🎯 主要发现:")
    print("="*50)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
        
    print("\n" + "="*50)
    print("✅ 系统分析完成!")
    print("="*50)
    
    return report

if __name__ == "__main__":
    main()
