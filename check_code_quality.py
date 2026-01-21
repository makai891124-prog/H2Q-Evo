#!/usr/bin/env python3
"""
深度代码质量检查 - 检查隐藏错误和无用代码
"""
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class CodeQualityAnalyzer:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {
            "unused_imports": 0,
            "unused_variables": 0,
            "empty_functions": 0,
            "dead_code": 0,
            "bare_excepts": 0,
        }

    def analyze_file(self, filepath: Path):
        """分析单个Python文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
        except SyntaxError as e:
            self.issues.append(f"❌ {filepath}: 语法错误 {e}")
            return
        except Exception as e:
            self.issues.append(f"❌ {filepath}: 解析失败 {e}")
            return

        # 遍历AST检查各种问题
        self._check_bare_excepts(tree, filepath)
        self._check_unused_imports(tree, filepath, content)
        self._check_dead_code(tree, filepath)
        self._check_empty_functions(tree, filepath)

    def _check_bare_excepts(self, tree: ast.AST, filepath: Path):
        """检查裸except块"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # 裸except
                    self.stats["bare_excepts"] += 1
                    self.warnings.append(
                        f"⚠️  {filepath}:{node.lineno} - 使用了裸except (建议指定异常类型)"
                    )

    def _check_unused_imports(self, tree: ast.AST, filepath: Path, content: str):
        """检查未使用的导入"""
        # 提取所有导入
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno

        # 检查哪些被使用
        source_lines = content.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                imports.pop(node.id, None)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    imports.pop(node.value.id, None)

        # 报告未使用的导入
        for name, lineno in imports.items():
            if not name.startswith('_'):  # 忽略private imports
                self.stats["unused_imports"] += 1
                self.warnings.append(
                    f"⚠️  {filepath}:{lineno} - 未使用的导入: {name}"
                )

    def _check_dead_code(self, tree: ast.AST, filepath: Path):
        """检查死亡代码（if False等）"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # 检查 if False 或 if 0
                if isinstance(node.test, ast.Constant):
                    if node.test.value is False or node.test.value == 0:
                        self.stats["dead_code"] += 1
                        self.warnings.append(
                            f"⚠️  {filepath}:{node.lineno} - 死亡代码 (if False/0)"
                        )

    def _check_empty_functions(self, tree: ast.AST, filepath: Path):
        """检查空函数"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 检查函数体是否为空或仅包含pass/docstring
                body = node.body
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    self.stats["empty_functions"] += 1
                    self.warnings.append(
                        f"ℹ️  {filepath}:{node.lineno} - 空函数: {node.name}()"
                    )

    def report(self):
        """生成报告"""
        print("=" * 80)
        print("代码质量检查报告")
        print("=" * 80)

        if self.issues:
            print(f"\n🔴 严重问题 ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")

        if self.warnings:
            print(f"\n🟡 警告 ({len(self.warnings)}):")
            for warning in self.warnings[:20]:  # 只显示前20个
                print(f"   {warning}")
            if len(self.warnings) > 20:
                print(f"   ... 还有 {len(self.warnings) - 20} 个警告")

        print(f"\n📊 统计:")
        for metric, count in self.stats.items():
            if count > 0:
                print(f"   • {metric}: {count}")

        # 总体评价
        total_problems = len(self.issues) + len(self.warnings)
        if total_problems == 0:
            print(f"\n✅ 代码质量优秀！")
            return 0
        elif total_problems < 10:
            print(f"\n✅ 代码质量良好（{total_problems}个小问题）")
            return 0
        else:
            print(f"\n⚠️  检测到{total_problems}个问题")
            return 1


# 主程序
if __name__ == "__main__":
    analyzer = CodeQualityAnalyzer()

    # 扫描关键Python文件
    key_files = [
        "/Users/imymm/H2Q-Evo/comprehensive_validation_final.py",
        "/Users/imymm/H2Q-Evo/comprehensive_validation_v2.py",
        "/Users/imymm/H2Q-Evo/verify_geometric_automation.py",
        "/Users/imymm/H2Q-Evo/api_inspection.py",
        "/Users/imymm/H2Q-Evo/h2q_project/run_experiment_fixed.py",
    ]

    print("扫描文件...")
    for filepath in key_files:
        p = Path(filepath)
        if p.exists():
            print(f"  检查: {p.name}")
            analyzer.analyze_file(p)
        else:
            print(f"  ⚠️  {p.name} 不存在")

    # 生成报告
    exit_code = analyzer.report()
    sys.exit(exit_code)
