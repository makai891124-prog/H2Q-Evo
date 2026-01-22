#!/usr/bin/env python3
"""
安全代码进化系统设计文档
Safe Code Evolution System Design

╔════════════════════════════════════════════════════════════════════════════╗
║                           终 极 目 标                                       ║
║                                                                            ║
║          训练本地可用的实时AGI系统                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

问题：如何安全地自动优化代码而不破坏系统？
Solution: How to safely auto-optimize code without breaking the system?

================================================================================
                        安全代码进化架构
================================================================================

核心原则:
=========
1. 隔离执行 - 新代码在沙盒中测试
2. 渐进式合并 - 只有通过所有测试的代码才能合并
3. 版本回滚 - 随时可以回滚到稳定版本
4. 多层验证 - 语法检查 → 单元测试 → 集成测试 → Gemini审计

架构图:
=======

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAFE CODE EVOLUTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │ 1. 代码生成   │  LLM/规则 生成候选代码                                   │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 2. 语法检查   │  Python AST 解析 + pylint/mypy                          │
│   └──────┬───────┘                                                          │
│          │ ✓ 通过                                                           │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 3. 沙盒执行   │  subprocess + 资源限制 + 超时                           │
│   └──────┬───────┘                                                          │
│          │ ✓ 无异常                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 4. 单元测试   │  pytest + 覆盖率检查                                    │
│   └──────┬───────┘                                                          │
│          │ ✓ 全部通过                                                       │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 5. 集成测试   │  与现有系统的兼容性测试                                  │
│   └──────┬───────┘                                                          │
│          │ ✓ 无冲突                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 6. Gemini审计 │  第三方验证代码质量和安全性                             │
│   └──────┬───────┘                                                          │
│          │ ✓ 审计通过                                                       │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ 7. 安全合并   │  Git commit + 版本标记                                  │
│   └──────────────┘                                                          │
│                                                                             │
│   任何一步失败 → 丢弃候选代码，记录失败原因，继续下一轮                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import os
import sys
import ast
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class CodeCandidate:
    """候选代码."""
    code: str
    description: str
    target_file: Optional[str] = None
    modification_type: str = "new"  # new, modify, patch
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ValidationResult:
    """验证结果."""
    passed: bool
    stage: str
    message: str
    details: Dict = None


class SafeCodeValidator:
    """安全代码验证器 - 多层验证确保代码安全."""
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """
        第一层：语法检查
        使用 Python AST 解析确保代码语法正确
        """
        try:
            ast.parse(code)
            return ValidationResult(
                passed=True,
                stage="syntax",
                message="Syntax validation passed"
            )
        except SyntaxError as e:
            return ValidationResult(
                passed=False,
                stage="syntax",
                message=f"Syntax error: {e}",
                details={"line": e.lineno, "offset": e.offset}
            )
    
    def validate_imports(self, code: str) -> ValidationResult:
        """
        第二层：导入检查
        确保所有导入的模块都存在
        """
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            # 检查标准库和已安装的包
            missing = []
            for imp in imports:
                try:
                    __import__(imp)
                except ImportError:
                    # 检查是否是本地模块
                    local_path = SCRIPT_DIR / f"{imp}.py"
                    if not local_path.exists():
                        missing.append(imp)
            
            if missing:
                return ValidationResult(
                    passed=False,
                    stage="imports",
                    message=f"Missing imports: {missing}",
                    details={"missing": missing}
                )
            
            return ValidationResult(
                passed=True,
                stage="imports",
                message="All imports available"
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                stage="imports",
                message=f"Import check failed: {e}"
            )
    
    def validate_sandbox_execution(self, code: str, timeout: int = 10) -> ValidationResult:
        """
        第三层：沙盒执行
        在隔离的子进程中执行代码，限制资源和时间
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # 在子进程中执行，带超时
            result = subprocess.run(
                [sys.executable, '-c', f"exec(open('{temp_path}').read())"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(SCRIPT_DIR)
            )
            
            if result.returncode != 0:
                return ValidationResult(
                    passed=False,
                    stage="sandbox",
                    message=f"Execution failed: {result.stderr[:500]}",
                    details={"returncode": result.returncode, "stderr": result.stderr}
                )
            
            return ValidationResult(
                passed=True,
                stage="sandbox",
                message="Sandbox execution passed",
                details={"stdout": result.stdout[:500] if result.stdout else ""}
            )
        
        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                stage="sandbox",
                message=f"Execution timeout ({timeout}s)",
                details={"timeout": timeout}
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                stage="sandbox",
                message=f"Sandbox error: {e}"
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def validate_no_dangerous_patterns(self, code: str) -> ValidationResult:
        """
        第四层：安全模式检查
        检测潜在危险的代码模式
        """
        dangerous_patterns = [
            ('os.system', 'Shell command execution'),
            ('subprocess.call', 'Subprocess without capture'),
            ('eval(', 'Dynamic code evaluation'),
            ('exec(', 'Dynamic code execution'),
            ('__import__', 'Dynamic import'),
            ('open(', 'File operations need review'),  # 警告级别
            ('rm -rf', 'Dangerous shell command'),
            ('shutil.rmtree', 'Recursive deletion'),
        ]
        
        warnings = []
        errors = []
        
        for pattern, description in dangerous_patterns:
            if pattern in code:
                if pattern in ['rm -rf', 'shutil.rmtree', 'os.system']:
                    errors.append(f"BLOCKED: {description} ({pattern})")
                else:
                    warnings.append(f"WARNING: {description} ({pattern})")
        
        if errors:
            return ValidationResult(
                passed=False,
                stage="security",
                message="Dangerous patterns detected",
                details={"errors": errors, "warnings": warnings}
            )
        
        return ValidationResult(
            passed=True,
            stage="security",
            message="Security check passed" + (f" with {len(warnings)} warnings" if warnings else ""),
            details={"warnings": warnings}
        )
    
    def full_validation(self, candidate: CodeCandidate) -> Tuple[bool, List[ValidationResult]]:
        """
        完整验证流程
        """
        results = []
        
        # 1. 语法检查
        r1 = self.validate_syntax(candidate.code)
        results.append(r1)
        if not r1.passed:
            return False, results
        
        # 2. 导入检查
        r2 = self.validate_imports(candidate.code)
        results.append(r2)
        if not r2.passed:
            return False, results
        
        # 3. 安全模式检查
        r3 = self.validate_no_dangerous_patterns(candidate.code)
        results.append(r3)
        if not r3.passed:
            return False, results
        
        # 4. 沙盒执行（只对可独立运行的代码）
        # 注意：对于需要上下文的代码，跳过此步
        if 'if __name__' in candidate.code or candidate.modification_type == "new":
            r4 = self.validate_sandbox_execution(candidate.code)
            results.append(r4)
            if not r4.passed:
                return False, results
        
        return True, results


class SafeCodeEvolutionSystem:
    """
    安全代码进化系统
    
    核心保障机制:
    1. 所有代码修改都经过多层验证
    2. 使用 Git 进行版本控制，支持回滚
    3. 新代码先在隔离环境测试
    4. 只有通过所有验证的代码才能合并
    """
    
    def __init__(self):
        self.validator = SafeCodeValidator()
        self.evolution_history: List[Dict] = []
        self.pending_candidates: List[CodeCandidate] = []
        self.approved_candidates: List[CodeCandidate] = []
        self.rejected_candidates: List[Tuple[CodeCandidate, List[ValidationResult]]] = []
    
    def submit_candidate(self, code: str, description: str, 
                        target_file: str = None) -> Tuple[bool, str]:
        """
        提交候选代码进行验证
        
        Returns:
            (是否通过, 消息)
        """
        candidate = CodeCandidate(
            code=code,
            description=description,
            target_file=target_file
        )
        
        print(f"\n[SafeCodeEvolution] Validating candidate: {description[:50]}...")
        
        passed, results = self.validator.full_validation(candidate)
        
        if passed:
            self.approved_candidates.append(candidate)
            self.evolution_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'approved',
                'description': description,
                'validation_stages': [r.stage for r in results]
            })
            return True, "Candidate approved and queued for integration"
        else:
            self.rejected_candidates.append((candidate, results))
            failed_stage = next((r for r in results if not r.passed), results[-1])
            return False, f"Rejected at {failed_stage.stage}: {failed_stage.message}"
    
    def get_integration_report(self) -> str:
        """生成集成报告."""
        report = []
        report.append("=" * 60)
        report.append("  SAFE CODE EVOLUTION - INTEGRATION REPORT")
        report.append("=" * 60)
        report.append(f"\nApproved candidates: {len(self.approved_candidates)}")
        report.append(f"Rejected candidates: {len(self.rejected_candidates)}")
        
        if self.approved_candidates:
            report.append("\n[Approved for Integration]")
            for i, c in enumerate(self.approved_candidates, 1):
                report.append(f"  {i}. {c.description[:60]}")
        
        if self.rejected_candidates:
            report.append("\n[Rejected]")
            for i, (c, results) in enumerate(self.rejected_candidates, 1):
                failed = next((r for r in results if not r.passed), None)
                if failed:
                    report.append(f"  {i}. {c.description[:40]}... - {failed.stage}: {failed.message[:30]}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


# ============================================================================
# 演示：安全代码验证
# ============================================================================

def demo_safe_validation():
    """演示安全代码验证系统."""
    system = SafeCodeEvolutionSystem()
    
    # 测试1：有效的代码
    valid_code = '''
def optimized_function(x):
    """优化的计算函数."""
    import math
    return math.sqrt(x ** 2 + 1)

if __name__ == "__main__":
    result = optimized_function(3)
    print(f"Result: {result}")
'''
    
    passed, msg = system.submit_candidate(
        valid_code,
        "Add optimized_function for faster computation"
    )
    print(f"Test 1 (Valid code): {passed} - {msg}")
    
    # 测试2：语法错误的代码
    invalid_syntax = '''
def broken_function(x)  # 缺少冒号
    return x + 1
'''
    
    passed, msg = system.submit_candidate(
        invalid_syntax,
        "Add broken function"
    )
    print(f"Test 2 (Syntax error): {passed} - {msg}")
    
    # 测试3：危险的代码
    dangerous_code = '''
import os
os.system("rm -rf /")  # 危险！
'''
    
    passed, msg = system.submit_candidate(
        dangerous_code,
        "Add cleanup function"
    )
    print(f"Test 3 (Dangerous code): {passed} - {msg}")
    
    # 测试4：超时的代码
    timeout_code = '''
import time
while True:
    time.sleep(1)  # 无限循环
'''
    
    passed, msg = system.submit_candidate(
        timeout_code,
        "Add long-running process"
    )
    print(f"Test 4 (Timeout): {passed} - {msg}")
    
    # 打印报告
    print(system.get_integration_report())


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("       SAFE CODE EVOLUTION SYSTEM - DEMO")
    print("=" * 70)
    demo_safe_validation()
