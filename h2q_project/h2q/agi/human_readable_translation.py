#!/usr/bin/env python3
"""
H2Q 人类可读翻译系统 (Human-Readable Translation Layer)

核心理念:
=========
H2Q系统的核心是【编码能力】，不需要还原成人类的现有工具。
但需要提供【翻译能力结构】使得人类能够看懂进行监督。

这个模块实现:
1. H2Q内部编码 → 人类可读文档的翻译
2. 执行过程的可视化
3. 系统决策的解释
4. 监督接口

翻译层设计原则:
==============
- 不改变系统的核心编码方式
- 只提供观察窗口给人类
- 保持翻译的准确性和完整性
- 支持不同详细程度的翻译
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import textwrap


# ============================================================================
# 第一部分: 翻译级别定义
# ============================================================================

class TranslationLevel(Enum):
    """翻译详细程度."""
    MINIMAL = "minimal"       # 最简：只显示关键信息
    STANDARD = "standard"     # 标准：显示主要步骤
    DETAILED = "detailed"     # 详细：显示所有细节
    DEBUG = "debug"           # 调试：包含内部状态


@dataclass
class TranslationContext:
    """翻译上下文 - 存储翻译过程的状态."""
    level: TranslationLevel = TranslationLevel.STANDARD
    show_internal_state: bool = False
    show_confidence: bool = True
    language: str = "zh-CN"  # 可扩展到多语言


# ============================================================================
# 第二部分: 核心翻译器
# ============================================================================

class H2QHumanTranslator:
    """
    H2Q人类翻译器 - 将系统内部表示翻译为人类可理解的格式.
    """
    
    def __init__(self, context: TranslationContext = None):
        self.context = context or TranslationContext()
        
        # 操作码翻译词典
        self.opcode_translations = {
            0x00: ("空操作", "NOP", "系统空转，不执行任何操作"),
            0x01: ("加法", "ADD", "将两个数值相加"),
            0x02: ("减法", "SUB", "从第一个数减去第二个数"),
            0x03: ("乘法", "MUL", "将两个数值相乘"),
            0x04: ("除法", "DIV", "将第一个数除以第二个数"),
            0x05: ("取模", "MOD", "计算除法的余数"),
            0x06: ("幂运算", "POW", "计算乘方"),
            0x10: ("逻辑与", "AND", "当两个条件都为真时返回真"),
            0x11: ("逻辑或", "OR", "当至少一个条件为真时返回真"),
            0x12: ("逻辑非", "NOT", "将真变假，假变真"),
            0x13: ("异或", "XOR", "当两个条件不同时返回真"),
            0x14: ("蕴含", "IMPLIES", "如果P则Q"),
            0x20: ("等于", "EQ", "判断两个值是否相等"),
            0x21: ("不等于", "NE", "判断两个值是否不相等"),
            0x22: ("小于", "LT", "判断第一个值是否小于第二个"),
            0x23: ("大于", "GT", "判断第一个值是否大于第二个"),
            0x30: ("跳转", "JUMP", "无条件跳转到指定位置"),
            0x31: ("条件跳转", "JUMP_IF", "如果条件为真则跳转"),
            0x32: ("调用", "CALL", "调用子程序"),
            0x33: ("返回", "RETURN", "从子程序返回"),
            0x34: ("循环", "LOOP", "重复执行指定次数"),
            0x40: ("加载", "LOAD", "从内存加载数据"),
            0x41: ("存储", "STORE", "将数据存储到内存"),
            0x60: ("打印", "PRINT", "输出信息到屏幕"),
            0x61: ("输入", "INPUT", "从用户获取输入"),
        }
    
    def translate_opcode(self, opcode: int) -> Dict[str, str]:
        """翻译单个操作码."""
        if opcode in self.opcode_translations:
            cn_name, en_name, description = self.opcode_translations[opcode]
            return {
                "chinese": cn_name,
                "english": en_name,
                "description": description,
                "hex": f"0x{opcode:02X}"
            }
        return {
            "chinese": f"未知操作({opcode})",
            "english": f"UNKNOWN_{opcode}",
            "description": "未定义的操作码",
            "hex": f"0x{opcode:02X}"
        }
    
    def translate_instruction(self, opcode: int, operands: List[Any], 
                             step_number: int = 0) -> str:
        """翻译单条指令为人类可读文本."""
        op_info = self.translate_opcode(opcode)
        
        if self.context.level == TranslationLevel.MINIMAL:
            return f"{op_info['chinese']}"
        
        elif self.context.level == TranslationLevel.STANDARD:
            return f"步骤{step_number}: {op_info['chinese']} ({op_info['english']}) - {op_info['description']}"
        
        elif self.context.level == TranslationLevel.DETAILED:
            lines = [
                f"┌─ 步骤 {step_number} ─────────────────────────────────",
                f"│ 操作: {op_info['chinese']} ({op_info['english']})",
                f"│ 代码: {op_info['hex']}",
                f"│ 说明: {op_info['description']}",
                f"│ 参数: {operands}",
                f"└───────────────────────────────────────────────────"
            ]
            return "\n".join(lines)
        
        else:  # DEBUG
            lines = [
                f"=== INSTRUCTION {step_number} ===",
                f"Opcode: {op_info['hex']} -> {op_info['english']}",
                f"Chinese: {op_info['chinese']}",
                f"Description: {op_info['description']}",
                f"Raw Operands: {operands}",
                f"Operand Types: {[type(o).__name__ for o in operands]}",
                "=" * 40
            ]
            return "\n".join(lines)
    
    def translate_program(self, instructions: List[Tuple[int, List[Any]]]) -> str:
        """翻译完整程序."""
        lines = []
        
        # 头部
        if self.context.level != TranslationLevel.MINIMAL:
            lines.extend([
                "╔════════════════════════════════════════════════════════════════════╗",
                "║                H2Q 程序翻译文档                                     ║",
                "║                (供人类监督使用)                                     ║",
                "╠════════════════════════════════════════════════════════════════════╣",
                f"║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<52} ║",
                f"║ 指令数量: {len(instructions):<52} ║",
                f"║ 翻译级别: {self.context.level.value:<52} ║",
                "╚════════════════════════════════════════════════════════════════════╝",
                ""
            ])
        
        # 指令列表
        lines.append("【程序指令序列】")
        lines.append("")
        
        for i, (opcode, operands) in enumerate(instructions):
            lines.append(self.translate_instruction(opcode, operands, i))
            lines.append("")
        
        # 摘要
        if self.context.level in [TranslationLevel.DETAILED, TranslationLevel.DEBUG]:
            op_counts = {}
            for opcode, _ in instructions:
                op_info = self.translate_opcode(opcode)
                name = op_info['chinese']
                op_counts[name] = op_counts.get(name, 0) + 1
            
            lines.extend([
                "【操作统计】",
                "-" * 40
            ])
            for name, count in sorted(op_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {count} 次")
        
        return "\n".join(lines)
    
    def translate_execution_result(self, result: Dict[str, Any]) -> str:
        """翻译执行结果."""
        lines = [
            "",
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                    执行结果翻译                                     ║",
            "╚════════════════════════════════════════════════════════════════════╝",
            ""
        ]
        
        # 状态
        success = result.get("success", False)
        status_icon = "✅" if success else "❌"
        status_text = "成功" if success else "失败"
        
        lines.extend([
            f"【执行状态】{status_icon} {status_text}",
            ""
        ])
        
        # 输出
        if result.get("stdout"):
            lines.extend([
                "【程序输出】",
                "-" * 40,
                result["stdout"],
                "-" * 40,
                ""
            ])
        
        # 错误
        if result.get("stderr") and not success:
            lines.extend([
                "【错误信息】",
                "-" * 40,
                result["stderr"],
                "-" * 40,
                "",
                "【错误解释】",
                self._explain_error(result["stderr"]),
                ""
            ])
        
        # 性能
        if "execution_time" in result:
            lines.extend([
                "【性能信息】",
                f"  执行时间: {result['execution_time']:.4f} 秒",
                f"  退出代码: {result.get('exit_code', 'N/A')}",
                ""
            ])
        
        return "\n".join(lines)
    
    def _explain_error(self, error_msg: str) -> str:
        """解释常见错误."""
        explanations = {
            "SyntaxError": "代码语法错误 - 可能是括号不匹配或关键字使用错误",
            "NameError": "名称未定义 - 使用了未声明的变量",
            "TypeError": "类型错误 - 对不兼容的类型进行了操作",
            "ZeroDivisionError": "除零错误 - 尝试除以零",
            "IndexError": "索引越界 - 访问了不存在的列表位置",
            "KeyError": "键不存在 - 访问了字典中不存在的键",
            "Docker": "Docker连接失败 - Docker服务可能未启动",
            "timeout": "执行超时 - 程序运行时间过长",
        }
        
        for key, explanation in explanations.items():
            if key.lower() in error_msg.lower():
                return explanation
        
        return "未能识别的错误，请检查详细错误信息"


# ============================================================================
# 第三部分: 决策解释器
# ============================================================================

class DecisionExplainer:
    """
    决策解释器 - 解释系统为什么做出特定决策.
    """
    
    def explain_compilation(self, task: str, opcodes: List[int], 
                           confidence_scores: List[float] = None) -> str:
        """解释编译决策."""
        translator = H2QHumanTranslator()
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                    编译决策解释                                     ║",
            "╚════════════════════════════════════════════════════════════════════╝",
            "",
            f"【输入任务】",
            f"  \"{task}\"",
            "",
            "【系统理解】",
            f"  任务被解析为 {len(opcodes)} 个操作步骤",
            "",
            "【生成的指令及理由】"
        ]
        
        for i, opcode in enumerate(opcodes):
            op_info = translator.translate_opcode(opcode)
            confidence = confidence_scores[i] if confidence_scores else 0.5
            
            lines.extend([
                f"",
                f"  指令 {i}: {op_info['chinese']} ({op_info['english']})",
                f"    ├─ 选择理由: 根据任务语义选择最匹配的操作",
                f"    └─ 置信度: {confidence * 100:.1f}%"
            ])
        
        return "\n".join(lines)
    
    def explain_learning(self, loss_history: List[float]) -> str:
        """解释学习过程."""
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                    学习过程解释                                     ║",
            "╚════════════════════════════════════════════════════════════════════╝",
            "",
            "【学习曲线】"
        ]
        
        if loss_history:
            max_loss = max(loss_history)
            for i, loss in enumerate(loss_history):
                bar_len = int((loss / max_loss) * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lines.append(f"  步骤{i+1}: {bar} {loss:.4f}")
            
            lines.extend([
                "",
                "【学习分析】",
                f"  初始损失: {loss_history[0]:.4f}",
                f"  最终损失: {loss_history[-1]:.4f}",
                f"  改进幅度: {(1 - loss_history[-1]/loss_history[0]) * 100:.1f}%",
                "",
                "【解释】",
                "  损失值越低，表示模型的预测越接近目标。",
                "  损失下降说明模型正在学习任务的模式。"
            ])
        else:
            lines.append("  (无学习历史)")
        
        return "\n".join(lines)


# ============================================================================
# 第四部分: 监督接口
# ============================================================================

class SupervisionInterface:
    """
    监督接口 - 提供给人类审查系统行为的工具.
    """
    
    def __init__(self):
        self.translator = H2QHumanTranslator()
        self.explainer = DecisionExplainer()
        self.audit_log: List[Dict] = []
    
    def review_task(self, task: str, program_data: Dict, 
                    execution_result: Dict) -> str:
        """生成完整的任务审查报告."""
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                 H2Q 任务执行审查报告                                ║",
            "║               (供人类监督者使用)                                    ║",
            "╚════════════════════════════════════════════════════════════════════╝",
            "",
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "═" * 70,
            "第一部分: 任务描述",
            "═" * 70,
            "",
            f"原始任务: {task}",
            "",
        ]
        
        # 编译结果
        lines.extend([
            "═" * 70,
            "第二部分: 系统编译结果",
            "═" * 70,
            ""
        ])
        
        if "instructions" in program_data:
            instructions = program_data["instructions"]
            for i, instr_str in enumerate(instructions):
                lines.append(f"  {i}: {instr_str}")
        
        # 执行结果
        lines.extend([
            "",
            "═" * 70,
            "第三部分: 执行结果",
            "═" * 70,
            ""
        ])
        
        if execution_result.get("success"):
            lines.append("✅ 执行成功")
            if execution_result.get("stdout"):
                lines.extend([
                    "",
                    "输出:",
                    "-" * 40,
                    execution_result["stdout"],
                    "-" * 40
                ])
        else:
            lines.append("❌ 执行失败")
            if execution_result.get("stderr"):
                lines.extend([
                    "",
                    "错误:",
                    execution_result["stderr"]
                ])
        
        # 审查建议
        lines.extend([
            "",
            "═" * 70,
            "第四部分: 监督建议",
            "═" * 70,
            "",
            "请人类监督者检查以下内容:",
            "  1. 系统是否正确理解了任务意图",
            "  2. 生成的指令序列是否合理",
            "  3. 执行结果是否符合预期",
            "  4. 是否存在安全或伦理问题",
            "",
            "如发现问题，请记录并反馈给开发团队。",
            "",
            "═" * 70,
            "报告结束",
            "═" * 70
        ])
        
        report = "\n".join(lines)
        
        # 记录审计日志
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "success": execution_result.get("success", False),
            "report_hash": hash(report)
        })
        
        return report
    
    def generate_summary_report(self) -> str:
        """生成审计摘要报告."""
        if not self.audit_log:
            return "暂无审计记录"
        
        total = len(self.audit_log)
        successful = sum(1 for log in self.audit_log if log["success"])
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                    审计摘要报告                                     ║",
            "╚════════════════════════════════════════════════════════════════════╝",
            "",
            f"总任务数: {total}",
            f"成功执行: {successful}",
            f"执行失败: {total - successful}",
            f"成功率: {successful/total*100:.1f}%",
            "",
            "最近任务:",
        ]
        
        for log in self.audit_log[-5:]:
            status = "✅" if log["success"] else "❌"
            lines.append(f"  {status} {log['timestamp']}: {log['task'][:30]}...")
        
        return "\n".join(lines)


# ============================================================================
# 第五部分: 演示
# ============================================================================

def demonstrate_translation_system():
    """演示翻译系统."""
    print("=" * 70)
    print("H2Q 人类可读翻译系统演示")
    print("=" * 70)
    
    # 创建翻译器
    translator = H2QHumanTranslator(TranslationContext(level=TranslationLevel.DETAILED))
    explainer = DecisionExplainer()
    supervisor = SupervisionInterface()
    
    # 示例程序
    sample_program = [
        (0x01, [10, 20]),      # ADD
        (0x41, [100]),         # STORE
        (0x60, ["结果"]),      # PRINT
        (0x20, [30, 30]),      # EQ
        (0x31, [5]),           # JUMP_IF
    ]
    
    print("\n【1. 程序翻译】")
    print(translator.translate_program(sample_program))
    
    # 执行结果翻译
    sample_result = {
        "success": True,
        "stdout": "结果: 30\n比较结果: True",
        "stderr": "",
        "execution_time": 0.0015,
        "exit_code": 0
    }
    
    print("\n【2. 执行结果翻译】")
    print(translator.translate_execution_result(sample_result))
    
    # 学习过程解释
    sample_loss_history = [5.5, 5.2, 4.8, 4.3, 3.9, 3.5, 3.2]
    
    print("\n【3. 学习过程解释】")
    print(explainer.explain_learning(sample_loss_history))
    
    # 完整审查报告
    print("\n【4. 监督审查报告】")
    report = supervisor.review_task(
        task="计算 10 + 20 并打印结果",
        program_data={"instructions": ["ADD 10, 20", "STORE R0", "PRINT R0"]},
        execution_result=sample_result
    )
    print(report)
    
    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)
    print("""
【系统设计总结】

这个翻译系统实现了以下目标:

1. H2Q内部编码 → 人类可读文档
   - 支持多级详细程度 (简洁/标准/详细/调试)
   - 操作码有中英文双语翻译
   - 包含操作说明

2. 执行过程可视化
   - 清晰显示执行状态
   - 格式化输出结果
   - 错误解释和分析

3. 决策解释
   - 解释编译选择的理由
   - 可视化学习曲线
   - 提供置信度信息

4. 监督接口
   - 生成完整审查报告
   - 维护审计日志
   - 提供监督建议

【核心原则】
系统的核心是编码能力，翻译层只是观察窗口，
不改变系统的工作方式，只提供人类可理解的视角。
    """)


def main():
    demonstrate_translation_system()


if __name__ == "__main__":
    main()
