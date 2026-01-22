#!/usr/bin/env python3
"""
H2Q 自主脚本生成与执行系统

核心理念:
=========
系统的核心能力是【编码】，不是模拟人类的现有工具。
但需要提供【翻译能力结构】使人类能够看懂进行监督。

这个系统:
1. 使用H2Q核心机的流式编码作为内部表示
2. 能够自动生成可执行脚本实现所需功能
3. 在Docker隔离环境中安全执行
4. 提供人类可理解的翻译层

关键区别:
- 传统方式: 人类写代码 → 机器执行
- H2Q方式: 系统生成内部编码 → 翻译为人类可读脚本 → 执行

系统不需要"还原成人类的现有工具"，但提供翻译使人类能监督。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import tempfile
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import struct
import re

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


# ============================================================================
# 第一部分: H2Q 内部编码 (H2Q Native Encoding)
# ============================================================================

class H2QOpCode(Enum):
    """H2Q操作码 - 系统的原生指令集."""
    # 基础计算
    NOP = 0x00
    ADD = 0x01
    SUB = 0x02
    MUL = 0x03
    DIV = 0x04
    MOD = 0x05
    POW = 0x06
    
    # 逻辑运算
    AND = 0x10
    OR = 0x11
    NOT = 0x12
    XOR = 0x13
    IMPLIES = 0x14
    
    # 比较
    EQ = 0x20
    NE = 0x21
    LT = 0x22
    GT = 0x23
    LE = 0x24
    GE = 0x25
    
    # 控制流
    JUMP = 0x30
    JUMP_IF = 0x31
    CALL = 0x32
    RETURN = 0x33
    LOOP = 0x34
    
    # 数据操作
    LOAD = 0x40
    STORE = 0x41
    PUSH = 0x42
    POP = 0x43
    
    # 字符串
    CONCAT = 0x50
    SPLIT = 0x51
    FORMAT = 0x52
    
    # I/O
    PRINT = 0x60
    INPUT = 0x61
    READ_FILE = 0x62
    WRITE_FILE = 0x63


@dataclass
class H2QInstruction:
    """H2Q指令 - 系统的原子操作单元."""
    opcode: H2QOpCode
    operands: List[Any]
    annotation: str = ""  # 人类可读注释
    
    def to_bytes(self) -> bytes:
        """序列化为字节."""
        result = struct.pack('B', self.opcode.value)
        operand_bytes = json.dumps(self.operands).encode('utf-8')
        result += struct.pack('H', len(operand_bytes))
        result += operand_bytes
        return result
    
    def to_human_readable(self) -> str:
        """翻译为人类可读格式."""
        op_names = {
            H2QOpCode.ADD: "加法",
            H2QOpCode.SUB: "减法",
            H2QOpCode.MUL: "乘法",
            H2QOpCode.DIV: "除法",
            H2QOpCode.AND: "逻辑与",
            H2QOpCode.OR: "逻辑或",
            H2QOpCode.NOT: "逻辑非",
            H2QOpCode.EQ: "等于",
            H2QOpCode.PRINT: "打印输出",
            H2QOpCode.LOOP: "循环",
        }
        name = op_names.get(self.opcode, self.opcode.name)
        return f"[{name}] 操作数: {self.operands}" + (f" // {self.annotation}" if self.annotation else "")


@dataclass
class H2QProgram:
    """H2Q程序 - 指令序列."""
    instructions: List[H2QInstruction]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_human_readable(self) -> str:
        """完整程序的人类可读翻译."""
        lines = [
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║           H2Q 程序 → 人类可读翻译                                     ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
        ]
        
        for i, instr in enumerate(self.instructions):
            lines.append(f"║ {i:3d}: {instr.to_human_readable():<64} ║")
        
        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ============================================================================
# 第二部分: 神经编码器 (Neural Encoder)
# ============================================================================

class H2QNeuralCompiler(nn.Module):
    """
    H2Q神经编译器 - 将自然语言任务编译为H2Q指令序列.
    
    这是一个真正的学习系统：
    - 输入: 自然语言任务描述
    - 输出: H2Q指令序列
    - 学习: 通过训练数据学习映射
    """
    
    def __init__(self, vocab_size: int = 257, hidden_dim: int = 256, max_instructions: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_instructions = max_instructions
        self.num_opcodes = len(H2QOpCode)
        
        # 编码器
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )
        
        # 指令生成器
        self.instruction_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, max_instructions * (self.num_opcodes + 8))  # opcode + 8个操作数槽
        )
        
        # 训练历史
        self.training_step = 0
    
    def forward(self, input_bytes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编译任务描述为指令.
        
        Args:
            input_bytes: [batch, seq_len] 输入字节序列
            
        Returns:
            opcode_logits: [batch, max_instructions, num_opcodes]
            operand_logits: [batch, max_instructions, 8]
        """
        # 编码
        embeds = self.embedding(input_bytes)
        encoded = self.encoder(embeds)
        
        # 池化
        seq_repr = encoded.mean(dim=1)  # [batch, hidden_dim]
        
        # 生成指令
        raw_output = self.instruction_generator(seq_repr)  # [batch, max_instr * (opcodes + 8)]
        
        # 重塑
        batch_size = input_bytes.shape[0]
        raw_output = raw_output.view(batch_size, self.max_instructions, -1)
        
        opcode_logits = raw_output[:, :, :self.num_opcodes]
        operand_logits = raw_output[:, :, self.num_opcodes:]
        
        return opcode_logits, operand_logits
    
    def compile(self, task_text: str) -> H2QProgram:
        """
        将任务文本编译为H2Q程序.
        """
        self.eval()
        
        # 文本转字节
        bytes_list = list(task_text.encode('utf-8'))[:256]
        while len(bytes_list) < 64:
            bytes_list.append(0)
        input_tensor = torch.tensor(bytes_list[:64], dtype=torch.long).unsqueeze(0)
        
        if next(self.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()
        elif str(next(self.parameters()).device).startswith('mps'):
            input_tensor = input_tensor.to('mps')
        
        with torch.no_grad():
            opcode_logits, operand_logits = self(input_tensor)
            
            # 解码指令
            opcodes = torch.argmax(opcode_logits, dim=-1)[0]  # [max_instructions]
            operands = operand_logits[0]  # [max_instructions, 8]
        
        # 构建程序
        instructions = []
        for i in range(self.max_instructions):
            opcode_idx = opcodes[i].item()
            if opcode_idx >= len(H2QOpCode):
                opcode_idx = 0
            
            opcode = list(H2QOpCode)[opcode_idx]
            if opcode == H2QOpCode.NOP and i > 0:
                continue  # 跳过NOP
            
            ops = operands[i].cpu().numpy().tolist()
            # 将浮点操作数转换为合理的整数
            ops = [int(x * 10) for x in ops[:4]]
            
            instructions.append(H2QInstruction(
                opcode=opcode,
                operands=ops,
                annotation=f"从任务'{task_text[:20]}...'生成"
            ))
        
        return H2QProgram(
            instructions=instructions[:10],  # 限制指令数
            metadata={
                "source": task_text,
                "compiled_at": datetime.now().isoformat()
            }
        )


# ============================================================================
# 第三部分: 脚本生成器 (Script Generator)
# ============================================================================

class PythonScriptGenerator:
    """
    Python脚本生成器 - 将H2Q程序翻译为Python代码.
    
    这是【翻译层】，使人类能够理解系统的操作。
    系统的核心是H2Q编码，Python只是人类可读的翻译。
    """
    
    def __init__(self):
        self.indent_level = 0
        
        # 操作映射 (H2Q → Python)
        self.op_generators = {
            H2QOpCode.ADD: self._gen_add,
            H2QOpCode.SUB: self._gen_sub,
            H2QOpCode.MUL: self._gen_mul,
            H2QOpCode.DIV: self._gen_div,
            H2QOpCode.MOD: self._gen_mod,
            H2QOpCode.POW: self._gen_pow,
            H2QOpCode.AND: self._gen_and,
            H2QOpCode.OR: self._gen_or,
            H2QOpCode.NOT: self._gen_not,
            H2QOpCode.EQ: self._gen_eq,
            H2QOpCode.LT: self._gen_lt,
            H2QOpCode.GT: self._gen_gt,
            H2QOpCode.PRINT: self._gen_print,
            H2QOpCode.LOOP: self._gen_loop,
            H2QOpCode.CONCAT: self._gen_concat,
        }
    
    def _indent(self) -> str:
        return "    " * self.indent_level
    
    def _gen_add(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} + {ops[1]}"
    
    def _gen_sub(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} - {ops[1]}"
    
    def _gen_mul(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} * {ops[1]}"
    
    def _gen_div(self, ops: List[Any]) -> str:
        divisor = ops[1] if ops[1] != 0 else 1
        return f"result = {ops[0]} / {divisor}"
    
    def _gen_mod(self, ops: List[Any]) -> str:
        divisor = ops[1] if ops[1] != 0 else 1
        return f"result = {ops[0]} % {divisor}"
    
    def _gen_pow(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} ** {min(ops[1], 10)}"  # 限制指数
    
    def _gen_and(self, ops: List[Any]) -> str:
        return f"result = {bool(ops[0])} and {bool(ops[1])}"
    
    def _gen_or(self, ops: List[Any]) -> str:
        return f"result = {bool(ops[0])} or {bool(ops[1])}"
    
    def _gen_not(self, ops: List[Any]) -> str:
        return f"result = not {bool(ops[0])}"
    
    def _gen_eq(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} == {ops[1]}"
    
    def _gen_lt(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} < {ops[1]}"
    
    def _gen_gt(self, ops: List[Any]) -> str:
        return f"result = {ops[0]} > {ops[1]}"
    
    def _gen_print(self, ops: List[Any]) -> str:
        return f"print({repr(str(ops[0]))})"
    
    def _gen_loop(self, ops: List[Any]) -> str:
        count = min(max(ops[0], 1), 10)  # 限制循环次数
        return f"for i in range({count}):\n{self._indent()}    pass  # 循环体"
    
    def _gen_concat(self, ops: List[Any]) -> str:
        return f"result = str({ops[0]}) + str({ops[1]})"
    
    def generate(self, program: H2QProgram) -> str:
        """生成完整的Python脚本."""
        lines = [
            "#!/usr/bin/env python3",
            '"""',
            "H2Q 自动生成脚本",
            f"生成时间: {datetime.now().isoformat()}",
            "",
            "【重要说明】",
            "这是H2Q内部编码的Python翻译，供人类监督使用。",
            "系统的核心表示是H2Q编码，不是这个Python脚本。",
            '"""',
            "",
            "# 初始化",
            "result = None",
            "variables = {}",
            "",
            "# === 主程序 (从H2Q指令翻译) ===",
            "",
        ]
        
        for i, instr in enumerate(program.instructions):
            lines.append(f"# 指令 {i}: {instr.opcode.name}")
            
            generator = self.op_generators.get(instr.opcode)
            if generator:
                code = generator(instr.operands)
                lines.append(code)
            else:
                lines.append(f"# (未实现的操作: {instr.opcode.name})")
            
            lines.append(f"variables['step_{i}'] = result if 'result' in dir() else None")
            lines.append("")
        
        # 添加输出
        lines.extend([
            "# === 输出结果 ===",
            "print('=' * 50)",
            "print('H2Q 执行结果')",
            "print('=' * 50)",
            "for key, value in variables.items():",
            "    print(f'{key}: {value}')",
            "print('执行完成')",
        ])
        
        return "\n".join(lines)
    
    def generate_with_translation(self, program: H2QProgram) -> Tuple[str, str]:
        """生成脚本和人类可读翻译."""
        script = self.generate(program)
        translation = program.to_human_readable()
        return script, translation


# ============================================================================
# 第四部分: Docker 执行器 (Docker Executor)
# ============================================================================

class H2QDockerExecutor:
    """
    Docker执行器 - 在隔离环境中执行H2Q生成的脚本.
    
    安全特性:
    - 网络隔离 (--network none)
    - 内存限制 (--memory)
    - CPU限制 (--cpus)
    - 超时保护
    - 只读挂载
    """
    
    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        self.execution_log: List[Dict] = []
    
    def execute(self, script: str, timeout: int = 30) -> Dict[str, Any]:
        """执行脚本并返回结果."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "execution_time": 0.0,
            "security_violations": []
        }
        
        # 安全检查
        violations = self._security_check(script)
        if violations:
            result["security_violations"] = violations
            result["stderr"] = f"安全检查失败: {violations}"
            return result
        
        # 创建临时脚本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            start_time = datetime.now()
            
            # Docker命令
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{script_path}:/app/script.py:ro",
                "--network", "none",
                "--memory", "64m",
                "--cpus", "0.5",
                "--user", "nobody",
                self.image,
                "python", "/app/script.py"
            ]
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["exit_code"] = proc.returncode
            result["success"] = proc.returncode == 0
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
            
        except subprocess.TimeoutExpired:
            result["stderr"] = f"执行超时 ({timeout}秒)"
        except FileNotFoundError:
            # Docker不可用，使用本地执行（受限）
            result = self._execute_local(script, timeout)
        except Exception as e:
            result["stderr"] = str(e)
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
        
        # 记录日志
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "script_hash": hashlib.md5(script.encode()).hexdigest()[:8],
            "result_summary": {
                "success": result["success"],
                "exit_code": result["exit_code"],
                "execution_time": result["execution_time"]
            }
        })
        
        return result
    
    def _security_check(self, script: str) -> List[str]:
        """脚本安全检查."""
        violations = []
        
        # 危险模式
        dangerous_patterns = [
            (r'\bimport\s+os\b', "禁止导入os模块"),
            (r'\bimport\s+subprocess\b', "禁止导入subprocess"),
            (r'\bimport\s+socket\b', "禁止导入socket"),
            (r'\bopen\s*\([^)]*[\'"]w', "禁止写文件操作"),
            (r'\beval\s*\(', "禁止eval"),
            (r'\bexec\s*\(', "禁止exec"),
            (r'__import__', "禁止__import__"),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, script):
                violations.append(message)
        
        return violations
    
    def _execute_local(self, script: str, timeout: int) -> Dict[str, Any]:
        """本地执行（当Docker不可用时）."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "execution_time": 0.0,
            "note": "使用本地执行（Docker不可用）"
        }
        
        # 安全检查
        violations = self._security_check(script)
        if violations:
            result["stderr"] = f"安全检查失败: {violations}"
            return result
        
        # 创建受限执行环境
        safe_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # 捕获输出
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            start_time = datetime.now()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(script, safe_globals)
            
            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()
            result["exit_code"] = 0
            result["success"] = True
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result["stderr"] = str(e)
            result["exit_code"] = 1
        
        return result


# ============================================================================
# 第五部分: 完整系统 (Complete System)
# ============================================================================

class H2QAutonomousScriptSystem:
    """
    H2Q 自主脚本系统 - 完整的编码→生成→执行→翻译流程.
    """
    
    def __init__(self, device: str = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        
        # 核心组件
        self.compiler = H2QNeuralCompiler().to(self.device)
        self.generator = PythonScriptGenerator()
        self.executor = H2QDockerExecutor()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.compiler.parameters(), lr=1e-4)
        
        # 历史记录
        self.execution_history: List[Dict] = []
    
    def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        完整处理任务:
        1. 编译任务为H2Q程序
        2. 生成Python脚本
        3. 执行脚本
        4. 提供人类可读翻译
        """
        result = {
            "task": task_description,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # 阶段1: 编译
        print(f"\n📝 任务: {task_description}")
        print("-" * 60)
        
        program = self.compiler.compile(task_description)
        result["stages"]["compilation"] = {
            "instructions_count": len(program.instructions),
            "instructions": [instr.to_human_readable() for instr in program.instructions]
        }
        print(f"✓ 编译完成: {len(program.instructions)} 条指令")
        
        # 阶段2: 生成脚本
        script, translation = self.generator.generate_with_translation(program)
        result["stages"]["generation"] = {
            "script_length": len(script),
            "script": script
        }
        print(f"✓ 脚本生成: {len(script)} 字符")
        
        # 阶段3: 执行
        exec_result = self.executor.execute(script)
        result["stages"]["execution"] = exec_result
        
        if exec_result["success"]:
            print(f"✓ 执行成功 ({exec_result['execution_time']:.2f}s)")
        else:
            print(f"✗ 执行失败: {exec_result['stderr'][:100]}")
        
        # 阶段4: 翻译
        result["stages"]["translation"] = {
            "human_readable": translation
        }
        print("✓ 翻译生成")
        
        # 记录
        self.execution_history.append(result)
        
        return result
    
    def train_on_task(self, task: str, expected_opcodes: List[int]) -> float:
        """训练编译器学习任务到指令的映射."""
        self.compiler.train()
        
        # 准备输入
        bytes_list = list(task.encode('utf-8'))[:256]
        while len(bytes_list) < 64:
            bytes_list.append(0)
        input_tensor = torch.tensor(bytes_list[:64], dtype=torch.long).unsqueeze(0).to(self.device)
        
        # 准备目标
        target_opcodes = torch.tensor(expected_opcodes[:self.compiler.max_instructions], dtype=torch.long)
        while len(target_opcodes) < self.compiler.max_instructions:
            target_opcodes = torch.cat([target_opcodes, torch.tensor([0])])
        target_opcodes = target_opcodes.unsqueeze(0).to(self.device)
        
        # 前向传播
        opcode_logits, _ = self.compiler(input_tensor)
        
        # 损失
        loss = F.cross_entropy(
            opcode_logits.view(-1, self.compiler.num_opcodes),
            target_opcodes.view(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def demonstrate(self):
        """完整演示."""
        print("=" * 70)
        print("H2Q 自主脚本生成系统 - 演示")
        print("=" * 70)
        
        print("""
【系统说明】
这是一个诚实的系统，展示真实的能力和限制：

能力：
  ✓ 使用神经网络将任务编译为H2Q内部编码
  ✓ 从H2Q编码生成可执行Python脚本
  ✓ 在隔离环境中安全执行脚本
  ✓ 提供人类可读的翻译供监督

限制（诚实承认）：
  × 当前模型未经大规模训练，能力有限
  × 生成的脚本较为简单
  × 需要更多数据才能学习复杂任务
        """)
        
        # 训练阶段
        print("\n" + "=" * 70)
        print("阶段1: 训练学习")
        print("=" * 70)
        
        training_data = [
            ("计算两数之和", [H2QOpCode.ADD.value, H2QOpCode.PRINT.value]),
            ("计算两数之差", [H2QOpCode.SUB.value, H2QOpCode.PRINT.value]),
            ("计算两数之积", [H2QOpCode.MUL.value, H2QOpCode.PRINT.value]),
            ("判断两数是否相等", [H2QOpCode.EQ.value, H2QOpCode.PRINT.value]),
            ("打印消息", [H2QOpCode.PRINT.value]),
        ]
        
        print("\n训练中...")
        for i, (task, expected) in enumerate(training_data):
            loss = self.train_on_task(task, expected)
            print(f"  任务{i+1}: '{task}' → 损失: {loss:.4f}")
        
        # 测试阶段
        print("\n" + "=" * 70)
        print("阶段2: 测试执行")
        print("=" * 70)
        
        test_tasks = [
            "计算 25 加 17",
            "判断 100 是否大于 50",
        ]
        
        for task in test_tasks:
            result = self.process_task(task)
            
            print("\n【人类可读翻译】")
            print(result["stages"]["translation"]["human_readable"])
            
            if result["stages"]["execution"]["success"]:
                print("\n【执行输出】")
                print(result["stages"]["execution"]["stdout"])
        
        # 总结
        print("\n" + "=" * 70)
        print("演示完成")
        print("=" * 70)


def main():
    """主函数."""
    system = H2QAutonomousScriptSystem()
    system.demonstrate()
    
    # 保存执行历史
    history_path = SCRIPT_DIR / "autonomous_execution_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(system.execution_history, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n执行历史已保存: {history_path}")


if __name__ == "__main__":
    main()
