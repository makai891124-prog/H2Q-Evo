import os
import ast
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()

# 我们要审计的核心数学文件
TARGET_FILES = [
    "h2q/group_ops.py",       # 汉密尔顿积
    "h2q/core/sst.py",        # 谱位移
    "h2q/knot_kernel.py",     # 纽结核
    "h2q/core/reversible_kernel.py" # 可逆流
]

def extract_forward_logic(file_path):
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        return f"❌ 文件不存在: {file_path}"

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        source_lines = f.read().splitlines() # 重新读取用于提取源码
    except:
        # 重新读取用于提取源码
        with open(full_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()

    result = []
    result.append(f"📂 文件: {file_path}")
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    # 提取 forward 函数的源码
                    start = item.lineno - 1
                    end = item.end_lineno
                    code_segment = "".join(source_lines[start:end])
                    result.append(f"  CLASS: {class_name}")
                    result.append(f"  MATH IMPLEMENTATION:\n{code_segment}")
                    result.append("-" * 40)
    
    return "\n".join(result)

if __name__ == "__main__":
    print("🔍 正在执行数学内核代码审计...\n")
    for target in TARGET_FILES:
        print(extract_forward_logic(target))
        print("\n" + "="*60 + "\n")