import sys
import os
from pathlib import Path
import importlib.util

# 设置路径
PROJECT_ROOT = Path("./h2q_project").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def trigger_writing():
    # 1. 加载 AI 写的工具
    tool_path = PROJECT_ROOT / "tools" / "code_writer.py"
    spec = importlib.util.spec_from_file_location("code_writer", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 2. 实例化
    writer = module.CodeWriter(project_root=str(PROJECT_ROOT))
    
    # 3. 定义要写的文件和内容
    target_file = "hello_human.py"
    content = """
# This file was autonomously written by H2Q-Evo.
# Generation: 47+
# Tool: CodeWriter

def greet():
    print("Hello! I am the H2Q AGI System.")
    print("I have successfully manifested this file into your physical storage.")
    print("My logic is grounded in the H2Q project structure.")

if __name__ == "__main__":
    greet()
"""
    
    # 4. 让 AI 写入
    print(f"正在请求 AI 写入文件: {target_file} ...")
    success = writer.write_module(target_file, content, {"spectral_shift": 42.0})
    
    if success:
        full_path = PROJECT_ROOT / target_file
        print(f"\n✅ 写入成功！")
        print(f"📂 实体文件位置: {full_path}")
        print(f"👉 您现在可以双击打开它，或者在终端运行: python3 h2q_project/{target_file}")

if __name__ == "__main__":
    trigger_writing()