import os
import re
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
DDE_FILE = PROJECT_ROOT / "h2q" / "dde.py"
SERVER_FILE = PROJECT_ROOT / "h2q_server.py"

def get_dde_init_args():
    """不依赖 import，直接分析源码文本获取参数名"""
    if not DDE_FILE.exists():
        print(f"❌ 错误：找不到 {DDE_FILE}")
        return None

    with open(DDE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则寻找 __init__ 定义
    # 匹配 def __init__(self, arg1, arg2=Val, ...)
    match = re.search(r'def\s+__init__\s*\((.*?)\)', content, re.DOTALL)
    if not match:
        print("❌ 无法解析 DDE 的 __init__ 函数")
        return None
    
    args_str = match.group(1)
    # 清理换行和空格
    args = [a.strip().split(':')[0].split('=')[0].strip() for a in args_str.split(',')]
    args = [a for a in args if a != 'self']
    
    print(f"🔍 检测到 DDE 参数列表: {args}")
    return args

def patch_server(args):
    if not SERVER_FILE.exists():
        print(f"❌ 错误：找不到 {SERVER_FILE}")
        return

    with open(SERVER_FILE, 'r', encoding='utf-8') as f:
        server_code = f.read()

    # 构造正确的实例化代码
    new_init_line = ""
    
    # 策略：根据检测到的参数名构造调用
    params = []
    
    # 1. 处理维度参数
    if 'latent_dim' in args:
        params.append("latent_dim=32")
    elif 'dim' in args:
        params.append("dim=32") # 假设 latent 对应 dim
    elif 'context_dim' in args:
        params.append("context_dim=32")
    elif 'input_dim' in args:
        params.append("input_dim=32")
        
    # 2. 处理动作参数
    if 'num_actions' in args:
        params.append("num_actions=10")
    elif 'action_dim' in args:
        params.append("action_dim=10")
        
    # 3. 处理其他必需参数 (如果有 vocab_size)
    if 'vocab_size' in args:
        params.append("vocab_size=257")

    new_init_line = f"    dde = DiscreteDecisionEngine({', '.join(params)})"
    print(f"✅ 生成新的初始化代码: \n{new_init_line.strip()}")

    # 替换旧的初始化行
    # 匹配 dde = DiscreteDecisionEngine(...)
    new_server_code = re.sub(
        r'dde\s*=\s*DiscreteDecisionEngine\(.*?\)', 
        new_init_line.strip(), 
        server_code
    )

    with open(SERVER_FILE, 'w', encoding='utf-8') as f:
        f.write(new_server_code)
    
    print("🚀 服务器代码已更新！接口已对齐。")

if __name__ == "__main__":
    print(">>> 开始自动对齐接口...")
    args = get_dde_init_args()
    if args:
        patch_server(args)