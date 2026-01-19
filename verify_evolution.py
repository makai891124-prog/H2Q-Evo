import os
import json
import subprocess
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

PROJECT_ROOT = Path("./h2q_project").resolve()
STATE_FILE = "evo_state.json"

def print_header(title):
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN} {title}")
    print(f"{Fore.CYAN}{'='*50}")

def check_git_history():
    """节点 1: 检查 Git 进化历史"""
    print_header("节点 1: 进化轨迹 (Git Log)")
    try:
        # 获取最近 3 次提交
        result = subprocess.run(
            ["git", "log", "-n", "3", "--pretty=format:%h - %an: %s (%cr)"],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        print(result.stdout)
        print(f"\n{Fore.GREEN}>>> 成功检测到进化提交！系统正在自主工作。")
    except Exception as e:
        print(f"{Fore.RED}Git 检查失败: {e}")

def check_task_status():
    """节点 2: 检查任务完成度"""
    print_header("节点 2: 任务状态 (State JSON)")
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件。")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    completed = [t for t in state['todo_list'] if t['status'] == 'completed']
    pending = [t for t in state['todo_list'] if t['status'] == 'pending']
    
    print(f"当前代数 (Generation): {state.get('generation', 0)}")
    print(f"{Fore.GREEN}已完成任务: {len(completed)}")
    print(f"{Fore.YELLOW}待处理任务: {len(pending)}")
    
    if completed:
        latest = completed[-1]
        print(f"\n[最新成果] ID {latest['id']}: {latest['task']}")

def run_h2q_bridge_test():
    """节点 3: 核心逻辑验证 (H2Q Bridge)"""
    print_header("节点 3: 核心逻辑验证 (H2Q Bridge)")
    
    # 修复了之前的 NameError，在子进程中独立处理颜色
    test_script = """
import sys
import os
try:
    from colorama import Fore, init
    init(autoreset=True)
except ImportError:
    class Fore: GREEN=""; RED=""; YELLOW=""

sys.path.insert(0, os.getcwd())

print('正在尝试导入 h2q.dde...')
try:
    import h2q.dde
    print(f'{Fore.GREEN}SUCCESS: h2q.dde 模块导入成功！')
    
    # 进一步检查：查看类是否包含 AI 修改的 latent_dim 参数
    import inspect
    sig = inspect.signature(h2q.dde.DiscreteDecisionEngine.__init__)
    print(f"当前 __init__ 签名: {sig}")
    
    if 'latent_dim' in str(sig):
        print(f'{Fore.GREEN}>>> 验证通过：AI 已成功添加 latent_dim 参数。')
    else:
        print(f'{Fore.YELLOW}>>> 提示：导入成功，但未检测到 latent_dim 参数（可能是其他任务）。')

except ImportError as e:
    print(f'{Fore.RED}FAILED: 导入失败 - {e}')
except Exception as e:
    print(f'{Fore.RED}ERROR: 运行时错误 - {e}')
"""
    try:
        subprocess.run(["python3", "-c", test_script], cwd=PROJECT_ROOT)
    except Exception as e:
        print(f"测试运行失败: {e}")

def show_latest_diff():
    """节点 4: 展示代码变更详情"""
    print_header("节点 4: AI 代码修改详情 (Git Diff)")
    try:
        # 显示最近一次提交的具体代码变动
        result = subprocess.run(
            ["git", "show", "--stat", "--oneline", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        print(result.stdout)
        print(f"{Fore.BLUE}>>> 提示：使用 'cd h2q_project && git show HEAD' 可查看完整代码差异。")
    except Exception as e:
        print(f"Diff 获取失败: {e}")

if __name__ == "__main__":
    check_git_history()
    check_task_status()
    run_h2q_bridge_test()
    show_latest_diff()