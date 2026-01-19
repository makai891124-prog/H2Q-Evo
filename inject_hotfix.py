import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

# 定义修复任务
HOTFIX_TASK = {
    "task": "CRITICAL BUGFIX in 'h2q_server.py': Runtime Error 'size of tensor a (256) must match b (23)'. The server fails to handle variable-length user input. You MUST implement a padding mechanism (pad with zeros) to ensure the input tensor always matches the model's required dimension (256).",
    "priority": "critical",
    "source": "user"
}

def inject():
    if not os.path.exists(STATE_FILE): return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # 1. 获取 ID
    last_id = state.get("last_task_id", 0) + 1
    HOTFIX_TASK["id"] = last_id
    state["last_task_id"] = last_id
    
    # 2. 强制插队到第一位
    state["todo_list"].insert(0, HOTFIX_TASK)
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"{Fore.RED}>>> [紧急] 已注入修复任务 (ID {last_id})。")
    print(f"{Fore.YELLOW}>>> 请确保 evolution_system.py 正在运行，它将立即捕获并修复此 Bug。")

if __name__ == "__main__":
    inject()