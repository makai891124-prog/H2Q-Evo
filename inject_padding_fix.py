import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

# 定义更加具体的修复任务
PADDING_TASK = {
    "task": "URGENT FIX 'h2q_server.py': Runtime Error 'tensor a (256) must match b (32)'. The server crashes on short inputs. You MUST implement explicit padding logic: 1. Convert text to bytes. 2. If length < 256, pad with zeros to exactly 256. 3. If length > 256, truncate to 256. 4. Ensure input tensor shape is always [1, 256].",
    "priority": "critical",
    "source": "user"
}

def inject():
    if not os.path.exists(STATE_FILE): return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # 1. 获取 ID
    last_id = state.get("last_task_id", 0) + 1
    PADDING_TASK["id"] = last_id
    state["last_task_id"] = last_id
    
    # 2. 强制插队到第一位
    state["todo_list"].insert(0, PADDING_TASK)
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"{Fore.RED}>>> [手术刀] 已注入填充修复任务 (ID {last_id})。")
    print(f"{Fore.YELLOW}>>> AI 将立即修改 h2q_server.py 以支持任意长度输入。")

if __name__ == "__main__":
    inject()