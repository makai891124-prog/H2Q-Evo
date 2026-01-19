import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

# 架构升级任务包
MIDDLEWARE_TASKS = [
    {
        "task": "Architect 'h2q/core/adapter.py': Create a 'UniversalAdapter' class. It must use Python's 'inspect' module to dynamically read the signature and input shape requirements of 'DiscreteDecisionEngine' at runtime. It acts as a buffer, automatically reshaping/padding user input to match whatever the current Kernel demands (e.g., 32, 64, 256).",
        "priority": "critical",
        "source": "user"
    },
    {
        "task": "Implement 'tests/test_api_contract.py': Create a 'Contract Guard'. This test suite simulates external API calls. If the Kernel evolves and breaks this test, the evolution MUST be rejected. This forces the AI to update the Adapter BEFORE committing Kernel changes.",
        "priority": "high",
        "source": "user"
    },
    {
        "task": "Refactor 'h2q_server.py' to use Middleware: Rewrite the server to ONLY interact with 'UniversalAdapter'. The server should never import DDE directly. This ensures that no matter how the Kernel evolves (AMX tiling, Fractal expansion), the API endpoint remains stable.",
        "priority": "high",
        "source": "user"
    }
]

def inject():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件。")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    last_id = state.get("last_task_id", 0)
    todos = state.get("todo_list", [])
    
    print(f"{Fore.CYAN}>>> 正在注入【自适应中间件】架构任务...")
    print(f"{Fore.CYAN}>>> 目标：赋予 AGI 接口稳定性管理的自我意识。")
    
    # 倒序插入到最前面
    for task_def in reversed(MIDDLEWARE_TASKS):
        last_id += 1
        new_task = {
            "id": last_id,
            "task": task_def['task'],
            "priority": task_def['priority'],
            "status": "pending",
            "source": "user"
        }
        todos.insert(0, new_task)
        print(f"{Fore.GREEN} + [ID {last_id}] {new_task['task'][:80]}...")

    state["last_task_id"] = last_id
    state["todo_list"] = todos
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n{Fore.MAGENTA}>>> 注入完成！请重启 evolution_system.py。")
    print(f"{Fore.MAGENTA}>>> AI 将开始学习如何做一个负责任的架构师。")

if __name__ == "__main__":
    inject()