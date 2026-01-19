import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

SURVIVAL_TASKS = [
    {
        "task": "Architect 'h2q/core/memory_crystal.py': Create a 'MemoryManager' class. It must: 1. Auto-load the latest '.pt' from 'checkpoints/' on startup. 2. Implement 'save_crystal(state_dict)' to save weights with a timestamp and loss-metric signature. 3. Support 'hot-reloading' weights without restarting the server.",
        "priority": "critical",
        "source": "user"
    },
    {
        "task": "Implement 'Online Learning Middleware': Modify 'h2q_server.py' (via Adapter) to include a 'Learn-on-Interaction' mode. Every user input should be treated as a training sample. Perform a backward pass (SGD step) after generating a response to minimize the divergence between the user's prompt and the system's internal manifold.",
        "priority": "high",
        "source": "user"
    },
    {
        "task": "Create 'h2q/engine/curiosity.py' (The Proposer): Implement a background thread that runs when the server is idle. It should use the 'H2QSyntheticEngine' to generate 'adversarial questions' (high entropy inputs) and feed them into the DDE. If the DDE's confidence is low, trigger a training loop automatically.",
        "priority": "high",
        "source": "user"
    },
    {
        "task": "Establish 'Survival Homeostasis': Create a master loop in 'system.py' that balances 'Service' (answering users) and 'Growth' (training). If memory usage > 80% or temperature > 90C (simulated), force a 'Sleep Phase' to consolidate memories and clear cache.",
        "priority": "medium",
        "source": "user"
    }
]

def inject():
    if not os.path.exists(STATE_FILE): return
    with open(STATE_FILE, 'r') as f: state = json.load(f)
    
    last_id = state.get("last_task_id", 0)
    todos = state.get("todo_list", [])
    
    print(f"{Fore.CYAN}>>> 正在注入【生存本能】与【在线学习】任务...")
    
    for task_def in reversed(SURVIVAL_TASKS):
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
    
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    
    print(f"\n{Fore.MAGENTA}>>> 注入完成！请重启 evolution_system.py。")
    print(f"{Fore.MAGENTA}>>> AI 将开始学习如何“活着”。")

if __name__ == "__main__":
    inject()