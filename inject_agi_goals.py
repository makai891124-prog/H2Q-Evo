import json
import os
from colorama import Fore, init

init(autoreset=True)

STATE_FILE = "evo_state.json"

# AGI 级任务清单
AGI_TASKS = [
    {
        "id": 101,
        "task": "Implement 'Meta-Learning Loop': Create a new module 'h2q/meta_learner.py' that allows the DiscreteDecisionEngine to update its own 'autonomy_weight' based on the loss history stored in SST (Spectral Shift Tracker).",
        "priority": "high",
        "status": "pending"
    },
    {
        "id": 102,
        "task": "Enable 'Dreaming' Mechanism: Modify 'train_full_stack_v2.py' to include a 'sleep phase' where the model generates synthetic data from its own Knot Kernel and trains on it to reinforce rare concepts (Replay Buffer).",
        "priority": "high",
        "status": "pending"
    },
    {
        "id": 103,
        "task": "Self-Modification Capability: Create a 'tools/code_writer.py' that allows the H2Q system to output Python code strings, theoretically enabling the model to write its own tools in the future (Sandbox restricted).",
        "priority": "medium",
        "status": "pending"
    }
]

def inject_tasks():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}错误：找不到 {STATE_FILE}")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # 获取当前最大的 ID
    current_ids = [t['id'] for t in state['todo_list']]
    max_id = max(current_ids) if current_ids else 0
    
    print(f"{Fore.CYAN}>>> 正在注入 AGI 级任务...")
    
    for task in AGI_TASKS:
        # 重新分配 ID 以防冲突
        max_id += 1
        task['id'] = max_id
        state['todo_list'].append(task)
        print(f"{Fore.GREEN} + [注入] {task['task'][:60]}...")
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n{Fore.MAGENTA}>>> 注入完成！请重启 evolution_system.py 或等待其自动读取。")

if __name__ == "__main__":
    inject_tasks()