import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

# 最终具象化任务清单
MANIFESTATION_TASKS = [
    {
        "task": "Create 'demo_interactive.py': Build a CLI chat interface. It must load the trained 'DiscreteDecisionEngine' and 'H2QSyntheticEngine', accept user text input, convert it to H2Q atoms, and output the system's 'thought trace' (spectral shift) and response.",
        "priority": "critical",
        "source": "user"
    },
    {
        "task": "Create 'h2q/utils/visualizer.py': Implement a visualization tool using 'matplotlib' to plot the eigenvalues of the SU(2) manifold (the 'Dream'). It should generate a 'dream_spectrum.png' showing the geometric shape of the AI's current thought process.",
        "priority": "high",
        "source": "user"
    },
    {
        "task": "Build 'h2q_server.py': Create a lightweight FastAPI server. Expose endpoints '/chat' (for text interaction) and '/dream' (to get the current spectral state). This allows external apps to talk to H2Q.",
        "priority": "high",
        "source": "user"
    },
    {
        "task": "Self-Documentation: Create 'README_H2Q.md'. The AI must write its own documentation, explaining its architecture (L0 Knot -> L1 Concept), how to run the demo, and its mathematical philosophy.",
        "priority": "medium",
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
    
    print(f"{Fore.CYAN}>>> 正在注入【最终具象化】任务包...")
    
    # 倒序插入到列表最前面，保证执行顺序
    for task_def in reversed(MANIFESTATION_TASKS):
        last_id += 1
        new_task = {
            "id": last_id,
            "task": task_def['task'],
            "priority": task_def['priority'],
            "status": "pending",
            "source": "user"
        }
        todos.insert(0, new_task)
        print(f"{Fore.GREEN} + [ID {last_id}] {new_task['task'][:60]}...")

    state["last_task_id"] = last_id
    state["todo_list"] = todos
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n{Fore.MAGENTA}>>> 注入完成！请重启 evolution_system.py 以立即执行。")

if __name__ == "__main__":
    inject()