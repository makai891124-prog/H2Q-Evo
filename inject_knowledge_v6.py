import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

# 外部知识注入任务 (Math, Code, Physics, DNA)
KNOWLEDGE_TASKS = [
    {
        "task": "System Upgrade: Update 'Dockerfile' to include 'datasets' (Hugging Face) and 'zstandard' libraries to enable petabyte-scale stream processing without local storage overhead.",
        "priority": "critical"
    },
    {
        "task": "Create 'h2q/data/universal_stream.py': Implement 'UniversalStreamLoader' to stream 'openwebmath' (Math) and 'the_stack' (Code) from Hugging Face, converting raw text into H2Q Byte-Stream format for the Knot Kernel.",
        "priority": "high"
    },
    {
        "task": "Implement 'train_omniscience.py': Create a multi-modal training loop that rotates between Math (Logic), Physics (ArXiv/Causality), and Genomics (DNA Topology) to minimize 'Fractal Dimension Collapse' across domains.",
        "priority": "high"
    },
    {
        "task": "DNA Topology Analysis: Use 'UniversalStreamLoader' to stream 'GenomicBenchmarks', mapping ATCG sequences to Quaternion rotations to identify topological invariants in non-coding regions.",
        "priority": "medium"
    }
]

def inject():
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件，请等待 evolution_system.py 初始化完毕。")
        return

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    except json.JSONDecodeError:
        print(f"{Fore.RED}状态文件损坏或为空。")
        return
    
    # 1. 获取全局 ID 计数器
    last_id = state.get("last_task_id", 0)
    todos = state.get("todo_list", [])
    
    print(f"{Fore.CYAN}>>> 正在注入【全知之桥】任务 (v6.0 Ledger Compatible)...")
    
    # 2. 批量注入
    for task_def in KNOWLEDGE_TASKS:
        last_id += 1
        new_task = {
            "id": last_id,
            "task": task_def['task'],
            "priority": task_def['priority'],
            "status": "pending",
            "source": "user"  # 关键：标记为用户注入，防止被 AI 清理
        }
        
        # 策略：Critical 插到最前，High/Medium 插到 Pending 列表的前部
        if new_task['priority'] == 'critical':
            todos.insert(0, new_task)
        else:
            # 找到插入点（在 critical 之后）
            insert_idx = 0
            for i, t in enumerate(todos):
                if t['status'] == 'pending':
                    if t.get('priority') == 'critical':
                        insert_idx = i + 1
                    else:
                        break
            todos.insert(insert_idx, new_task)
            
        print(f"{Fore.GREEN} + [ID {last_id}] {new_task['task'][:60]}...")

    # 3. 更新状态
    state["last_task_id"] = last_id
    state["todo_list"] = todos
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n{Fore.MAGENTA}>>> 注入完成！AI 将在当前感知结束后立即处理这些任务。")

if __name__ == "__main__":
    inject()