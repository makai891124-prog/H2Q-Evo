import json
import os
import sys
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def inject_task(task_desc, priority="high"):
    if not os.path.exists(STATE_FILE):
        print(f"{Fore.RED}找不到状态文件。")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # 1. 获取并自增全局 ID
    last_id = state.get("last_task_id", 0)
    new_id = last_id + 1
    state["last_task_id"] = new_id
    
    # 2. 构建新任务对象
    new_task = {
        "id": new_id,
        "task": task_desc,
        "priority": priority,
        "status": "pending",
        "source": "user"  # 标记为用户注入，受保护
    }
    
    # 3. 插入队列
    # 策略：如果是 critical，插到最前面；否则追加到 pending 列表的末尾（或按优先级排序）
    todos = state.get("todo_list", [])
    
    # 简单的插入逻辑：找到第一个非 critical 的位置插入，或者直接插到最前
    if priority == "critical":
        todos.insert(0, new_task)
        print(f"{Fore.MAGENTA}>>> [插队] 任务已置顶 (ID {new_id})")
    else:
        # 找到最后一个 pending 任务的位置
        insert_idx = 0
        for i, t in enumerate(todos):
            if t['status'] == 'pending':
                insert_idx = i + 1
            else:
                break # 假设 pending 在前
        
        # 如果列表是混合的，简单点，直接放在所有 pending 的最前面，但在 critical 之后
        # 这里为了简单，直接插到 index 0，依靠 evolution_system 的排序逻辑在下一次运行时自动调整
        todos.insert(0, new_task)
        print(f"{Fore.GREEN}>>> [注入] 任务已添加 (ID {new_id})")

    state["todo_list"] = todos
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"任务内容: {task_desc}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 safe_inject.py '任务描述' [priority]")
        print("示例: python3 safe_inject.py 'Fix the bug in dde.py' critical")
    else:
        task = sys.argv[1]
        prio = sys.argv[2] if len(sys.argv) > 2 else "high"
        inject_task(task, prio)