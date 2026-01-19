import json
import os
from colorama import Fore, init

init(autoreset=True)
STATE_FILE = "evo_state.json"

def reset():
    if not os.path.exists(STATE_FILE): return
    with open(STATE_FILE, 'r') as f: state = json.load(f)
    
    count = 0
    for t in state['todo_list']:
        if t['status'] == 'failed':
            t['status'] = 'pending'
            count += 1
            print(f"{Fore.GREEN}>>> 重置任务 ID {t['id']}: {t['task'][:50]}...")
            
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    print(f"\n{Fore.MAGENTA}>>> 已重置 {count} 个失败任务。请重启 evolution_system.py")

if __name__ == "__main__":
    reset()