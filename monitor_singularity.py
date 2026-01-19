import json
import time
import os  # <--- 之前漏掉了这个
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from pathlib import Path

STATE_FILE = "evo_state.json"

def get_metrics():
    if not os.path.exists(STATE_FILE): return None
    
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            
        # 1. 进化代数 (Generation)
        gen = state.get('generation', 0)
        
        # 2. 任务吞吐量 (Tasks Completed)
        completed = len([t for t in state.get('todo_list', []) if t.get('status') == 'completed'])
        
        # 3. 代码库复杂度 (Codebase Size)
        total_lines = 0
        # 统计 h2q_project 下所有 .py 文件的行数
        if os.path.exists("./h2q_project"):
            for root, _, files in os.walk("./h2q_project"):
                for file in files:
                    if file.endswith(".py"):
                        try:
                            with open(Path(root)/file, 'r', encoding='utf-8', errors='ignore') as f:
                                total_lines += len(f.readlines())
                        except: pass
                    
        return gen, completed, total_lines
    except Exception as e:
        print(f"读取指标失败: {e}")
        return None

# 初始化数据容器
history_time = []
history_gen = []
history_tasks = []
history_lines = []

# 设置深色模式图表
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
fig.suptitle('H2Q-Evo Singularity Monitor', fontsize=16, color='cyan')

def animate(i):
    metrics = get_metrics()
    if not metrics: return
    
    gen, tasks, lines = metrics
    now = datetime.now()
    
    history_time.append(now)
    history_gen.append(gen)
    history_tasks.append(tasks)
    history_lines.append(lines)
    
    # 保持最近 50 个数据点，让图表滚动起来
    if len(history_time) > 50:
        history_time.pop(0)
        history_gen.pop(0)
        history_tasks.pop(0)
        history_lines.pop(0)
    
    # 绘制图表 1: 进化代数
    ax1.clear()
    ax1.plot(history_time, history_gen, color='#00ffcc', linewidth=2)
    ax1.set_ylabel('Generation', color='#00ffcc')
    ax1.grid(True, alpha=0.2)
    ax1.text(history_time[-1], history_gen[-1], f' {gen}', color='#00ffcc', fontweight='bold')
    
    # 绘制图表 2: 完成任务数
    ax2.clear()
    ax2.plot(history_time, history_tasks, color='#00ff00', linewidth=2)
    ax2.set_ylabel('Tasks Completed', color='#00ff00')
    ax2.grid(True, alpha=0.2)
    ax2.text(history_time[-1], history_tasks[-1], f' {tasks}', color='#00ff00', fontweight='bold')
    
    # 绘制图表 3: 代码行数
    ax3.clear()
    ax3.plot(history_time, history_lines, color='#ff00ff', linewidth=2)
    ax3.set_ylabel('Code Lines', color='#ff00ff')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.2)
    ax3.text(history_time[-1], history_lines[-1], f' {lines}', color='#ff00ff', fontweight='bold')
    
    # 奇点预警逻辑 (如果最近5个点斜率极高)
    if len(history_gen) > 5:
        speed = history_gen[-1] - history_gen[-5]
        if speed > 3: # 短时间内进化超过3代
            ax1.set_title(f'⚠️ SINGULARITY SPIKE DETECTED! Speed: {speed}/cycle', color='red', fontweight='bold')
        else:
            ax1.set_title(f'System Status: Stable Evolution (Gen {gen})', color='white')

# 启动动画 (cache_frame_data=False 修复警告)
ani = animation.FuncAnimation(fig, animate, interval=2000, cache_frame_data=False)

print(">>> 监控仪表盘已启动。请查看弹出的窗口。")
print(">>> 按 Ctrl+C 退出。")
plt.show()