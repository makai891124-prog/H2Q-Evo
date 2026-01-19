# h2q_project/tools/heartbeat.py
import sys
import time

# 强制将项目根目录加入路径
sys.path.insert(0, "/app/h2q_project")

try:
    from h2q.core.brain import H2QBrain
    
    # 实例化大脑
    brain = H2QBrain()
    
    # 永不停歇的心跳
    while True:
        brain.live_and_learn()
        # 稍微休息一下，防止 CPU 100% 占用
        time.sleep(0.1)
        
except ImportError as e:
    print(f"FATAL BOOTSTRAP ERROR: Cannot import H2Q Brain - {e}")
    print("This likely means the architecture is evolving. Supervisor will restart me.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FATAL RUNTIME ERROR in Life Cycle: {e}")