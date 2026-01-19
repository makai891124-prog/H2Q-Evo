# h2q/sst.py

from typing import List, Dict

class SpectralShiftTracker:
    """
    谱位移追踪器 (SST)。
    
    负责记录学习历史 (η_t)，检查学习的单调性（防止灾难性遗忘），
    并计算全局学习不变量。
    """
    
    def __init__(self):
        # 学习历史记录，每个元素是 {'t': float, 'eta': float}
        self.eta_history: List[Dict[str, float]] = []
        # 状态标志，初始为True
        self.is_monotone: bool = True
        
    def update(self, t: float, eta_cumulative: float) -> bool:
        """
        用新的累积谱位移值更新历史记录，并检查单调性。
        
        Args:
            t (float): 当前时间点。
            eta_cumulative (float): 到当前时间为止的总累积谱位移。
            
        Returns:
            bool: 当前学习轨迹是否仍然是单调的。
        """
        if len(self.eta_history) > 0:
            # 获取上一个记录的η值
            prev_eta = self.eta_history[-1]['eta']
            if eta_cumulative < prev_eta:
                # 发生遗忘！
                self.is_monotone = False
                print(f"WARNING: Non-monotone learning detected at t={t}. "
                      f"η decreased from {prev_eta:.4f} to {eta_cumulative:.4f}.")
        
        self.eta_history.append({'t': t, 'eta': eta_cumulative})
        
        return self.is_monotone

    def compute_global_invariants(self) -> Dict[str, float]:
        """
        从完整的学习历史中计算宏观统计数据。
        """
        if not self.eta_history:
            return {
                'total_learning': 0.0,
                'duration': 0.0,
                'learning_rate_avg': 0.0
            }
            
        etas = [h['eta'] for h in self.eta_history]
        times = [h['t'] for h in self.eta_history]
        
        total_learning = etas[-1] # 累积学习量就是最后一个值
        duration = times[-1] - times[0]
        
        # 平均学习率 = 总学习量 / 总时长
        learning_rate_avg = total_learning / duration if duration > 0 else 0.0
        
        return {
            'total_learning': total_learning,
            'duration': duration,
            'learning_rate_avg': learning_rate_avg
        }

    def reset(self):
        """
        重置追踪器状态，用于新的实验。
        """
        self.eta_history = []
        self.is_monotone = True