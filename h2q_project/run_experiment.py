# run_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from h2q_project.src.h2q.core.discrete_decision_engine import get_canonical_dde, LatentConfig
from h2q_project.src.h2q.core.sst import SpectralShiftTracker
from h2q_project.src.h2q.core.unified_orchestrator import get_orchestrator, Unified_Homeostatic_Orchestrator
from typing import Dict, Any, Optional

# 统一数学架构（用于特征增强与指标记录）
try:
    from das_core import create_das_based_architecture
    _das_arch = create_das_based_architecture(dim=256)
except Exception:
    _das_arch = None

ORCHESTRATOR_CONFIG: Dict[str, Any] = {
    "memory_threshold_gb": 14.0,
    "ssd_cache_path": "./vault/ssd_paging",
}

try:
    _orchestrator: Optional[Unified_Homeostatic_Orchestrator] = get_orchestrator(ORCHESTRATOR_CONFIG)
except Exception:
    _orchestrator = None

DDE_CONFIG = LatentConfig(dim=256, n_choices=3, device="cpu")
_dde = get_canonical_dde(config=DDE_CONFIG)
_sst = SpectralShiftTracker()
_cumulative_eta = 0.0


class ExperimentManifold:
    """Thin adapter exposing experiment state to the orchestrator."""

    def __init__(self):
        self.latest_eta = 0.0
        self.snapshot: Dict[str, torch.Tensor] = {}

    def update(self, eta: float, snapshot: Optional[Dict[str, torch.Tensor]] = None) -> None:
        self.latest_eta = eta
        self.snapshot = snapshot or {}

    def get_knot_state(self) -> Dict[str, torch.Tensor]:
        return self.snapshot

    def get_activity_spectrum(self) -> torch.Tensor:
        return torch.tensor([self.latest_eta], dtype=torch.float32)

    def compress_idle_knots(self) -> int:
        return 0

    def calculate_hdi(self) -> float:
        return 0.0

    def inject_fractal_noise(self, strength: float) -> None:
        return None

def pad_to_dim(x: torch.Tensor, target_dim: int = 256) -> torch.Tensor:
    """将输入右侧零填充到指定维度"""
    if x.shape[1] >= target_dim:
        return x[:, :target_dim]
    pad = torch.zeros(x.shape[0], target_dim - x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

# --- 1. 定义我们的“世界”：一个简单的数据生成器 ---
def get_data_batch(batch_size=32):
    """生成一批简单的线性序列数据"""
    # 例如，生成 [x, x+1, x+2] -> y = x+3
    start = torch.randn(batch_size, 1) * 10
    X = torch.cat([start, start + 1, start + 2], dim=1) # Shape: [B, 3]
    y = start + 3 # Shape: [B, 1]
    return X, y

# --- 2. 定义任务损失函数 ---
# 这是一个标准的均方误差损失
loss_fn = nn.MSELoss()

# --- 3. 初始化决策引擎和优化器 ---
optimizer = optim.Adam(_dde.parameters(), lr=0.001)

# --- 4. 实验记录器 ---
history = {
    'loss': [],
    'autonomy_score': [], # 我们将定义一个简单的自主性得分
    'eta_total': [],
    'trace_error': [], # (未来添加)
    'math_integrity': [],
    'fueter_curvature': [],
    'homeostasis': [],
}

manifold_view = ExperimentManifold()

# --- 5. 训练循环 (核心修改) ---
print("Starting experiment...")
for episode in range(2000):
    # a. 获取数据 (不变)
    context, y_true = get_data_batch()

    # 使用DAS架构进行特征增强与指标记录（可选）
    if _das_arch is not None:
        with torch.no_grad():
            enriched_input = pad_to_dim(context, 256)
            das_result = _das_arch(enriched_input)
            # 记录DAS指标
            fueter_curv = das_result.get('invariant_distances', 0.0)
            integrity = das_result.get('dimension', 3) / 8.0  # 简化的完整性度量
            history['fueter_curvature'].append(float(fueter_curv))
            history['math_integrity'].append(float(integrity))
    
    # b. 生成候选行动 (不变)
    # 为了简化，我们让系统围绕一个“基本预测”生成几个候选行动
    # (一个更复杂的DDE会自己生成候选行动)
    base_prediction = torch.mean(context, dim=1, keepdim=True) # 一个简单的启发式
    candidate_actions = torch.stack([
        base_prediction - 0.5,
        base_prediction,
        base_prediction + 0.5
    ], dim=1) # Shape: [B, 3, 1] (3个候选行动)

    
    # c. 定义任务损失函数 (逐样本 MSE)
    def step_task_loss_fn(ctx, action):
        return ((action - y_true) ** 2).mean(dim=1)

    # d. DDE 决策（使用候选索引进行选择）
    batch_size = context.size(0)
    candidate_ids = torch.tensor([0, 1, 2], device=context.device).repeat(batch_size, 1)
    dde_input = pad_to_dim(context, 256)
    chosen_indices, metadata = _dde(dde_input, candidate_ids)

    chosen_actions = candidate_actions[torch.arange(batch_size), chosen_indices]

    # e. 计算“奖励” (Reward)
    sample_losses = ((chosen_actions - y_true) ** 2).mean(dim=1)
    reward = -sample_losses

    # f. 计算策略梯度损失 (Policy Gradient Loss)
    probs = metadata.get('probabilities') if isinstance(metadata, dict) else None
    if probs is not None:
        selected_prob = probs[torch.arange(batch_size), chosen_indices].clamp_min(1e-8)
        log_prob = torch.log(selected_prob)
    else:
        log_prob = torch.zeros_like(reward)
    policy_loss = -(log_prob * reward).mean()

    # g. 优化
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # h. 记录数据 (逻辑微调)
    with torch.no_grad():
        task_loss = sample_losses.mean().item()
        eta_vals = metadata.get('eta_values') if isinstance(metadata, dict) else None
        eta_this_step = float(eta_vals.mean().item()) if eta_vals is not None else 0.0
        _cumulative_eta += eta_this_step
        _sst.update(episode, _cumulative_eta)
        
        history['loss'].append(task_loss)
        history['eta_total'].append(_cumulative_eta)
        
        # 自主性得分：选择是否偏离最优任务损失
        candidate_losses = torch.stack([step_task_loss_fn(context, candidate_actions[:, i, :]) for i in range(3)], dim=1)
        best_task_action_idx = torch.argmin(candidate_losses, dim=1)
        autonomy_choice = (chosen_indices != best_task_action_idx).float().mean().item()
        history['autonomy_score'].append(autonomy_choice)

        manifold_view.update(
            eta_this_step,
            {"candidate_actions": candidate_actions.detach()}
        )
        if _orchestrator is not None:
            status = _orchestrator.step(manifold_view)
            history['homeostasis'].append(status or {"action": "none"})
        else:
            history['homeostasis'].append({"action": "orchestrator_unavailable"})

    if episode % 200 == 0:
        # 我们打印 policy_loss 来监控优化过程
        extra = ""
        if _das_arch is not None:
            extra = f", Integrity={history['math_integrity'][-1]:.3f}, Invariant Distance={history['fueter_curvature'][-1]:.3f}"
        print(f"Episode {episode}: Task Loss={task_loss:.4f}, Policy Loss={policy_loss.item():.4f}, Autonomy Score={autonomy_choice:.2f}{extra}")

# --- 6. 可视化结果 ---
print("Experiment finished. Plotting results...")
rows = 4 if _das_arch is not None else 3
fig, axs = plt.subplots(rows, 1, figsize=(10, 4 * rows))
fig.suptitle('DAS-Based Autonomous System - First Experiment')

axs[0].plot(history['loss'])
axs[0].set_title('Task Loss over Episodes')
axs[0].set_ylabel('MSE Loss')
axs[0].grid(True)

axs[1].plot(history['autonomy_score'])
axs[1].set_title('Autonomy Score (Fraction of non-optimal task choices)')
axs[1].set_ylabel('Score')
axs[1].grid(True)

axs[2].plot(history['eta_total'])
axs[2].set_title('Cumulative Spectral Shift (Total Learning)')
axs[2].set_ylabel('Cumulative η')
axs[2].set_xlabel('Episode')
axs[2].grid(True)

if _das_arch is not None:
    axs[3].plot(history['math_integrity'])
    axs[3].set_title('Mathematical Integrity (DAS Architecture)')
    axs[3].set_ylabel('Integrity')
    axs[3].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()