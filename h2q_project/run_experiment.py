# run_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from h2q.system import AutonomousSystem

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

# --- 3. 初始化我们的自主系统和优化器 ---
# 系统的“上下文”是输入序列的维度
# 系统的“行动”是它对输出的预测（一个标量）
system = AutonomousSystem(context_dim=3, action_dim=1)

# 我们需要优化的参数来自 DDE 和 CEM
params_to_optimize = list(system.dde.parameters()) + list(system.cem.parameters())
optimizer = optim.Adam(params_to_optimize, lr=0.001)

# --- 4. 实验记录器 ---
history = {
    'loss': [],
    'autonomy_score': [], # 我们将定义一个简单的自主性得分
    'eta_total': [],
    'trace_error': [] # (未来添加)
}

# --- 5. 训练循环 (核心修改) ---
print("Starting experiment...")
for episode in range(2000):
    # a. 获取数据 (不变)
    context, y_true = get_data_batch()
    
    # b. 生成候选行动 (不变)
    # 为了简化，我们让系统围绕一个“基本预测”生成几个候选行动
    # (一个更复杂的DDE会自己生成候选行动)
    base_prediction = torch.mean(context, dim=1, keepdim=True) # 一个简单的启发式
    candidate_actions = torch.stack([
        base_prediction - 0.5,
        base_prediction,
        base_prediction + 0.5
    ], dim=1) # Shape: [B, 3, 1] (3个候选行动)

    
    # c. 定义任务损失函数 (不变)
    def step_task_loss_fn(ctx, action):
        return loss_fn(action, y_true)

    # d. DDE 决策 (现在返回概率和对数概率)
    chosen_actions, metadata = system.dde(context, candidate_actions, step_task_loss_fn)
    
    # e. 计算“奖励” (Reward)
    # 奖励 = -任务损失。我们希望奖励越大越好。
    with torch.no_grad():
        reward = -loss_fn(chosen_actions, y_true)
    
    # f. 计算策略梯度损失 (Policy Gradient Loss)
    # 损失 = -log(π(a|s)) * R
    # 我们希望最大化 log(π)*R，所以最小化 -log(π)*R
    log_prob = metadata['log_prob']
    policy_loss = -log_prob * reward
    policy_loss = policy_loss.mean() # 对批次取平均

    # g. 优化
    optimizer.zero_grad()
    policy_loss.backward() # 现在 policy_loss 是可微分的！
    optimizer.step()
    
    # h. 记录数据 (逻辑微调)
    with torch.no_grad():
        task_loss = loss_fn(chosen_actions, y_true)
        eta_this_step = system.dde.spectral_shift_fn(chosen_actions).mean().item()
        
        system.cumulative_eta += eta_this_step
        system.sst.update(episode, system.cumulative_eta)
        
        history['loss'].append(task_loss.item())
        history['eta_total'].append(system.cumulative_eta)
        
        # 自主性得分计算逻辑不变
        best_task_action_idx = torch.argmin(
            torch.stack([step_task_loss_fn(context, candidate_actions[:,i,:]) for i in range(3)]),
            dim=0
        )
        autonomy_choice = (metadata['chosen_action_indices'] != best_task_action_idx).float().mean().item()
        history['autonomy_score'].append(autonomy_choice)

    if episode % 200 == 0:
        # 我们打印 policy_loss 来监控优化过程
        print(f"Episode {episode}: Task Loss={task_loss.item():.4f}, Policy Loss={policy_loss.item():.4f}, Autonomy Score={autonomy_choice:.2f}")

# --- 6. 可视化结果 ---
print("Experiment finished. Plotting results...")
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Autonomous System - First Experiment')

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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()