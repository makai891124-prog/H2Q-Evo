# tests/test_dde.py

import torch
import pytest
import torch.nn as nn
from h2q.dde import DiscreteDecisionEngine

# 模拟一个简单的任务损失函数
# 假设行动向量的“长度”越小，损失越低
def mock_task_loss_fn(context, action):
    return torch.mean(action**2, dim=1)

def test_dde_chooses_low_loss_when_alpha_is_zero():
    # 思想实验1：当自主性权重为0，DDE应退化为标准的“损失最小化”智能体
    
    context_dim = 16
    action_dim = 4
    
    # α = 0
    dde = DiscreteDecisionEngine(context_dim, action_dim, autonomy_weight=0)
    
    context = torch.randn(1, context_dim)
    
    # 行动A: 损失低 (向量长度小)
    action_A = torch.tensor([[[0.1, 0.1, 0.1, 0.1]]]) # loss ≈ 0.01
    # 行动B: 损失高 (向量长度大)
    action_B = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]) # loss ≈ 1.0
    
    candidate_actions = torch.cat([action_A, action_B], dim=1)
    
    chosen_actions, metadata = dde(context, candidate_actions, mock_task_loss_fn)
    
    # 预期：必须选择行动A (索引为0)
    chosen_index = metadata['chosen_action_indices'].item()
    assert chosen_index == 0

def test_dde_chooses_high_eta_when_alpha_is_high():
    # 思想实验2：当自主性权重足够高，DDE会为了高η值而容忍更高的任务损失
    
    context_dim = 16
    action_dim = 4
    
    # α = 10.0 (一个很高的值，足以压过任务损失的差异)
    dde = DiscreteDecisionEngine(context_dim, action_dim, autonomy_weight=10.0)
    
    # 关键修正：我们的“假”函数现在只接收一个参数，
    # 因为当它被调用时，只会传递 action 这一个参数。
    def manipulated_eta_fn(action):
        # action 的形状是 [B, action_dim]，例如 [[0.1, 0.1, 0.1, 0.1]]
        # 我们需要比较 action 的内容，所以我们取第一个元素
        action_content = action[0]
        
        # 如果行动是A (向量小)，返回低η
        if torch.allclose(action_content, torch.tensor([0.1, 0.1, 0.1, 0.1])):
            return torch.tensor([0.01]) # 低学习价值
        # 如果行动是B (向量大)，返回高η
        else:
            return torch.tensor([0.5]) # 高学习价值
            
    # 我们仍然替换整个模块，但这次我们用一个只接受一个参数的“可调用对象”
    # 为了避免上次的 TypeError，我们创建一个简单的 nn.Module 包装器
    class Manipulator(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
        def forward(self, action):
            return self.func(action)

    dde.spectral_shift_fn = Manipulator(manipulated_eta_fn)
    
    context = torch.randn(1, context_dim)
    action_A = torch.tensor([[[0.1, 0.1, 0.1, 0.1]]]) # loss≈0.01, η=0.01 -> score≈-0.01+10*0.01=0.09
    action_B = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]) # loss≈1.0,  η=0.5  -> score≈-1.0 +10*0.5 =4.0
    
    candidate_actions = torch.cat([action_A, action_B], dim=1)
    
    chosen_actions, metadata = dde(context, candidate_actions, mock_task_loss_fn)
    
    # 预期：必须选择行动B (索引为1)，因为它的总得分更高
    chosen_index = metadata['chosen_action_indices'].item()
    assert chosen_index == 1