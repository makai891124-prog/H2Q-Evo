# tests/test_system_integration.py

import torch
import pytest
from h2q.system import AutonomousSystem

# 模拟一个简单的任务损失函数
def mock_task_loss_fn(context, action):
    return torch.mean(action**2, dim=1)

def test_system_end_to_end_flow():
    # 思想实验：验证当系统执行几个步骤后，其内部状态是否被正确更新
    
    # 1. 初始化系统
    context_dim = 16
    action_dim = 4
    system = AutonomousSystem(context_dim, action_dim)
    
    # 检查初始状态
    initial_status = system.get_system_status()
    assert initial_status['time'] == 0
    assert initial_status['sst_history_length'] == 0
    
    # 2. 模拟第一步决策
    context = torch.randn(1, context_dim)
    actions = torch.randn(1, 5, action_dim) # 5个候选行动
    
    summary_step1 = system.step(context, actions, mock_task_loss_fn)
    
    # 3. 验证第一步之后的状态
    status_step1 = system.get_system_status()
    assert status_step1['time'] == 1.0
    assert status_step1['sst_history_length'] == 1
    assert summary_step1['cumulative_eta'] == status_step1['sst_invariants']['total_learning']
    
    # 4. 模拟第二步决策
    context = torch.randn(1, context_dim)
    actions = torch.randn(1, 5, action_dim)
    
    summary_step2 = system.step(context, actions, mock_task_loss_fn)
    
    # 5. 验证第二步之后的状态
    status_step2 = system.get_system_status()
    assert status_step2['time'] == 2.0
    assert status_step2['sst_history_length'] == 2
    
    # 验证累积η是否正确
    expected_cumulative_eta = summary_step1['eta_this_step'] + summary_step2['eta_this_step']
    assert summary_step2['cumulative_eta'] == pytest.approx(expected_cumulative_eta)
    
    # 验证SST记录的最后一个值是否是当前的累积值
    assert status_step2['sst_invariants']['total_learning'] == summary_step2['cumulative_eta']