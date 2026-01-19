# tests/test_cost_functional.py

import torch
import pytest
from h2q.cost_functional import AutonomyCost

# 定义一个简单的mu函数用于测试
def mock_mu_func(E: torch.Tensor) -> torch.Tensor:
    # 假设环境拖拽随能量线性增加
    return 0.1 * E

def test_autonomy_cost_components():
    cost_fn = AutonomyCost()
    
    decisions = torch.tensor([[1, 2]], dtype=torch.long) # Batch size = 1
    eta_map = {1: 0.5, 2: -0.2, 3: 1.0}
    t = 10.0

    # 实验1：验证每个组件的值
    # 散射成本
    expected_scatter = -0.5 * t * torch.log(torch.tensor(t))
    # 学习迹
    expected_trace = 0.5 + (-0.2) # 0.3
    # 环境拖拽 (∫ 0.1E dE from 0 to 10 = 0.1 * [E^2/2]_0^10 = 0.1 * 50 = 5.0)
    # torch.trapz会给出非常接近的近似值
    
    total_cost = cost_fn(decisions, eta_map, mock_mu_func, t)
    
    # 验证迹
    # 手动计算迹
    trace_cost_manual = torch.tensor([eta_map[d.item()] for d in decisions[0]]).sum()
    assert torch.isclose(trace_cost_manual, torch.tensor(expected_trace))

    # 验证总成本是否大致正确
    # 注意：由于trapz是近似，我们使用一个宽松的容忍度
    expected_total = expected_scatter + expected_trace + 5.0
    assert torch.allclose(total_cost, expected_total, atol=1e-1)

def test_autonomy_cost_batching():
    cost_fn = AutonomyCost()
    
    decisions = torch.tensor([[1, 2], [3, 1]], dtype=torch.long) # Batch size = 2
    eta_map = {1: 0.5, 2: -0.2, 3: 1.0}
    t = 10.0

    # 实验2：验证批处理
    total_cost = cost_fn(decisions, eta_map, mock_mu_func, t)
    
    # 验证输出形状
    assert total_cost.shape == (2,)
    
    # 验证第一个样本
    cost_sample_1 = cost_fn(decisions[0].unsqueeze(0), eta_map, mock_mu_func, t)
    assert torch.isclose(total_cost[0], cost_sample_1)
    
    # 验证第二个样本
    cost_sample_2 = cost_fn(decisions[1].unsqueeze(0), eta_map, mock_mu_func, t)
    assert torch.isclose(total_cost[1], cost_sample_2)
    # 在 tests/test_cost_functional.py 中添加
from h2q.cost_functional import SpectralShiftFunction

def test_spectral_shift_function():
    context_dim = 64
    matrix_dim = 8
    batch_size = 4
    
    ssf = SpectralShiftFunction(context_dim, matrix_dim)
    context = torch.randn(batch_size, context_dim)
    
    # 实验3：验证谱位移函数的输出
    eta = ssf(context)
    
    # 验证输出形状
    assert eta.shape == (batch_size,)
    
    # 验证输出范围。arg(det) / pi 的范围是 [-1, 1]
    assert torch.all(eta >= -1.0) and torch.all(eta <= 1.0)