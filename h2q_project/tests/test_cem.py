# tests/test_cem.py

import torch
import torch.nn as nn
import pytest
from h2q.cem import ContinuousEnvironmentModel

def test_cem_can_learn_a_simple_function():
    """
    思想实验：验证CEM是否具备学习能力。
    我们将定义一个目标函数 μ_true(E)，然后训练CEM来拟合它。
    """
    
    # 1. 定义“真理”环境函数：μ_true(E) = 0.1 * E + 0.05 * E^2
    def mu_true_func(E):
        return 0.1 * E + 0.05 * E**2

    # 2. 创建CEM实例和优化器
    cem = ContinuousEnvironmentModel()
    optimizer = torch.optim.Adam(cem.parameters(), lr=0.01)
    loss_fn = nn.MSELoss() # 使用均方误差作为损失函数

    # 3. 生成训练数据
    # 就像智能体在环境中探索，收集到200个 (能量, 压力) 数据点
    E_train = torch.linspace(0, 5, 200).unsqueeze(-1)
    mu_true = mu_true_func(E_train)

    # 4. 训练循环
    for epoch in range(300):
        # CEM 做出预测
        mu_pred = cem(E_train)
        
        # 计算预测与“真理”之间的差距
        loss = loss_fn(mu_pred, mu_true)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 5. 验证学习效果
    # 我们用一组新的、模型没见过的数据来测试
    E_test = torch.tensor([[0.5], [1.5], [2.5], [4.5]])
    mu_true_test = mu_true_func(E_test)
    
    # 关闭梯度计算，进行评估
    with torch.no_grad():
        mu_pred_test = cem(E_test)

    # 打印结果，方便观察
    print("\nCEM Learning Validation:")
    for i in range(len(E_test)):
        print(f"E={E_test[i].item():.1f}, True μ={mu_true_test[i].item():.4f}, Predicted μ={mu_pred_test[i].item():.4f}")

    # 断言：预测值和真实值之间的差距应该非常小
    assert torch.allclose(mu_pred_test, mu_true_test, atol=0.1) # atol是绝对容忍度