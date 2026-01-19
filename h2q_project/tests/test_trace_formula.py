# tests/test_trace_formula.py

import torch
import pytest
from h2q.trace_formula import TraceFormulaValidator

# 场景1：一个“理想”的环境和决策历史，它们应该几乎完美匹配
def ideal_mu_func(E: torch.Tensor) -> torch.Tensor:
    # 假设μ(E) = 0.02 * E，这是一个简单的线性增长
    return 0.02 * E

# 对应的理想决策历史
# ∫ 0.02E dE from 0 to 10 = 0.02 * [E^2/2]_0^10 = 1.0
# 我们构造一组决策，其谱位移之和恰好为1.0
ideal_eta_map = {1: 0.4, 2: 0.3, 3: 0.3}
ideal_decisions = [1, 2, 3]

# 场景2：一个“失衡”的决策历史
unbalanced_eta_map = {1: 5.0, 2: 3.0, 3: 2.0} # 和为10.0，远大于1.0
unbalanced_decisions = [1, 2, 3]

def test_validator_balanced_scenario():
    validator = TraceFormulaValidator(tolerance=0.1)
    
    is_valid, error = validator.validate(
        decision_indices=ideal_decisions,
        eta_map=ideal_eta_map,
        mu_func=ideal_mu_func,
        energy_limit=10.0
    )
    
    print(f"Balanced Scenario - Error: {error:.4f}")
    assert is_valid is True
    assert error < 0.1

def test_validator_unbalanced_scenario():
    validator = TraceFormulaValidator(tolerance=0.1)
    
    is_valid, error = validator.validate(
        decision_indices=unbalanced_decisions,
        eta_map=unbalanced_eta_map,
        mu_func=ideal_mu_func, # 环境不变
        energy_limit=10.0
    )
    
    print(f"Unbalanced Scenario - Error: {error:.4f}")
    assert is_valid is False
    assert error > 0.1

def test_validator_no_decisions():
    validator = TraceFormulaValidator(tolerance=0.1)
    
    # 假设环境积分为0
    zero_mu_func = lambda E: 0.0 * E
    
    is_valid, error = validator.validate(
        decision_indices=[],
        eta_map={},
        mu_func=zero_mu_func,
        energy_limit=10.0
    )
    
    # 离散和为0，积分为0，误差应该接近0
    print(f"No Decision Scenario - Error: {error:.4f}")
    assert is_valid is True
    assert error < 1e-6