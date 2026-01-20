# tests/test_sst.py

import pytest
from h2q.core.sst import SpectralShiftTracker

def test_sst_normal_learning_is_monotone():
    # 思想实验1：模拟一个正常的、持续学习的过程
    sst = SpectralShiftTracker()
    
    # 累积η值持续增加
    sst.update(t=1.0, eta_cumulative=0.5)
    sst.update(t=2.0, eta_cumulative=1.2)
    monotone_status = sst.update(t=3.0, eta_cumulative=1.8)
    
    assert sst.is_monotone is True
    assert monotone_status is True

def test_sst_detects_catastrophic_forgetting():
    # 思想实验2：模拟一次灾难性遗忘
    sst = SpectralShiftTracker()
    
    sst.update(t=1.0, eta_cumulative=0.5)
    sst.update(t=2.0, eta_cumulative=1.2)
    # 突然插入一个更小的值！
    monotone_status = sst.update(t=3.0, eta_cumulative=0.9)
    
    assert sst.is_monotone is False
    assert monotone_status is False
    
    # 即使之后恢复增长，状态也应该是False，因为它记录了发生过遗忘
    sst.update(t=4.0, eta_cumulative=1.5)
    assert sst.is_monotone is False

def test_sst_computes_invariants_correctly():
    # 思想实验3：验证全局统计数据的计算
    sst = SpectralShiftTracker()
    
    sst.update(t=0.0, eta_cumulative=0.0)
    sst.update(t=1.0, eta_cumulative=0.5)
    sst.update(t=2.0, eta_cumulative=1.2)
    sst.update(t=3.0, eta_cumulative=1.8)
    
    invariants = sst.compute_global_invariants()
    
    # 预期总学习量是最后一个η值
    assert invariants['total_learning'] == 1.8
    # 预期总时长是 3.0 - 0.0
    assert invariants['duration'] == 3.0
    # 预期平均学习率是 1.8 / 3.0
    assert invariants['learning_rate_avg'] == pytest.approx(0.6)