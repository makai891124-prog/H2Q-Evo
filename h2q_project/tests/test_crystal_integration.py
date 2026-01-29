# tests/test_crystal_integration.py

import torch
import pytest
import os
from h2q.dde import DiscreteDecisionEngine
from h2q.core.discrete_decision_engine import get_canonical_dde

def test_crystal_loading_and_inference():
    print("\n🧪 [测试] 启动记忆晶体集成测试...")
    
    # 1. 检查晶体文件是否存在
    crystal_path = "h2q_memory.pt"
    if not os.path.exists(crystal_path):
        pytest.skip("跳过测试：未找到 h2q_memory.pt，请先运行 tools/prism_converter.py")
    
    # 2. 初始化 DDE (几何版)
    # 我们使用标准的 256 维空间
    dde = get_canonical_dde(dim=256, n_choices=256)
    print("✅ DDE 引擎初始化完成")

    # 3. 加载记忆晶体
    print(f"🔮 正在尝试加载: {crystal_path}")
    dde.load_memory_crystal(crystal_path)
    
    # 验证是否加载进去了
    assert dde.external_memory is not None
    # GPT-2 的词表大小是 50257，压缩维度是 256
    assert dde.external_memory.shape == (50257, 256)
    print(f"✅ 晶体挂载成功！维度确认: {dde.external_memory.shape}")

    # 4. 试运行：模拟一次决策
    # 构造一个虚拟的上下文 (Batch=1, Dim=256)
    context = torch.randn(1, 256)
    # 强制归一化，模拟真实的几何状态
    context = torch.nn.functional.normalize(context, p=2, dim=-1)
    
    # 构造一组候选行动 (索引形式)，模拟系统在思考要选择哪个词
    # 假设我们在考虑 ID 为 100, 200, 300 的三个词
    candidate_actions = torch.tensor([[100, 200, 300]], dtype=torch.long)
    
    print("⚙️ 正在执行几何决策 (Forward Pass)...")
    # 执行决策
    chosen, metadata = dde(context, candidate_actions)
    
    # 5. 验证输出
    assert chosen.shape == (1,) # 应该选出一个行动
    assert 'eta_values' in metadata
    
    # 检查 η (谱位移) 是否被计算出来
    eta = metadata['eta_values']
    print(f"✅ 决策完成。计算出的谱位移 (η): {eta.detach().cpu().numpy()}")
    
    # 验证 η 的范围是否在 [0, 3.14] (0 到 Pi 弧度)
    assert (eta >= 0).all() and (eta <= 3.14159).all()
    print("✅ 谱位移数值符合射影几何约束。")

if __name__ == "__main__":
    # 允许直接运行此脚本
    test_crystal_loading_and_inference()