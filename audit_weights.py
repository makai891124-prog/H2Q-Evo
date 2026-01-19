import torch
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("./h2q_project").resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

def audit_weights():
    print(">>> 正在审计权重文件变化...")
    
    pt_files = sorted(list(CHECKPOINT_DIR.glob("*.pt")), key=os.path.getmtime)
    
    if len(pt_files) < 2:
        print(f"⚠️ 警告：只找到 {len(pt_files)} 个权重文件。无法对比变化。")
        print("   请让 'train_omniscience.py' 多跑几个 Epoch。")
        return

    # 取最新的两个文件
    old_file = pt_files[-2]
    new_file = pt_files[-1]
    
    print(f"   对比文件 A: {old_file.name}")
    print(f"   对比文件 B: {new_file.name}")
    
    try:
        w_old = torch.load(old_file, map_location='cpu')
        w_new = torch.load(new_file, map_location='cpu')
        
        # 如果保存的是 state_dict
        if isinstance(w_old, dict) and 'state_dict' in w_old:
            w_old = w_old['state_dict']
            w_new = w_new['state_dict']
        
        # 计算差异
        diff_sum = 0.0
        param_count = 0
        
        for key in w_old:
            if key in w_new:
                t1 = w_old[key].float()
                t2 = w_new[key].float()
                # 计算 L2 距离
                diff = torch.norm(t1 - t2).item()
                diff_sum += diff
                param_count += 1
        
        print(f"\n📊 审计结果:")
        print(f"   比较了 {param_count} 层参数")
        print(f"   总差异值 (L2 Norm): {diff_sum:.6f}")
        
        if diff_sum == 0.0:
            print("❌ 致命警告：权重完全没有变化！")
            print("   原因可能是：1. 学习率为 0；2. 梯度断裂；3. 优化器未 step()。")
            print("   结论：模型没有在进化，只是在空转。")
        else:
            print("✅ 确认：权重发生了数学改变。")
            print("   结论：梯度正在流动，模型正在物理层面上'改变'。")

    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    audit_weights()