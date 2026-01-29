#!/usr/bin/env python3
"""
检查真实权重文件的结构和内容
"""

import torch
import os
import sys

def check_weight_file(file_path):
    """检查权重文件"""
    print(f"\n=== 检查文件: {os.path.basename(file_path)} ===")

    if not os.path.exists(file_path):
        print("文件不存在")
        return

    try:
        # 尝试加载
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"加载成功! 类型: {type(data)}")

        if isinstance(data, dict):
            print(f"字典键数量: {len(data)}")
            keys = list(data.keys())
            print(f"前5个键: {keys[:5]}")

            # 检查第一个张量的形状
            if keys:
                first_key = keys[0]
                first_value = data[first_key]
                if isinstance(first_value, torch.Tensor):
                    print(f"第一个张量形状: {first_value.shape}")
                    print(f"第一个张量类型: {first_value.dtype}")

            # 计算总参数量
            total_params = 0
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    total_params += value.numel()
            print(f"总参数量: {total_params:,}")

            # 检查是否有常见的Transformer键
            transformer_keys = [k for k in keys if any(x in k.lower() for x in ['attention', 'mlp', 'embed', 'norm', 'lm_head'])]
            print(f"Transformer相关键数量: {len(transformer_keys)}")

        elif isinstance(data, torch.Tensor):
            print(f"单个张量形状: {data.shape}")
            print(f"单个张量类型: {data.dtype}")
            print(f"参数量: {data.numel():,}")

        else:
            print(f"其他类型: {type(data)}")

    except Exception as e:
        print(f"加载失败: {e}")

def main():
    """主函数"""
    files_to_check = [
        '/Users/imymm/H2Q-Evo/h2q_project/h2q_qwen_crystal.pt',
        '/Users/imymm/H2Q-Evo/h2q_project/h2q_model_v2.pth',
        '/Users/imymm/H2Q-Evo/h2q_project/h2q/agi/real_checkpoints/best_model.pt'
    ]

    for file_path in files_to_check:
        check_weight_file(file_path)

if __name__ == "__main__":
    main()