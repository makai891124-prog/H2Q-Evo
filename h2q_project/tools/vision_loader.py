# tools/vision_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# --- 配置 ---
# CIFAR-10 图片大小为 32x32x3 = 3072 字节
IMG_SIZE = 32
SEQ_LEN = IMG_SIZE * IMG_SIZE * 3 

class H2QCIFAR10(Dataset):
    def __init__(self, split="train", download=True):
        """
        标准 CIFAR-10 数据集加载器
        将图片展平为 [3072] 的字节流 (0-255)
        """
        root = "./data_cifar"
        train = (split == "train")
        
        print(f"🖼️ [Vision] 正在加载 CIFAR-10 ({split}) ...")
        
        # 标准预处理：转为 Tensor (0-1)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=download, 
            transform=transform
        )
        
        print(f"   样本数: {len(self.dataset)}")
        print(f"   单样本长度: {SEQ_LEN} 字节 (32x32 RGB)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img: [3, 32, 32] float 0.0-1.0
        img, label = self.dataset[idx]
        
        # 1. 转换为 0-255 整数
        img = (img * 255).long()
        
        # 2. 调整维度顺序: [3, 32, 32] -> [32, 32, 3] (H, W, C)
        # 这样符合像素的物理排列顺序
        img = img.permute(1, 2, 0)
        
        # 3. 展平为一维字节流 [3072]
        byte_stream = img.reshape(-1)
        
        return byte_stream

def get_vision_dataloader(split="train", batch_size=64):
    dataset = H2QCIFAR10(split=split)
    # num_workers=0 避免多进程在 Mac 上的一些兼容性问题
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=0)

if __name__ == "__main__":
    # 测试下载和加载
    loader = get_vision_dataloader(batch_size=1)
    data = next(iter(loader))
    print(f"Data Shape: {data.shape}") # 应该是 [1, 3072]
    print(f"Sample Values: {data[0, :10]}") # 前10个像素值