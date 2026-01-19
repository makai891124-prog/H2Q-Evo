# tools/spacetime_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class H2QSpacetimeDataset(Dataset):
    def __init__(self, split="train"):
        """
        将 CIFAR-10 图片映射为 YCbCr 四元数时空体
        """
        root = "./data_cifar"
        train = (split == "train")
        
        print(f"🌌 [Spacetime] 正在加载 CIFAR-10 ({split}) ...")
        
        # [关键] 我们需要 PIL Image 格式来进行 YCbCr 转换
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None # 不做 ToTensor
        )
        
        print(f"   样本数: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img: PIL Image
        img, _ = self.dataset[idx]
        
        # 1. 转换为 YCbCr 模式
        img_ycbcr = img.convert('YCbCr')
        
        # 2. 分离通道
        y, cb, cr = img_ycbcr.split()
        
        # 3. 转为 Tensor 并归一化到 [-1, 1]
        to_tensor = transforms.ToTensor()
        y_t = (to_tensor(y) * 2) - 1
        cb_t = (to_tensor(cb) * 2) - 1
        cr_t = (to_tensor(cr) * 2) - 1
        
        # 4. [核心] 构造四元数
        # w (时间/颜色): Cb
        # x (空间): Cr
        # y (空间): Y (亮度)
        # z (空间): 0 (预留给视频的时间轴)
        # 这是一个新的映射，将颜色和亮度分离
        
        # 简化版：w=Cb, x=Cr, y=Y, z=0
        # 或者更符合你的 (x,y,z,c) -> (x,y,Y, Cb+Cr)
        # 我们这里用一个更直接的映射：
        # w=0, x=Y, y=Cb, z=Cr
        
        w = torch.zeros_like(y_t)
        
        # 拼接: [4, 32, 32]
        q_img = torch.cat([w, y_t, cb_t, cr_t], dim=0)
        
        # 转换维度: [32, 32, 4]
        q_img = q_img.permute(1, 2, 0)
        
        return q_img

def get_spacetime_dataloader(split="train", batch_size=64):
    dataset = H2QSpacetimeDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))