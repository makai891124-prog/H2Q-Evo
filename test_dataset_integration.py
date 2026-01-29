#!/usr/bin/env python3
"""
数据集集成测试脚本
测试自动下载的数据集是否可以被AGI训练系统正确加载和使用
"""

import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

def test_cifar10_loading():
    """测试CIFAR-10数据集加载"""
    print("🧪 测试CIFAR-10数据集加载...")
    try:
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加载CIFAR-10训练集
        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,  # 已经下载了
            transform=transform
        )

        # 创建数据加载器
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )

        # 获取一个批次的数据
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        print(f"✅ CIFAR-10加载成功!")
        print(f"   批次大小: {images.shape}")
        print(f"   标签: {labels}")
        print(f"   数据类型: {images.dtype}")
        print(f"   数据范围: [{images.min():.3f}, {images.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ CIFAR-10加载失败: {e}")
        return False

def test_cifar100_loading():
    """测试CIFAR-100数据集加载"""
    print("\n🧪 测试CIFAR-100数据集加载...")
    try:
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加载CIFAR-100训练集
        trainset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,  # 已经下载了
            transform=transform
        )

        # 创建数据加载器
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )

        # 获取一个批次的数据
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        print(f"✅ CIFAR-100加载成功!")
        print(f"   批次大小: {images.shape}")
        print(f"   标签: {labels}")
        print(f"   数据类型: {images.dtype}")
        print(f"   数据范围: [{images.min():.3f}, {images.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ CIFAR-100加载失败: {e}")
        return False

def test_ucf101_structure():
    """测试UCF101数据集结构"""
    print("\n🧪 测试UCF101数据集结构...")
    try:
        ucf101_path = Path('./data/ucf101/UCF-101/UCF-101')

        if not ucf101_path.exists():
            print("❌ UCF101路径不存在")
            return False

        # 统计视频文件数量
        video_files = list(ucf101_path.rglob('*.avi'))
        print(f"✅ UCF101结构检查成功!")
        print(f"   视频文件总数: {len(video_files)}")

        # 显示前几个类别
        categories = [d.name for d in ucf101_path.iterdir() if d.is_dir()]
        print(f"   类别数量: {len(categories)}")
        print(f"   示例类别: {categories[:5]}")

        return True

    except Exception as e:
        print(f"❌ UCF101结构检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎬 AGI训练系统数据集集成测试")
    print("=" * 50)

    results = []

    # 测试各个数据集
    results.append(test_cifar10_loading())
    results.append(test_cifar100_loading())
    results.append(test_ucf101_structure())

    # 总结结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")

    passed = sum(results)
    total = len(results)

    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 所有数据集集成测试通过！AGI训练系统可以正常使用这些数据集。")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查数据集完整性。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)