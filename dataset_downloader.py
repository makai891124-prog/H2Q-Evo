#!/usr/bin/env python3
"""
自动数据集下载器 - 支持ImageNet、COCO、Kinetics等视觉数据集
"""

import os
import sys
import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
import json

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

class DatasetDownloader:
    """自动数据集下载器"""

    def __init__(self, datasets_path: str = './datasets'):
        self.datasets_path = Path(datasets_path)
        self.datasets_path.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 数据集配置
        self.dataset_configs = {
            'imagenet': {
                'name': 'ImageNet',
                'description': 'Large-scale image classification dataset',
                'urls': [
                    'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
                    'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
                    'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test.tar'
                ],
                'size': '150GB',
                'requires_login': True,
                'note': 'Requires ImageNet account and manual download'
            },
            'coco': {
                'name': 'COCO Dataset',
                'description': 'Common Objects in Context dataset',
                'urls': {
                    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
                    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
                    'test2017': 'http://images.cocodataset.org/zips/test2017.zip',
                    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                },
                'size': '25GB',
                'requires_login': False
            },
            'kinetics': {
                'name': 'Kinetics-400',
                'description': 'Large-scale video action recognition dataset',
                'urls': [
                    'https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz'
                ],
                'size': '450GB',
                'requires_login': False,
                'note': 'Very large dataset, consider downloading subsets'
            },
            'ucf101': {
                'name': 'UCF101',
                'description': 'Action recognition dataset',
                'urls': [
                    'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar',
                    'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
                ],
                'size': '7GB',
                'requires_login': False,
                'note': 'Already available locally'
            },
            'cifar10': {
                'name': 'CIFAR-10',
                'description': 'Small image classification dataset for testing',
                'urls': [
                    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                ],
                'size': '170MB',
                'requires_login': False
            },
            'cifar100': {
                'name': 'CIFAR-100',
                'description': 'Fine-grained image classification dataset',
                'urls': [
                    'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
                ],
                'size': '170MB',
                'requires_login': False
            }
        }

    def list_available_datasets(self) -> Dict[str, Dict]:
        """列出所有可用数据集"""
        return self.dataset_configs

    def check_dataset_status(self, dataset_name: str) -> Dict[str, bool]:
        """检查数据集下载状态"""
        if dataset_name not in self.dataset_configs:
            return {'available': False, 'error': 'Dataset not found'}

        dataset_path = self.datasets_path / dataset_name
        config = self.dataset_configs[dataset_name]

        # 检查本地是否已有数据
        if dataset_name == 'ucf101':
            # 特殊处理UCF101，因为已经在本地
            ucf101_path = Path('/Users/imymm/H2Q-Evo/data/ucf101/UCF-101/UCF-101')
            has_data = ucf101_path.exists() and any(ucf101_path.rglob('*.avi'))
            return {
                'available': has_data,
                'local_path': str(ucf101_path),
                'size': config['size'],
                'note': 'Already available locally'
            }

        # 检查标准数据集
        has_data = dataset_path.exists() and any(dataset_path.rglob('*'))
        return {
            'available': has_data,
            'local_path': str(dataset_path),
            'size': config['size']
        }

    def download_dataset(self, dataset_name: str, max_workers: int = 4) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.dataset_configs:
            self.logger.error(f"Dataset {dataset_name} not found")
            return False

        config = self.dataset_configs[dataset_name]
        dataset_path = self.datasets_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"开始下载数据集: {config['name']} ({config['size']})")

        # 特殊处理UCF101（已在本地）
        if dataset_name == 'ucf101':
            self.logger.info("UCF101数据集已在本地可用")
            return True

        # 特殊处理ImageNet（需要手动下载）
        if dataset_name == 'imagenet':
            self.logger.warning("ImageNet需要手动下载，请访问: https://image-net.org/download.php")
            return False

        # 下载其他数据集
        try:
            if isinstance(config['urls'], dict):
                # 多个文件的情况（如COCO）
                return self._download_multiple_files(dataset_name, config['urls'], dataset_path, max_workers)
            else:
                # 单个文件的情况
                return self._download_single_file(dataset_name, config['urls'][0], dataset_path)
        except Exception as e:
            self.logger.error(f"下载数据集 {dataset_name} 失败: {e}")
            return False

    def _download_single_file(self, dataset_name: str, url: str, dest_path: Path) -> bool:
        """下载单个文件"""
        try:
            filename = url.split('/')[-1]
            file_path = dest_path / filename

            # 检查文件是否已存在
            if file_path.exists():
                self.logger.info(f"文件已存在: {filename}")
                return self._extract_file(file_path, dest_path)

            self.logger.info(f"下载文件: {filename}")

            # 下载文件
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # 显示下载进度
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            self.logger.info(".1f")

            self.logger.info(f"文件下载完成: {filename}")

            # 解压文件
            return self._extract_file(file_path, dest_path)

        except Exception as e:
            self.logger.error(f"下载文件失败: {e}")
            return False

    def _download_multiple_files(self, dataset_name: str, urls: Dict[str, str], dest_path: Path, max_workers: int) -> bool:
        """并行下载多个文件"""
        self.logger.info(f"并行下载 {len(urls)} 个文件")

        success_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for name, url in urls.items():
                future = executor.submit(self._download_single_file, f"{dataset_name}_{name}", url, dest_path)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        self.logger.info(f"文件 {name} 下载成功")
                    else:
                        self.logger.error(f"文件 {name} 下载失败")
                except Exception as e:
                    self.logger.error(f"文件 {name} 下载异常: {e}")

        return success_count == len(urls)

    def _extract_file(self, file_path: Path, dest_path: Path) -> bool:
        """解压文件"""
        try:
            self.logger.info(f"解压文件: {file_path.name}")

            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_path)
            elif file_path.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(dest_path)
            elif file_path.suffix == '.rar':
                # 使用unrar命令（需要安装unrar）
                try:
                    subprocess.run(['unrar', 'x', str(file_path), str(dest_path)], check=True)
                except subprocess.CalledProcessError:
                    self.logger.error("需要安装unrar来解压RAR文件")
                    return False
            else:
                self.logger.warning(f"不支持的文件格式: {file_path.suffix}")
                return False

            self.logger.info(f"文件解压完成: {file_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"解压文件失败: {e}")
            return False

    def verify_dataset_integrity(self, dataset_name: str) -> bool:
        """验证数据集完整性"""
        if dataset_name not in self.dataset_configs:
            return False

        dataset_path = self.datasets_path / dataset_name

        # 特殊处理已在data目录中的数据集
        if dataset_name in ['cifar10', 'cifar100']:
            if dataset_name == 'cifar10':
                dataset_path = Path('./data/cifar-10-batches-py')
            elif dataset_name == 'cifar100':
                dataset_path = Path('./data/cifar-100-python')

        if dataset_name == 'ucf101':
            dataset_path = Path('/Users/imymm/H2Q-Evo/data/ucf101/UCF-101/UCF-101')

        # 检查基本文件结构
        if not dataset_path.exists():
            return False

        # 根据数据集类型进行特定检查
        if dataset_name == 'coco':
            # 检查COCO数据集结构
            required_dirs = ['train2017', 'val2017', 'annotations']
            return all((dataset_path / d).exists() for d in required_dirs)

        elif dataset_name == 'cifar10':
            # 检查CIFAR-10数据文件
            return any(dataset_path.rglob('data_batch_*'))

        elif dataset_name == 'cifar100':
            # 检查CIFAR-100数据文件
            return (dataset_path / 'train').exists() and (dataset_path / 'test').exists()

        elif dataset_name == 'ucf101':
            # 检查UCF101视频文件
            return any(dataset_path.rglob('*.avi'))

        # 默认检查：目录不为空
        return any(dataset_path.rglob('*'))

def main():
    """主函数"""
    print("🎬 自动数据集下载器")
    print("=" * 50)

    downloader = DatasetDownloader()

    # 显示可用数据集
    print("\n📋 可用数据集:")
    datasets = downloader.list_available_datasets()
    for name, config in datasets.items():
        status = downloader.check_dataset_status(name)
        available = "✅" if status['available'] else "❌"
        print(f"  {available} {name}: {config['name']} ({config['size']})")

    # 下载小数据集进行测试
    print("\n⬇️  下载测试数据集...")

    # 下载CIFAR-10（小数据集用于测试）
    if not downloader.check_dataset_status('cifar10')['available']:
        print("下载CIFAR-10数据集...")
        success = downloader.download_dataset('cifar10')
        if success:
            print("✅ CIFAR-10下载成功")
        else:
            print("❌ CIFAR-10下载失败")

    # 下载CIFAR-100
    if not downloader.check_dataset_status('cifar100')['available']:
        print("下载CIFAR-100数据集...")
        success = downloader.download_dataset('cifar100')
        if success:
            print("✅ CIFAR-100下载成功")
        else:
            print("❌ CIFAR-100下载失败")

    print("\n📝 注意:")
    print("- ImageNet需要手动下载（访问 https://image-net.org/download.php）")
    print("- COCO数据集较大，下载可能需要时间")
    print("- Kinetics数据集非常大（450GB），建议按需下载")

if __name__ == "__main__":
    main()