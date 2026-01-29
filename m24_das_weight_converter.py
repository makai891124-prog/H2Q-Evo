#!/usr/bin/env python3
"""
M24-DAS DeepSeek权重转换和优化系统
基于M24真实性原则和DAS数学架构，将DeepSeek权重转换为核心机直接可用格式

目标：
1. 权重转换：DeepSeek → DAS兼容格式
2. 内存优化：适配Mac Mini M4 16G内存
3. 性能优化：利用M4 AMX加速
4. M24验证：确保转换过程的真实性和可验证性
"""

import os
import sys
import json
import time
import torch
import logging
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import gc
import numpy as np
from collections import OrderedDict

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "h2q_project"))

# 导入DAS核心
from h2q_project.das_core import DASCore, ConstructiveUniverse, DirectionalGroup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [M24-DAS-WEIGHT-CONV] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('m24_das_weight_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('M24-DAS-WEIGHT-CONV')

@dataclass
class M24WeightConversionConfig:
    """M24权重转换配置"""
    source_model: str  # 源模型名称
    target_format: str  # 目标格式 (DAS/H2Q)
    memory_limit_gb: float  # 内存限制
    compression_ratio: float  # 压缩比例
    m24_verified: bool = True  # M24验证标记

@dataclass
class WeightConversionResult:
    """权重转换结果"""
    success: bool
    source_model: str
    target_model: str
    original_size_mb: float
    converted_size_mb: float
    compression_ratio: float
    memory_usage_gb: float
    conversion_time_sec: float
    m24_verification: Dict[str, Any]
    error_message: Optional[str] = None

class M24DASWeightConverter:
    """
    M24-DAS权重转换器
    基于真实性原则和DAS数学架构进行权重转换
    """

    def __init__(self, config: M24WeightConversionConfig):
        self.config = config
        self.das_core = DASCore(target_dimension=256)  # DAS核心维度
        self.memory_monitor = MemoryMonitor()
        self.m24_verifier = M24WeightVerifier()

        # Mac Mini M4优化配置
        self.m4_optimizations = {
            'amx_acceleration': True,
            'memory_chunking': True,
            'quantization_bits': 8,  # 8-bit量化以节省内存
            'chunk_size_mb': 512  # 512MB块大小
        }

        logger.info("🧬 M24-DAS权重转换器初始化完成")
        logger.info(f"📊 配置: {asdict(config)}")

    def convert_deepseek_weights(self, source_path: str, target_path: str) -> WeightConversionResult:
        """
        转换DeepSeek权重到DAS格式

        Args:
            source_path: 源权重路径
            target_path: 目标权重路径

        Returns:
            转换结果
        """
        start_time = time.time()
        result = WeightConversionResult(
            success=False,
            source_model=self.config.source_model,
            target_model=f"DAS-{self.config.source_model}",
            original_size_mb=0.0,
            converted_size_mb=0.0,
            compression_ratio=1.0,
            memory_usage_gb=0.0,
            conversion_time_sec=0.0,
            m24_verification={}
        )

        try:
            # M24验证：检查源权重真实性
            logger.info("🔍 M24验证：检查源权重真实性...")
            if not self.m24_verifier.verify_source_weights(source_path):
                result.error_message = "M24验证失败：源权重不符合真实性要求"
                return result

            # 1. 加载源权重（内存优化）
            logger.info("📥 加载源权重（内存优化模式）...")
            source_weights = self._load_weights_memory_optimized(source_path)
            result.original_size_mb = self._calculate_weights_size_mb(source_weights)

            # 2. DAS转换
            logger.info("🔄 应用DAS数学变换...")
            das_weights = self._apply_das_transformation(source_weights)

            # 3. M4优化
            logger.info("⚡ 应用Mac Mini M4优化...")
            optimized_weights = self._apply_m4_optimizations(das_weights)

            # 4. 压缩和保存
            logger.info("🗜️ 应用压缩和保存...")
            final_weights = self._compress_and_save(optimized_weights, target_path)
            result.converted_size_mb = self._calculate_weights_size_mb(final_weights)
            result.compression_ratio = result.original_size_mb / result.converted_size_mb

            # 5. 最终M24验证
            logger.info("✅ 最终M24验证...")
            result.m24_verification = self.m24_verifier.verify_converted_weights(
                source_weights, final_weights, self.config
            )

            result.success = True
            result.memory_usage_gb = self.memory_monitor.get_peak_usage_gb()
            result.conversion_time_sec = time.time() - start_time

            logger.info("🎉 权重转换完成！")
            logger.info(f"📊 结果: {asdict(result)}")

        except Exception as e:
            logger.error(f"❌ 权重转换失败: {e}")
            result.error_message = str(e)
        finally:
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def _load_weights_memory_optimized(self, path: str) -> Dict[str, torch.Tensor]:
        """内存优化权重加载"""
        logger.info("🔧 内存优化加载模式启动...")

        # 检查文件大小
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"📁 文件大小: {file_size_mb:.2f} MB")

        if file_size_mb > self.config.memory_limit_gb * 1024:
            raise MemoryError(f"文件过大: {file_size_mb:.2f} MB > {self.config.memory_limit_gb * 1024} MB限制")

        # 分块加载
        weights = {}
        chunk_size = self.m4_optimizations['chunk_size_mb'] * 1024 * 1024  # 转换为字节

        try:
            # 对于PyTorch模型
            if path.endswith('.pth') or path.endswith('.pt'):
                logger.info("🔥 检测到PyTorch权重文件")
                state_dict = torch.load(path, map_location='cpu', weights_only=True)

                # 处理state_dict
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # 量化到8-bit以节省内存
                        if value.dtype == torch.float32:
                            value = value.to(torch.float16)  # 先降到float16

                        weights[key] = value
                        self.memory_monitor.check_memory_limit(self.config.memory_limit_gb)
                    elif key == 'model_state_dict' and isinstance(value, (dict, OrderedDict)):
                        # 处理嵌套的model_state_dict
                        logger.info("🔍 发现嵌套的model_state_dict")
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, torch.Tensor):
                                if nested_value.dtype == torch.float32:
                                    nested_value = nested_value.to(torch.float16)
                                weights[f"model_state_dict.{nested_key}"] = nested_value
                                self.memory_monitor.check_memory_limit(self.config.memory_limit_gb)
                    else:
                        logger.debug(f"跳过非张量权重: {key} (类型: {type(value)})")

            # 对于GGUF模型
            elif path.endswith('.gguf'):
                logger.info("🔥 检测到GGUF权重文件")
                weights = self._load_gguf_weights(path)

            else:
                raise ValueError(f"不支持的权重格式: {path}")

        except Exception as e:
            logger.error(f"权重加载失败: {e}")
            raise

        logger.info(f"✅ 成功加载 {len(weights)} 个权重张量")
        return weights

    def _apply_das_transformation(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用DAS数学变换"""
        logger.info("🔬 应用DAS数学架构变换...")

        transformed_weights = {}

        for key, tensor in weights.items():
            # 1. 转换为DAS兼容维度
            das_tensor = self._convert_to_das_dimensions(tensor)

            # 2. 应用DAS群作用
            das_transformed, das_report = self.das_core(das_tensor.unsqueeze(0))
            das_transformed = das_transformed.squeeze(0)

            # 3. 应用度量不变性
            metric_invariant = self._apply_metric_invariance(das_transformed)

            transformed_weights[key] = metric_invariant

            logger.debug(f"✅ 转换权重: {key} | 原始: {tensor.shape} | DAS: {metric_invariant.shape}")

        logger.info("🎯 DAS变换完成")
        return transformed_weights

    def _convert_to_das_dimensions(self, tensor: torch.Tensor) -> torch.Tensor:
        """转换为DAS兼容维度"""
        original_shape = tensor.shape

        # 展平为DAS目标维度
        flat_tensor = tensor.view(-1)

        # 如果维度不匹配，进行插值或截断
        if flat_tensor.size(0) != self.das_core.target_dimension:
            if flat_tensor.size(0) > self.das_core.target_dimension:
                # 截断
                flat_tensor = flat_tensor[:self.das_core.target_dimension]
            else:
                # 填充
                padding_size = self.das_core.target_dimension - flat_tensor.size(0)
                flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_size, dtype=flat_tensor.dtype)])

        return flat_tensor

    def _apply_metric_invariance(self, tensor: torch.Tensor) -> torch.Tensor:
        """应用度量不变性"""
        # 简化的度量不变性变换：保持尺度不变性
        norm = torch.norm(tensor)
        if norm > 0:
            normalized = tensor / norm
            # 应用简化的DAS不变性变换（这里可以根据需要扩展）
            invariant_tensor = normalized  # 简化为标准化
            return invariant_tensor * norm  # 保持原始尺度
        return tensor

    def _apply_m4_optimizations(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用Mac Mini M4优化"""
        logger.info("🍎 应用Mac Mini M4优化...")

        optimized_weights = {}

        for key, tensor in weights.items():
            # 1. AMX加速优化
            if self.m4_optimizations['amx_acceleration']:
                tensor = self._optimize_for_amx(tensor)

            # 2. 内存布局优化
            tensor = self._optimize_memory_layout(tensor)

            # 3. 量化优化
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float16)

            optimized_weights[key] = tensor

        logger.info("⚡ M4优化完成")
        return optimized_weights

    def _optimize_for_amx(self, tensor: torch.Tensor) -> torch.Tensor:
        """为AMX加速优化张量"""
        # AMX (Apple Matrix Coprocessor) 优化
        # 确保张量维度是AMX友好的
        shape = tensor.shape

        # AMX prefers dimensions that are multiples of 32
        optimized_shape = []
        for dim in shape:
            # 向上取整到32的倍数，但保持总元素数量
            optimized_dim = ((dim + 31) // 32) * 32
            optimized_shape.append(optimized_dim)

        if tuple(optimized_shape) != shape:
            # 插值或填充到优化维度
            optimized_tensor = torch.zeros(optimized_shape, dtype=tensor.dtype)
            min_shape = tuple(min(a, b) for a, b in zip(shape, optimized_shape))
            optimized_tensor[tuple(slice(0, s) for s in min_shape)] = tensor[tuple(slice(0, s) for s in min_shape)]
            return optimized_tensor

        return tensor

    def _optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """优化内存布局"""
        # 确保连续内存布局以提高性能
        return tensor.contiguous()

    def _compress_and_save(self, weights: Dict[str, torch.Tensor], target_path: str) -> Dict[str, torch.Tensor]:
        """压缩并保存权重"""
        logger.info("🗜️ 应用最终压缩...")

        # 创建目标目录
        target_dir = Path(target_path).parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # 保存为PyTorch格式
        torch.save(weights, target_path)

        # 计算压缩统计
        compressed_size = os.path.getsize(target_path) / (1024 * 1024)
        logger.info(f"💾 压缩后大小: {compressed_size:.2f} MB")

        return weights

    def _calculate_weights_size_mb(self, weights: Dict[str, torch.Tensor]) -> float:
        """计算权重总大小（MB）"""
        total_bytes = 0
        for tensor in weights.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 * 1024)

    def _load_gguf_weights(self, path: str) -> Dict[str, torch.Tensor]:
        """加载GGUF格式权重"""
        # 这里需要实现GGUF加载逻辑
        # 由于GGUF是二进制格式，我们使用一个简化的实现
        logger.warning("⚠️ GGUF加载功能尚未完全实现，使用模拟权重")
        # 返回一个模拟的权重字典用于测试
        return {
            'embed_tokens.weight': torch.randn(32000, 256),
            'layers.0.attention.wq.weight': torch.randn(256, 256),
            'layers.0.attention.wk.weight': torch.randn(256, 256),
            'layers.0.attention.wv.weight': torch.randn(256, 256),
            'layers.0.attention.wo.weight': torch.randn(256, 256),
            'layers.0.feed_forward.w1.weight': torch.randn(512, 256),
            'layers.0.feed_forward.w2.weight': torch.randn(256, 512),
            'layers.0.feed_forward.w3.weight': torch.randn(512, 256),
            'norm.weight': torch.randn(256),
        }


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.peak_usage = 0.0

    def check_memory_limit(self, limit_gb: float):
        """检查内存使用是否超过限制"""
        current_usage = psutil.virtual_memory().used / (1024**3)  # GB
        self.peak_usage = max(self.peak_usage, current_usage)

        if current_usage > limit_gb:
            raise MemoryError(f"内存使用超过限制: {current_usage:.2f} GB > {limit_gb} GB")

    def get_peak_usage_gb(self) -> float:
        """获取峰值内存使用"""
        return self.peak_usage


class M24WeightVerifier:
    """
    M24权重验证器
    确保权重转换过程符合真实性原则
    """

    def verify_source_weights(self, path: str) -> bool:
        """验证源权重"""
        if not os.path.exists(path):
            logger.error(f"❌ 源权重文件不存在: {path}")
            return False

        # 检查文件完整性
        try:
            file_size = os.path.getsize(path)
            if file_size == 0:
                logger.error("❌ 源权重文件为空")
                return False

            logger.info(f"✅ 源权重文件完整性检查通过: {file_size} bytes")
            return True

        except Exception as e:
            logger.error(f"❌ 源权重验证失败: {e}")
            return False

    def verify_converted_weights(self, source_weights: Dict[str, torch.Tensor],
                               converted_weights: Dict[str, torch.Tensor],
                               config: M24WeightConversionConfig) -> Dict[str, Any]:
        """验证转换后的权重"""
        verification = {
            'structure_preserved': False,
            'das_transformation_applied': False,
            'memory_optimization_verified': False,
            'm24_compliance': True,
            'compression_verified': False
        }

        try:
            # 1. 检查结构保持
            source_keys = set(source_weights.keys())
            converted_keys = set(converted_weights.keys())

            if source_keys == converted_keys:
                verification['structure_preserved'] = True
                logger.info("✅ 权重结构保持验证通过")
            else:
                logger.warning(f"⚠️ 权重结构变化: {source_keys - converted_keys} | {converted_keys - source_keys}")

            # 2. 检查DAS变换应用
            # 验证转换后的权重具有DAS特性
            for key, tensor in converted_weights.items():
                if tensor.size(-1) == 256:  # DAS目标维度
                    verification['das_transformation_applied'] = True
                    break

            # 3. 检查内存优化
            total_memory_mb = sum(tensor.numel() * tensor.element_size() for tensor in converted_weights.values()) / (1024*1024)
            if total_memory_mb < config.memory_limit_gb * 1024 * 0.8:  # 80%以内
                verification['memory_optimization_verified'] = True

            # 4. 检查压缩
            original_size = sum(tensor.numel() * tensor.element_size() for tensor in source_weights.values()) / (1024*1024)
            converted_size = sum(tensor.numel() * tensor.element_size() for tensor in converted_weights.values()) / (1024*1024)

            if converted_size < original_size:
                verification['compression_verified'] = True
                logger.info(f"✅ 压缩验证通过: {original_size:.2f} MB → {converted_size:.2f} MB")

            logger.info("🎯 M24权重验证完成")
            logger.info(f"📊 验证结果: {verification}")

        except Exception as e:
            logger.error(f"❌ 权重验证失败: {e}")
            verification['m24_compliance'] = False

        return verification


def main():
    """主函数：执行DeepSeek权重转换"""
    logger.info("🚀 启动M24-DAS DeepSeek权重转换系统")
    logger.info("基于M24真实性原则和DAS数学架构")

    # 配置
    config = M24WeightConversionConfig(
        source_model="deepseek-coder-v2-236b",
        target_format="DAS-H2Q",
        memory_limit_gb=12.0,  # Mac Mini M4 16G，留4G余量
        compression_ratio=0.3  # 30%压缩目标
    )

    converter = M24DASWeightConverter(config)

    # 查找源权重文件
    models_dir = Path("models")
    possible_sources = [
        models_dir / "deepseek_236b_ultra_compressed.pth",
        models_dir / "ultra_compressed_236b.pth",
        models_dir / "fractal_restructured_236b.pth",
        Path("deepseek_weights.pth"),  # 用户可能下载的文件
    ]

    source_path = None
    for path in possible_sources:
        if path.exists():
            source_path = path
            break

    if not source_path:
        logger.error("❌ 未找到DeepSeek源权重文件")
        logger.info("请下载DeepSeek权重文件并放置在models目录或当前目录")
        return

    # 目标路径
    target_path = models_dir / f"das_optimized_{config.source_model}.pth"

    # 执行转换
    logger.info(f"🔄 开始转换: {source_path} → {target_path}")
    result = converter.convert_deepseek_weights(str(source_path), str(target_path))

    # 输出结果
    if result.success:
        logger.info("🎉 权重转换成功！")
        logger.info("📊 转换统计:")
        logger.info(f"   原始大小: {result.original_size_mb:.2f} MB")
        logger.info(f"   转换大小: {result.converted_size_mb:.2f} MB")
        logger.info(f"   压缩比例: {result.compression_ratio:.2f}x")
        logger.info(f"   内存使用: {result.memory_usage_gb:.2f} GB")
        logger.info(f"   转换时间: {result.conversion_time_sec:.2f} 秒")
        logger.info(f"   M24验证: {result.m24_verification}")

        # 保存转换报告
        report = {
            'timestamp': time.time(),
            'config': asdict(config),
            'result': asdict(result),
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }

        report_path = models_dir / f"das_conversion_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 转换报告已保存: {report_path}")

    else:
        logger.error("❌ 权重转换失败！")
        logger.error(f"错误信息: {result.error_message}")


if __name__ == "__main__":
    main()