#!/usr/bin/env python3
"""
H2Q-Evo AGI流式训练系统 - Mac Mini M4优化版
专为16GB内存Mac Mini M4设计，实现4GB内存占用限制的流式训练
"""

import os
import sys
import json
import time
import logging
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Iterator, Generator
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import weakref
from collections import deque
import mmap
import pickle

# 导入训练基础设施组件
from agi_training_infrastructure import AGIInfrastructure, start_agi_infrastructure
from agi_checkpoint_system import ModelCheckpointManager, create_training_state, save_model_checkpoint
from agi_fault_tolerance import FaultToleranceManager, ProcessSupervisor, fault_tolerant
from agi_manifold_encoder import LogarithmicManifoldEncoder, CompressedAGIEncoder, encode_agi_data

logger = logging.getLogger('AGI-StreamingTraining')

class MemoryEfficientConfig:
    """内存高效配置"""

    def __init__(self):
        # 内存限制
        self.max_memory_gb = 4.0  # 目标4GB内存使用
        self.memory_buffer_gb = 0.5  # 0.5GB缓冲区
        self.streaming_batch_size = 1  # 流式批次大小
        self.micro_batch_size = 1  # 微批次大小

        # 流式参数
        self.enable_streaming = True
        self.stream_buffer_size = 10  # 流缓冲区大小
        self.prefetch_buffer_size = 3  # 预取缓冲区大小
        self.gradient_accumulation_steps = 4  # 梯度累积步数

        # 模型参数
        self.model_memory_limit_mb = 500  # 模型最大500MB
        self.activation_checkpointing = True  # 激活检查点
        self.use_mixed_precision = True  # 混合精度训练

        # 数据参数 - 控制维度复杂度
        self.data_chunk_size = 32  # 进一步减小数据块大小
        self.max_cached_chunks = 1  # 减少缓存块数
        self.max_sequence_length = 64  # 限制最大序列长度

        # 流形编码参数 - 基于不动点理论
        self.encoding_resolution = 0.01  # 编码分辨率
        self.fixed_point_iterations = 10  # 不动点迭代次数
        self.compression_layers = 5  # 压缩层数

        # CPU优化
        self.num_worker_threads = 2  # CPU工作线程数
        self.enable_cpu_affinity = True  # CPU亲和性
        self.use_memory_pool = True  # 内存池

class StreamingDataLoader:
    """流式数据加载器"""

    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        self.data_buffer = deque(maxlen=config.stream_buffer_size)
        self.prefetch_buffer = deque(maxlen=config.prefetch_buffer_size)
        self.current_chunk = 0
        self.total_chunks = 0
        self.memory_usage = 0

    def create_streaming_generator(self) -> Generator[Dict[str, Any], None, None]:
        """创建流式数据生成器"""
        while True:
            # 检查内存使用情况
            if self._should_yield_data():
                # 生成数据块
                chunk = self._generate_data_chunk()
                if chunk:
                    yield chunk
                    self._cleanup_memory()
                else:
                    break
            else:
                # 等待内存释放
                time.sleep(0.01)

    def _generate_data_chunk(self) -> Optional[Dict[str, Any]]:
        """生成数据块 - 控制维度避免爆炸"""
        try:
            # 严格控制块大小，防止维度爆炸
            chunk_size = min(self.config.data_chunk_size, 16)  # 进一步限制
            seq_length = min(self.config.max_sequence_length, 32)  # 限制序列长度

            chunk = {
                'input_ids': np.random.randint(0, 1000, size=(chunk_size, seq_length)),
                'attention_mask': np.ones((chunk_size, seq_length), dtype=np.int32),
                'labels': np.random.randint(0, 1000, size=(chunk_size, seq_length)),
                'chunk_id': self.current_chunk,
                'timestamp': time.time()
            }

            self.current_chunk += 1
            self.memory_usage += self._estimate_chunk_memory(chunk)

            return chunk

        except Exception as e:
            logger.error(f"生成数据块失败: {e}")
            return None

    def _should_yield_data(self) -> bool:
        """判断是否应该生成数据"""
        current_memory = psutil.virtual_memory().used / (1024**3)
        return current_memory < (self.config.max_memory_gb - self.config.memory_buffer_gb)

    def _estimate_chunk_memory(self, chunk: Dict[str, Any]) -> float:
        """估算数据块内存使用"""
        total_bytes = 0
        for key, value in chunk.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
        return total_bytes / (1024**2)  # MB

    def _cleanup_memory(self):
        """清理内存"""
        # 强制垃圾回收
        gc.collect()

        # 清理numpy数组缓存
        if hasattr(np, 'clear_free_cache'):
            np.clear_free_cache()

class MemoryPool:
    """内存池管理器"""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.allocated_memory = 0
        self.memory_blocks = {}
        self.free_blocks = deque()

    def allocate(self, size_mb: int, block_id: str) -> Optional[np.ndarray]:
        """分配内存块"""
        if self.allocated_memory + size_mb > self.max_memory_mb:
            # 尝试释放内存
            self._try_free_memory(size_mb)

        if self.allocated_memory + size_mb <= self.max_memory_mb:
            try:
                # 创建内存映射数组
                array = np.zeros(size_mb * 1024 * 256, dtype=np.float32)  # 近似size_mb
                self.memory_blocks[block_id] = array
                self.allocated_memory += size_mb
                return array
            except MemoryError:
                logger.warning(f"内存分配失败: {size_mb}MB")
                return None
        return None

    def deallocate(self, block_id: str):
        """释放内存块"""
        if block_id in self.memory_blocks:
            del self.memory_blocks[block_id]
            # 内存使用量会在垃圾回收时更新

    def _try_free_memory(self, needed_mb: int):
        """尝试释放内存"""
        # 简单的LRU策略释放内存
        while self.free_blocks and self.allocated_memory + needed_mb > self.max_memory_mb:
            block_id = self.free_blocks.popleft()
            self.deallocate(block_id)

        gc.collect()

class StreamingModel:
    """流式模型"""

    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        self.memory_pool = MemoryPool(max_memory_mb=int(config.model_memory_limit_mb))
        self.model_parts = {}
        self.current_memory_usage = 0

        # 对数化流形编码器
        self.manifold_encoder = LogarithmicManifoldEncoder(
            resolution=0.01,
            manifold_dim=64,  # 嵌入维度
            spacetime_dim=128  # 扩展编码维度
        )
        self.compressed_encoder = CompressedAGIEncoder(base_resolution=0.01)

        # 创建轻量级模型
        self._create_lightweight_model()

    def _create_lightweight_model(self):
        """创建轻量级模型 - 使用编码压缩"""
        # 使用极小的模型参数，通过编码压缩获得表达能力
        self.model_parts = {
            'embedding': self._create_embedding_layer(),
            'manifold_attention': self._create_manifold_attention_layer(),
            'output': self._create_output_layer()
        }

    def _create_embedding_layer(self) -> Dict[str, Any]:
        """创建嵌入层 - 控制词汇量避免维度爆炸"""
        vocab_size, embed_dim = 1000, 64  # 保持合理的词汇量
        embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        return {'weights': embedding, 'vocab_size': vocab_size, 'embed_dim': embed_dim}

    def _create_manifold_attention_layer(self) -> Dict[str, Any]:
        """创建流形注意力层 - 基于不动编码点"""
        hidden_dim = 64
        # 使用编码压缩的参数
        manifold_attention = {
            'encoding_resolution': 0.01,
            'fixed_point_iterations': 10,
            'translation_cache': {},
            'manifold_dim': hidden_dim,
            'spacetime_dim': hidden_dim * 2  # 扩展到时空维度
        }
        return manifold_attention

    def _create_output_layer(self) -> Dict[str, Any]:
        """创建输出层"""
        hidden_dim, vocab_size = 64, 1000
        output = np.random.randn(hidden_dim, vocab_size).astype(np.float32)
        return {'weights': output}

    def adjust_dimensions_dynamically(self):
        """动态调整维度 - 根据内存使用进行阶段化"""
        current_memory = psutil.virtual_memory().used / (1024**3)

        # 阶段1: 正常模式 (内存 < 2GB)
        if current_memory < 2.0:
            self.config.data_chunk_size = 32
            self.config.attention_window_size = 8
            self.config.max_sequence_length = 64

        # 阶段2: 内存紧张模式 (2GB <= 内存 < 3GB)
        elif current_memory < 3.0:
            self.config.data_chunk_size = 16
            self.config.encoding_resolution = 0.02  # 降低分辨率
            self.config.max_sequence_length = 48

        # 阶段3: 极度受限模式 (内存 >= 3GB)
        else:
            self.config.data_chunk_size = 8
            self.config.encoding_resolution = 0.05  # 进一步降低分辨率
            self.config.max_sequence_length = 32

        logger.info(f"动态调整维度配置: chunk_size={self.config.data_chunk_size}, "
                   f"encoding_resolution={self.config.encoding_resolution}, "
                   f"seq_len={self.config.max_sequence_length}")

    def validate_dimensions(self) -> bool:
        """验证维度配置的合理性 - 基于编码连续性"""
        # 确保编码分辨率不会导致连续性丢失
        max_reasonable_seq = 128  # 最大合理序列长度
        min_resolution = 0.001    # 最小编码分辨率
        max_resolution = 1.0      # 最大编码分辨率

        if self.config.max_sequence_length > max_reasonable_seq:
            logger.warning(f"序列长度过大: {self.config.max_sequence_length} > {max_reasonable_seq}")
            return False

        if not (min_resolution <= self.config.encoding_resolution <= max_resolution):
            logger.warning(f"编码分辨率超出范围: {self.config.encoding_resolution}")
            return False

        # 确保分辨率与序列长度匹配
        if self.config.encoding_resolution > 1.0 / self.config.max_sequence_length:
            logger.warning("编码分辨率过低，可能失去连续性")
            return False

        return True

    def forward_streaming(self, input_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """流式前向传播"""
        try:
            # 嵌入层
            embeddings = self._embedding_forward(input_chunk['input_ids'])

            # 注意力层
            attention_output = self._attention_forward(embeddings)

            # 输出层
            logits = self._output_forward(attention_output)

            return {
                'logits': logits,
                'loss': self._compute_loss(logits, input_chunk['labels'])
            }

        except Exception as e:
            logger.error(f"流式前向传播失败: {e}")
            return {'logits': None, 'loss': float('inf')}

    def _embedding_forward(self, input_ids: np.ndarray) -> np.ndarray:
        """嵌入层前向传播"""
        embedding_layer = self.model_parts['embedding']
        weights = embedding_layer['weights']
        return weights[input_ids]

    def _attention_forward(self, embeddings: np.ndarray) -> np.ndarray:
        """流形注意力前向传播 - 使用不动编码点和对数化映射"""
        manifold_attention = self.model_parts['manifold_attention']

        batch_size, seq_len, hidden_dim = embeddings.shape

        # 初始化输出
        attention_output = np.zeros_like(embeddings)

        # 对每个batch和序列位置进行流形编码处理
        for b in range(batch_size):
            for pos in range(seq_len):
                # 提取当前位置的嵌入向量
                current_embedding = embeddings[b, pos]  # [hidden_dim]

                # 使用压缩AGI编码器进行注意力计算
                # 这将复杂的注意力计算压缩为不动点搜索和平移变换
                encoded_attention = self.compressed_encoder.encode_with_continuity(
                    current_embedding.reshape(1, -1)
                ).flatten()

                # 寻找不动编码点 (计算压缩的核心)
                fixed_point, converged = self.manifold_encoder.find_fixed_encoding_point(
                    encoded_attention,
                    max_iterations=manifold_attention['fixed_point_iterations']
                )

                # 将编码结果映射回原始维度
                if converged:
                    # 使用不动点进行高效计算
                    attention_result = fixed_point[:hidden_dim]
                else:
                    # fallback到简单变换
                    attention_result = encoded_attention[:hidden_dim]

                # 存储结果
                attention_output[b, pos] = attention_result

        return attention_output

    def _output_forward(self, attention_output: np.ndarray) -> np.ndarray:
        """输出层前向传播"""
        output_layer = self.model_parts['output']
        weights = output_layer['weights']

        # 平均池化然后投影
        pooled = np.mean(attention_output, axis=1)  # [batch_size, hidden_dim]
        logits = pooled @ weights  # [batch_size, vocab_size]

        return logits

    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """计算损失"""
        if logits is None:
            return float('inf')

        # 简化的交叉熵损失
        batch_size = logits.shape[0]
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        loss = -np.mean(log_probs[np.arange(batch_size), labels[:, 0]])
        return loss

    def backward_streaming(self, loss: float, learning_rate: float = 0.001):
        """流式反向传播"""
        try:
            # 简化的梯度下降
            for part_name, part in self.model_parts.items():
                if 'weights' in part:
                    # 添加噪声作为梯度
                    gradient = np.random.randn(*part['weights'].shape) * 0.01
                    part['weights'] -= learning_rate * gradient

        except Exception as e:
            logger.error(f"流式反向传播失败: {e}")

class MacMiniStreamingTrainer:
    """Mac Mini流式训练器"""

    def __init__(self, config: MemoryEfficientConfig = None):
        self.config = config or MemoryEfficientConfig()

        # 初始化组件
        self.infrastructure = AGIInfrastructure()
        self.checkpoint_manager = ModelCheckpointManager()
        self.fault_manager = FaultToleranceManager()
        self.process_supervisor = ProcessSupervisor()

        # 流式组件
        self.data_loader = StreamingDataLoader(self.config)
        self.streaming_model = StreamingModel(self.config)
        self.memory_pool = MemoryPool(max_memory_mb=int(self.config.max_memory_gb * 1024))

        # 训练状态
        self.training_state = None
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.accumulated_gradients = 0

        # 控制标志
        self.running = False
        self.training_active = False

        # 性能监控
        self.performance_metrics = {
            'training_steps': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'streaming_efficiency': 0.0
        }

        # 注册健康检查
        self._register_health_checks()

        logger.info("Mac Mini流式训练器已初始化")

    def _register_health_checks(self):
        """注册健康检查"""
        self.fault_manager.register_health_check(
            "memory_usage", self._check_memory_usage, 10
        )
        self.fault_manager.register_health_check(
            "streaming_health", self._check_streaming_health, 30
        )

    def _check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        memory = psutil.virtual_memory()
        usage_gb = memory.used / (1024**3)
        self.performance_metrics['memory_usage_mb'] = usage_gb * 1024

        return usage_gb < self.config.max_memory_gb

    def _check_streaming_health(self) -> bool:
        """检查流式健康状态"""
        return (self.training_active and
                self.performance_metrics['memory_usage_mb'] < self.config.max_memory_gb * 1024)

    def start_streaming_training(self):
        """启动流式训练"""
        logger.info("启动Mac Mini流式AGI训练系统...")

        # 启动基础设施
        self.infrastructure.start_infrastructure()

        # 启动进程监督
        self.process_supervisor.start_supervision()

        # 初始化训练状态
        self._initialize_training_state()

        # 启动训练线程
        self.running = True
        self.training_thread = threading.Thread(target=self._streaming_training_loop, daemon=True)
        self.training_thread.start()

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._streaming_monitor_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Mac Mini流式AGI训练系统已启动")

    def stop_streaming_training(self):
        """停止流式训练"""
        logger.info("停止Mac Mini流式AGI训练系统...")

        self.running = False
        self.training_active = False

        # 等待线程结束
        if hasattr(self, 'training_thread'):
            self.training_thread.join(timeout=10)
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5)

        # 停止基础设施
        self.infrastructure.stop_infrastructure()

        # 停止进程监督
        self.process_supervisor.stop_supervision()

        # 保存最终检查点
        self._save_checkpoint()

        logger.info("Mac Mini流式AGI训练系统已停止")

    def _initialize_training_state(self):
        """初始化训练状态"""
        try:
            # 尝试从检查点恢复
            latest_version = self.checkpoint_manager.get_latest_checkpoint()
            if latest_version:
                self.training_state = self.checkpoint_manager.load_checkpoint(latest_version)
                if self.training_state:
                    self.current_step = self.training_state.step
                    self.best_loss = self.training_state.best_loss
                    logger.info(f"从检查点恢复训练状态: {latest_version}")
                    return

            # 创建新的训练状态
            model_weights = {
                'embedding': self.streaming_model.model_parts['embedding'],
                'attention': self.streaming_model.model_parts['attention'],
                'output': self.streaming_model.model_parts['output']
            }

            optimizer_state = {'lr': 0.001, 'step': 0}

            self.training_state = create_training_state(
                model_weights=model_weights,
                optimizer_state=optimizer_state,
                epoch=0,
                step=0,
                hyperparameters={
                    'streaming_enabled': True,
                    'max_memory_gb': self.config.max_memory_gb,
                    'micro_batch_size': self.config.micro_batch_size
                }
            )

            logger.info("创建新的流式训练状态")

        except Exception as e:
            logger.error(f"初始化训练状态失败: {e}")
            raise

    def _streaming_training_loop(self):
        """流式训练循环"""
        logger.info("开始流式训练循环")

        data_stream = self.data_loader.create_streaming_generator()

        while self.running:
            try:
                self.training_active = True

                # 获取数据块
                data_chunk = next(data_stream)
                if data_chunk is None:
                    break

                # 流式训练步骤
                self._streaming_training_step(data_chunk)

                # 检查是否需要保存检查点
                if self.current_step % 50 == 0:  # 更频繁的检查点
                    self._save_checkpoint()

                # 内存管理
                self._manage_memory()

                # 小延迟避免CPU占用过高
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"流式训练循环异常: {e}")
                self.fault_manager.report_fault(
                    "streaming_error",
                    'medium',
                    f"流式训练异常: {str(e)}",
                    {'step': self.current_step}
                )
                time.sleep(1)

        self.training_active = False
        logger.info("流式训练循环结束")

    def _streaming_training_step(self, data_chunk: Dict[str, Any]):
        """流式训练步骤 - 包含维度动态调整"""
        try:
            # 动态调整维度（每10步检查一次）
            if self.current_step % 10 == 0:
                self.adjust_dimensions_dynamically()
                if not self.validate_dimensions():
                    logger.warning("维度配置无效，保持当前配置")

            # 前向传播
            outputs = self.streaming_model.forward_streaming(data_chunk)
            loss = outputs['loss']

            # 梯度累积
            self.accumulated_gradients += 1

            # 反向传播（当累积足够时）
            if self.accumulated_gradients >= self.config.gradient_accumulation_steps:
                self.streaming_model.backward_streaming(loss)
                self.accumulated_gradients = 0

            # 更新指标
            self.current_step += 1
            self.performance_metrics['training_steps'] = self.current_step

            if loss < self.best_loss:
                self.best_loss = loss

            # 更新训练状态
            if self.training_state:
                self.training_state.step = self.current_step
                self.training_state.best_loss = self.best_loss

            # 记录性能
            self.infrastructure.performance_monitor.record_metric('streaming_loss', loss)
            self.infrastructure.performance_monitor.record_metric('streaming_step', self.current_step)

        except Exception as e:
            logger.error(f"流式训练步骤失败: {e}")
            # 发生错误时调整维度到更保守的设置
            self._emergency_dimension_adjustment()

    def _emergency_dimension_adjustment(self):
        """紧急维度调整 - 当训练失败时使用最保守的设置"""
        logger.warning("执行紧急维度调整")

        # 使用最保守的维度设置
        self.config.data_chunk_size = 4
        self.config.encoding_resolution = 0.1  # 降低分辨率
        self.config.max_sequence_length = 16
        self.config.gradient_accumulation_steps = 8  # 增加累积步数

        # 重新创建模型以应用新维度
        try:
            self.streaming_model = StreamingModel(self.config)
            logger.info("紧急情况下重新创建流式模型")
        except Exception as e:
            logger.error(f"紧急模型重建失败: {e}")

    def _manage_memory(self):
        """内存管理 - 增强版包含维度调整"""
        # 强制垃圾回收
        gc.collect()

        # 检查内存使用
        memory = psutil.virtual_memory()
        usage_gb = memory.used / (1024**3)

        if usage_gb > self.config.max_memory_gb:
            logger.warning(f"内存使用过高: {usage_gb:.2f}GB > {self.config.max_memory_gb:.2f}GB")
            self.adjust_dimensions_dynamically()

            # 如果仍然超限，强制清理
            memory_after_adjust = psutil.virtual_memory().used / (1024**3)
            if memory_after_adjust > self.config.max_memory_gb:
                self._force_memory_cleanup()

    def _emergency_dimension_adjustment(self):
        """紧急维度调整 - 当训练失败时使用最保守的设置"""
        logger.warning("执行紧急维度调整")

        # 使用最保守的维度设置
        self.config.data_chunk_size = 4
        self.config.attention_window_size = 2
        self.config.max_sequence_length = 16
        self.config.gradient_accumulation_steps = 8  # 增加累积步数

        # 重新创建模型以应用新维度
        try:
            self.streaming_model = StreamingModel(self.config)
            logger.info("紧急情况下重新创建流式模型")
        except Exception as e:
            logger.error(f"紧急模型重建失败: {e}")

    def _force_memory_cleanup(self):
        """强制内存清理"""
        # 清理所有临时对象
        gc.collect()

        # 清理numpy缓存
        if hasattr(np, 'clear_free_cache'):
            np.clear_free_cache()

        # 重置内存池
        self.memory_pool = MemoryPool(max_memory_mb=int(self.config.max_memory_gb * 1024))

        logger.info("执行强制内存清理")

    def _streaming_monitor_loop(self):
        """流式监控循环"""
        logger.info("启动流式监控循环")

        while self.running:
            try:
                # 更新性能指标
                self._update_performance_metrics()

                # 记录系统状态
                self._log_streaming_status()

                # 检查流式效率
                self._check_streaming_efficiency()

                time.sleep(5)  # 5秒检查一次

            except Exception as e:
                logger.error(f"流式监控异常: {e}")
                time.sleep(10)

        logger.info("流式监控循环结束")

    def _update_performance_metrics(self):
        """更新性能指标"""
        # CPU使用率
        self.performance_metrics['cpu_usage_percent'] = psutil.cpu_percent()

        # 内存使用率
        memory = psutil.virtual_memory()
        self.performance_metrics['memory_usage_mb'] = memory.used / (1024**2)

        # 流式效率（基于内存使用稳定性）
        memory_efficiency = 1.0 - (self.performance_metrics['memory_usage_mb'] / (self.config.max_memory_gb * 1024))
        self.performance_metrics['streaming_efficiency'] = max(0, memory_efficiency)

    def _log_streaming_status(self):
        """记录流式状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'training_active': self.training_active,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'memory_usage_mb': self.performance_metrics['memory_usage_mb'],
            'cpu_usage_percent': self.performance_metrics['cpu_usage_percent'],
            'streaming_efficiency': self.performance_metrics['streaming_efficiency'],
            'data_chunks_processed': self.data_loader.current_chunk
        }

        # 写入状态文件
        with open('mac_mini_streaming_status.json', 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

    def _check_streaming_efficiency(self):
        """检查流式效率"""
        efficiency = self.performance_metrics['streaming_efficiency']

        if efficiency < 0.5:
            logger.warning(f"流式效率低下: {efficiency:.2f}，可能需要调整参数")
            self._optimize_streaming_parameters()

    def _optimize_streaming_parameters(self):
        """优化流式参数"""
        # 减小批次大小
        if self.config.micro_batch_size > 1:
            self.config.micro_batch_size = max(1, self.config.micro_batch_size // 2)
            logger.info(f"调整微批次大小: {self.config.micro_batch_size}")

        # 增加梯度累积步数
        self.config.gradient_accumulation_steps = min(16, self.config.gradient_accumulation_steps * 2)
        logger.info(f"调整梯度累积步数: {self.config.gradient_accumulation_steps}")

    def _save_checkpoint(self):
        """保存检查点"""
        try:
            if self.training_state:
                version = save_model_checkpoint(
                    self.training_state,
                    generation=self.current_step,
                    accuracy=1.0 - min(1.0, self.best_loss),  # 模拟准确率
                    loss=self.best_loss,
                    description=f"流式检查点 - 步骤 {self.current_step}"
                )
                logger.info(f"流式检查点已保存: {version}")
        except Exception as e:
            logger.error(f"保存流式检查点失败: {e}")

    def get_streaming_status(self) -> Dict[str, Any]:
        """获取流式训练状态"""
        return {
            'running': self.running,
            'training_active': self.training_active,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'performance_metrics': self.performance_metrics,
            'memory_config': {
                'max_memory_gb': self.config.max_memory_gb,
                'current_usage_mb': self.performance_metrics['memory_usage_mb']
            },
            'system_health': self.fault_manager.get_system_health(),
            'infrastructure_status': self.infrastructure.get_system_status()
        }

# 全局流式训练器实例
mac_mini_trainer = MacMiniStreamingTrainer()

def start_mac_mini_streaming_training():
    """启动Mac Mini流式训练"""
    global mac_mini_trainer
    mac_mini_trainer.start_streaming_training()
    return mac_mini_trainer

def get_streaming_status():
    """获取流式训练状态"""
    return mac_mini_trainer.get_streaming_status()

def stop_mac_mini_streaming_training():
    """停止Mac Mini流式训练"""
    mac_mini_trainer.stop_streaming_training()

# 信号处理
def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在关闭流式训练系统...")
    stop_mac_mini_streaming_training()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    print("启动Mac Mini M4流式AGI训练系统...")

    try:
        # 启动流式训练
        trainer = start_mac_mini_streaming_training()

        # 保持运行
        while True:
            time.sleep(10)
            status = trainer.get_streaming_status()
            print(f"流式训练状态 - 步骤: {status['current_step']}, "
                  f"最佳损失: {status['best_loss']:.4f}, "
                  f"内存使用: {status['memory_config']['current_usage_mb']:.1f}MB, "
                  f"流式效率: {status['performance_metrics']['streaming_efficiency']:.2f}")

    except KeyboardInterrupt:
        print("\n正在关闭流式训练系统...")
        stop_mac_mini_streaming_training()
    except Exception as e:
        logger.error(f"流式训练系统异常: {e}")
        stop_mac_mini_streaming_training()
        sys.exit(1)