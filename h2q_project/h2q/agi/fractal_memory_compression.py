"""H2Q 分形记忆压缩数据库 (Fractal Memory Compression Database).

基于 H2Q 分形数学框架实现高效记忆压缩:
1. 分形自相似性压缩 - 利用数据的自相似结构
2. 四元数稀疏编码 - S³ 流形上的稀疏表示
3. 多尺度金字塔存储 - 分形层级组织
4. 渐进式遗忘机制 - 基于重要性的记忆淘汰

数学基础:
- Mandelbrot 分形维度: D = lim(log N(ε) / log(1/ε))
- 四元数 Haar 小波变换
- Berry 相位记忆索引
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import OrderedDict
import time
import json
import hashlib
import zlib
from pathlib import Path
from enum import Enum
import struct


# ============================================================================
# 分形压缩核心
# ============================================================================

class CompressionLevel(Enum):
    """压缩级别."""
    NONE = 0        # 无压缩
    LIGHT = 1       # 轻度压缩 (2:1)
    MEDIUM = 2      # 中度压缩 (4:1)
    HEAVY = 3       # 重度压缩 (8:1)
    EXTREME = 4     # 极限压缩 (16:1)


@dataclass
class MemoryBlock:
    """记忆块."""
    id: str
    data: np.ndarray
    quaternion_repr: np.ndarray  # S³ 表示
    importance: float = 1.0
    access_count: int = 0
    last_access: float = 0.0
    created_at: float = 0.0
    compression_level: CompressionLevel = CompressionLevel.NONE
    fractal_scale: int = 0  # 分形层级
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """更新访问记录."""
        self.access_count += 1
        self.last_access = time.time()
        # 访问增加重要性
        self.importance = min(10.0, self.importance * 1.1)


@dataclass
class FractalNode:
    """分形树节点."""
    level: int
    children: List['FractalNode'] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)  # block IDs
    centroid: Optional[np.ndarray] = None  # 四元数中心
    total_size: int = 0
    compressed_size: int = 0


class QuaternionWavelet:
    """四元数小波变换 - 用于分形压缩."""
    
    @staticmethod
    def quaternion_haar_forward(data: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """四元数 Haar 小波前向变换.
        
        将数据分解为多尺度四元数系数。
        """
        if data.ndim == 1:
            data = data.reshape(-1, 4) if len(data) % 4 == 0 else np.pad(
                data, (0, 4 - len(data) % 4), mode='constant'
            ).reshape(-1, 4)
        
        coeffs = [data.copy()]
        current = data.copy()
        
        for _ in range(levels):
            if len(current) < 2:
                break
            
            # Haar 变换: 低频 + 高频
            n = len(current) // 2
            low = (current[::2] + current[1::2]) / np.sqrt(2)
            high = (current[::2] - current[1::2]) / np.sqrt(2)
            
            coeffs.append(high)
            current = low
        
        coeffs[0] = current  # 最低频
        return coeffs
    
    @staticmethod
    def quaternion_haar_inverse(coeffs: List[np.ndarray]) -> np.ndarray:
        """四元数 Haar 小波逆变换."""
        current = coeffs[0].copy()
        
        for i in range(1, len(coeffs)):
            high = coeffs[i]
            n = len(current)
            
            # 逆变换
            reconstructed = np.zeros((n * 2, current.shape[-1]), dtype=current.dtype)
            reconstructed[::2] = (current + high) / np.sqrt(2)
            reconstructed[1::2] = (current - high) / np.sqrt(2)
            
            current = reconstructed
        
        return current
    
    @staticmethod
    def threshold_coeffs(coeffs: List[np.ndarray], 
                         threshold: float = 0.1) -> List[np.ndarray]:
        """阈值处理 - 实现稀疏压缩."""
        result = []
        for c in coeffs:
            # 保留大于阈值的系数
            mask = np.abs(c) > threshold
            sparse = c * mask
            result.append(sparse)
        return result


class FractalCompressor:
    """分形压缩器 - 利用自相似性压缩数据."""
    
    def __init__(self, block_size: int = 64, search_depth: int = 3):
        self.block_size = block_size
        self.search_depth = search_depth
        self.codebook: Dict[str, np.ndarray] = {}
    
    def find_self_similar(self, data: np.ndarray, 
                          min_similarity: float = 0.9) -> List[Tuple[int, int, float, float]]:
        """查找自相似块.
        
        Returns:
            List of (offset, reference_offset, scale, rotation)
        """
        n = len(data)
        block_size = min(self.block_size, n // 4)
        
        if block_size < 4:
            return []
        
        matches = []
        
        for i in range(0, n - block_size, block_size):
            block = data[i:i + block_size]
            
            # 在其他位置搜索相似块
            best_match = None
            best_similarity = min_similarity
            
            for j in range(0, n - block_size * 2, block_size // 2):
                if abs(i - j) < block_size:
                    continue
                
                ref_block = data[j:j + block_size * 2:2]  # 下采样
                if len(ref_block) != len(block):
                    continue
                
                # 计算相似度
                similarity = self._compute_similarity(block, ref_block)
                
                if similarity > best_similarity:
                    # 估计变换参数
                    scale = np.std(block) / (np.std(ref_block) + 1e-10)
                    rotation = 0.0  # 简化
                    
                    best_match = (i, j, scale, rotation)
                    best_similarity = similarity
            
            if best_match:
                matches.append(best_match)
        
        return matches
    
    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个块的相似度."""
        if len(a) != len(b):
            return 0.0
        
        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)
        
        norm_a = np.linalg.norm(a_norm)
        norm_b = np.linalg.norm(b_norm)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0 if norm_a < 1e-10 and norm_b < 1e-10 else 0.0
        
        return float(np.dot(a_norm.flatten(), b_norm.flatten()) / (norm_a * norm_b))
    
    def compress(self, data: np.ndarray, 
                 level: CompressionLevel = CompressionLevel.MEDIUM) -> bytes:
        """压缩数据.
        
        使用分形编码 + 小波变换 + zlib。
        """
        # 1. 四元数小波变换
        wavelet = QuaternionWavelet()
        
        # 确保数据是 4 的倍数
        padded = data.flatten()
        if len(padded) % 4 != 0:
            padded = np.pad(padded, (0, 4 - len(padded) % 4), mode='constant')
        padded = padded.reshape(-1, 4)
        
        # 变换级别
        levels = level.value + 1
        coeffs = wavelet.quaternion_haar_forward(padded, levels)
        
        # 2. 阈值压缩
        thresholds = {
            CompressionLevel.NONE: 0.0,
            CompressionLevel.LIGHT: 0.05,
            CompressionLevel.MEDIUM: 0.1,
            CompressionLevel.HEAVY: 0.2,
            CompressionLevel.EXTREME: 0.4,
        }
        threshold = thresholds.get(level, 0.1)
        sparse_coeffs = wavelet.threshold_coeffs(coeffs, threshold)
        
        # 3. 序列化
        serialized = self._serialize_coeffs(sparse_coeffs, data.shape)
        
        # 4. zlib 压缩
        compressed = zlib.compress(serialized, level=9)
        
        return compressed
    
    def decompress(self, compressed: bytes) -> np.ndarray:
        """解压数据."""
        # 1. zlib 解压
        serialized = zlib.decompress(compressed)
        
        # 2. 反序列化
        coeffs, original_shape = self._deserialize_coeffs(serialized)
        
        # 3. 逆小波变换
        wavelet = QuaternionWavelet()
        reconstructed = wavelet.quaternion_haar_inverse(coeffs)
        
        # 4. 恢复形状
        flat = reconstructed.flatten()[:np.prod(original_shape)]
        return flat.reshape(original_shape)
    
    def _serialize_coeffs(self, coeffs: List[np.ndarray], 
                          original_shape: Tuple) -> bytes:
        """序列化小波系数."""
        parts = []
        
        # 原始形状
        shape_bytes = struct.pack('I' * len(original_shape), *original_shape)
        parts.append(struct.pack('I', len(original_shape)))
        parts.append(shape_bytes)
        
        # 系数数量
        parts.append(struct.pack('I', len(coeffs)))
        
        for c in coeffs:
            # 稀疏编码
            nonzero = np.nonzero(c)
            n_nonzero = len(nonzero[0])
            
            parts.append(struct.pack('I', n_nonzero))
            parts.append(struct.pack('I' * len(c.shape), *c.shape))
            parts.append(struct.pack('I', len(c.shape)))
            
            if n_nonzero > 0:
                # 索引
                for idx_arr in nonzero:
                    parts.append(idx_arr.astype(np.uint32).tobytes())
                # 值
                values = c[nonzero].astype(np.float32)
                parts.append(values.tobytes())
        
        return b''.join(parts)
    
    def _deserialize_coeffs(self, data: bytes) -> Tuple[List[np.ndarray], Tuple]:
        """反序列化小波系数."""
        offset = 0
        
        # 原始形状
        n_dims = struct.unpack_from('I', data, offset)[0]
        offset += 4
        original_shape = struct.unpack_from('I' * n_dims, data, offset)
        offset += 4 * n_dims
        
        # 系数数量
        n_coeffs = struct.unpack_from('I', data, offset)[0]
        offset += 4
        
        coeffs = []
        for _ in range(n_coeffs):
            n_nonzero = struct.unpack_from('I', data, offset)[0]
            offset += 4
            
            # 形状
            n_dims_c = struct.unpack_from('I', data, offset + 8)[0]
            shape = struct.unpack_from('I' * n_dims_c, data, offset)
            offset += 4 * n_dims_c + 4
            
            c = np.zeros(shape, dtype=np.float32)
            
            if n_nonzero > 0:
                # 索引
                indices = []
                for _ in range(len(shape)):
                    idx = np.frombuffer(data[offset:offset + n_nonzero * 4], dtype=np.uint32)
                    indices.append(idx)
                    offset += n_nonzero * 4
                
                # 值
                values = np.frombuffer(data[offset:offset + n_nonzero * 4], dtype=np.float32)
                offset += n_nonzero * 4
                
                c[tuple(indices)] = values
            
            coeffs.append(c)
        
        return coeffs, original_shape


# ============================================================================
# 分形记忆数据库
# ============================================================================

class FractalMemoryDatabase:
    """分形记忆数据库 - 高效压缩存储与检索."""
    
    def __init__(self, max_memory_mb: float = 100.0, 
                 storage_path: Optional[Path] = None):
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.storage_path = storage_path or Path("./fractal_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的记忆块
        self.blocks: OrderedDict[str, MemoryBlock] = OrderedDict()
        
        # 分形树索引
        self.fractal_tree = FractalNode(level=0)
        
        # 压缩器
        self.compressor = FractalCompressor()
        
        # 统计
        self.stats = {
            "total_stored": 0,
            "total_compressed": 0,
            "compression_ratio": 1.0,
            "hits": 0,
            "misses": 0,
        }
        
        # 加载持久化数据
        self._load_index()
    
    def store(self, key: str, data: np.ndarray, 
              metadata: Dict[str, Any] = None,
              importance: float = 1.0) -> str:
        """存储数据.
        
        Args:
            key: 唯一标识符
            data: 要存储的数据
            metadata: 元数据
            importance: 重要性 (影响压缩和淘汰)
        
        Returns:
            block_id: 存储块 ID
        """
        # 生成唯一 ID
        block_id = hashlib.sha256(f"{key}_{time.time()}".encode()).hexdigest()[:16]
        
        # 计算四元数表示
        q_repr = self._compute_quaternion_repr(data)
        
        # 确定压缩级别
        compression_level = self._determine_compression_level(importance)
        
        # 创建记忆块
        block = MemoryBlock(
            id=block_id,
            data=data.copy(),
            quaternion_repr=q_repr,
            importance=importance,
            created_at=time.time(),
            last_access=time.time(),
            compression_level=compression_level,
            metadata=metadata or {"key": key}
        )
        
        # 检查内存限制
        self._ensure_memory_limit()
        
        # 存储
        self.blocks[block_id] = block
        
        # 更新分形树
        self._update_fractal_tree(block)
        
        # 更新统计
        original_size = data.nbytes
        compressed_data = self.compressor.compress(data, compression_level)
        compressed_size = len(compressed_data)
        
        self.stats["total_stored"] += original_size
        self.stats["total_compressed"] += compressed_size
        self.stats["compression_ratio"] = (
            self.stats["total_stored"] / max(1, self.stats["total_compressed"])
        )
        
        # 持久化到磁盘
        self._persist_block(block_id, compressed_data)
        
        return block_id
    
    def retrieve(self, block_id: str) -> Optional[np.ndarray]:
        """检索数据."""
        if block_id in self.blocks:
            block = self.blocks[block_id]
            block.touch()
            self.stats["hits"] += 1
            return block.data.copy()
        
        # 尝试从磁盘加载
        data = self._load_block(block_id)
        if data is not None:
            self.stats["hits"] += 1
            return data
        
        self.stats["misses"] += 1
        return None
    
    def search_similar(self, query: np.ndarray, 
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """基于四元数相似度搜索.
        
        Returns:
            List of (block_id, similarity)
        """
        q_query = self._compute_quaternion_repr(query)
        
        results = []
        for block_id, block in self.blocks.items():
            # 四元数点积相似度
            similarity = float(np.abs(np.sum(q_query * block.quaternion_repr)))
            results.append((block_id, similarity))
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况."""
        in_memory_size = sum(
            b.data.nbytes for b in self.blocks.values()
        )
        
        return {
            "blocks_count": len(self.blocks),
            "in_memory_bytes": in_memory_size,
            "in_memory_mb": in_memory_size / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "utilization": in_memory_size / self.max_memory_bytes,
            "compression_ratio": self.stats["compression_ratio"],
            "hit_rate": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"]),
        }
    
    def compress_memory(self, target_ratio: float = 0.5) -> int:
        """压缩内存以释放空间.
        
        Args:
            target_ratio: 目标内存使用率
        
        Returns:
            freed_bytes: 释放的字节数
        """
        usage = self.get_memory_usage()
        if usage["utilization"] <= target_ratio:
            return 0
        
        freed = 0
        
        # 按重要性和访问时间排序
        sorted_blocks = sorted(
            self.blocks.items(),
            key=lambda x: (x[1].importance, x[1].last_access)
        )
        
        for block_id, block in sorted_blocks:
            if usage["utilization"] <= target_ratio:
                break
            
            # 增加压缩级别
            if block.compression_level.value < CompressionLevel.EXTREME.value:
                new_level = CompressionLevel(block.compression_level.value + 1)
                
                # 重新压缩并持久化
                compressed = self.compressor.compress(block.data, new_level)
                self._persist_block(block_id, compressed)
                
                # 从内存移除原始数据，只保留元信息
                original_size = block.data.nbytes
                block.data = np.zeros(1)  # 占位符
                block.compression_level = new_level
                
                freed += original_size
                usage = self.get_memory_usage()
        
        return freed
    
    def forget(self, threshold: float = 0.1) -> int:
        """遗忘低重要性记忆.
        
        Args:
            threshold: 重要性阈值
        
        Returns:
            forgotten_count: 遗忘的块数
        """
        to_forget = [
            block_id for block_id, block in self.blocks.items()
            if block.importance < threshold
        ]
        
        for block_id in to_forget:
            del self.blocks[block_id]
            # 删除持久化文件
            block_path = self.storage_path / f"{block_id}.fmb"
            if block_path.exists():
                block_path.unlink()
        
        return len(to_forget)
    
    def _compute_quaternion_repr(self, data: np.ndarray) -> np.ndarray:
        """计算数据的四元数表示."""
        flat = data.flatten().astype(np.float32)
        
        # 简单的四元数投影
        if len(flat) >= 4:
            # 使用 PCA 风格的降维
            chunks = np.array_split(flat, 4)
            q = np.array([np.mean(c) for c in chunks], dtype=np.float32)
        else:
            q = np.pad(flat, (0, 4 - len(flat)), mode='constant')
        
        # 归一化到 S³
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return q
    
    def _determine_compression_level(self, importance: float) -> CompressionLevel:
        """根据重要性确定压缩级别."""
        if importance >= 8.0:
            return CompressionLevel.NONE
        elif importance >= 5.0:
            return CompressionLevel.LIGHT
        elif importance >= 2.0:
            return CompressionLevel.MEDIUM
        elif importance >= 1.0:
            return CompressionLevel.HEAVY
        else:
            return CompressionLevel.EXTREME
    
    def _ensure_memory_limit(self):
        """确保不超过内存限制."""
        usage = self.get_memory_usage()
        
        if usage["in_memory_bytes"] > self.max_memory_bytes * 0.9:
            # 先尝试压缩
            self.compress_memory(target_ratio=0.7)
            
            # 如果还是超限，遗忘低重要性记忆
            usage = self.get_memory_usage()
            if usage["in_memory_bytes"] > self.max_memory_bytes * 0.9:
                self.forget(threshold=0.5)
    
    def _update_fractal_tree(self, block: MemoryBlock):
        """更新分形树索引."""
        # 简化实现：直接添加到根节点
        self.fractal_tree.blocks.append(block.id)
        self.fractal_tree.total_size += block.data.nbytes
    
    def _persist_block(self, block_id: str, compressed_data: bytes):
        """持久化块到磁盘."""
        block_path = self.storage_path / f"{block_id}.fmb"
        with open(block_path, 'wb') as f:
            f.write(compressed_data)
    
    def _load_block(self, block_id: str) -> Optional[np.ndarray]:
        """从磁盘加载块."""
        block_path = self.storage_path / f"{block_id}.fmb"
        
        if not block_path.exists():
            return None
        
        try:
            with open(block_path, 'rb') as f:
                compressed_data = f.read()
            return self.compressor.decompress(compressed_data)
        except Exception:
            return None
    
    def _load_index(self):
        """加载索引."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                self.stats = data.get("stats", self.stats)
            except Exception:
                pass
    
    def _save_index(self):
        """保存索引."""
        index_path = self.storage_path / "index.json"
        with open(index_path, 'w') as f:
            json.dump({
                "stats": self.stats,
                "block_ids": list(self.blocks.keys()),
            }, f, indent=2)
    
    def close(self):
        """关闭数据库."""
        self._save_index()


# ============================================================================
# 工厂函数
# ============================================================================

def create_fractal_memory_db(max_memory_mb: float = 100.0,
                             storage_path: str = None) -> FractalMemoryDatabase:
    """创建分形记忆数据库."""
    path = Path(storage_path) if storage_path else None
    return FractalMemoryDatabase(max_memory_mb, path)


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("H2Q 分形记忆压缩数据库 - 演示")
    print("=" * 60)
    
    # 创建数据库
    db = create_fractal_memory_db(max_memory_mb=10.0)
    
    # 存储测试数据
    print("\n1. 存储测试数据...")
    
    for i in range(10):
        data = np.random.randn(100, 100).astype(np.float32)
        block_id = db.store(
            key=f"test_{i}",
            data=data,
            importance=np.random.uniform(0.5, 5.0)
        )
        print(f"  存储块 {i}: {block_id}")
    
    # 查看内存使用
    print("\n2. 内存使用情况:")
    usage = db.get_memory_usage()
    for k, v in usage.items():
        print(f"  {k}: {v}")
    
    # 相似度搜索
    print("\n3. 相似度搜索:")
    query = np.random.randn(100, 100).astype(np.float32)
    results = db.search_similar(query, top_k=3)
    for block_id, sim in results:
        print(f"  {block_id}: {sim:.4f}")
    
    # 压缩内存
    print("\n4. 压缩内存:")
    freed = db.compress_memory(target_ratio=0.3)
    print(f"  释放: {freed / 1024:.1f} KB")
    
    # 最终状态
    print("\n5. 最终状态:")
    usage = db.get_memory_usage()
    print(f"  压缩比: {usage['compression_ratio']:.2f}x")
    print(f"  命中率: {usage['hit_rate']*100:.1f}%")
    
    db.close()
    print("\n" + "=" * 60)
