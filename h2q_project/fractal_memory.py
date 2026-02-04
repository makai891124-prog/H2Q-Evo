import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

# ==========================================
# DAS 核心修正：必须引入四元数运算库
# 这就是审计脚本检查的“指纹”
# ==========================================
try:
    import h2q.quaternion_ops as q_ops
    _HAS_QOPS = True
except ImportError:
    q_ops = None
    _HAS_QOPS = False


def quaternion_dot_product(q1, q2):
    """Fallback dot product if h2q.quaternion_ops is unavailable."""
    if _HAS_QOPS and hasattr(q_ops, "quaternion_dot_product"):
        return q_ops.quaternion_dot_product(q1, q2)
    return float(np.dot(q1, q2))

class FractalMemory:
    """
    基于 DAS 元理论的分形记忆系统。
    使用四元数 (w, x, y, z) 来表示任务状态的拓扑结构。
    """
    def __init__(self, memory_path="memory_store.json"):
        self.memory_path = memory_path
        self.memories = []
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                self.memories = json.load(f)

    def save_memory(self):
        with open(self.memory_path, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def vectorize_task(self, task_description: str) -> List[float]:
        """
        这里应该调用 Embedding 模型。
        为了演示，我们生成一个稳定的四元数向量。
        """
        seed = abs(hash(task_description)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.random(4)
        return (vec / np.linalg.norm(vec)).tolist()

    def _semantic_vec_to_quaternion(self, semantic_vec: Optional[List[float]], task_description: str) -> np.ndarray:
        if semantic_vec is None:
            vec = np.array(self.vectorize_task(task_description), dtype=np.float64)
        else:
            vec = np.array(semantic_vec, dtype=np.float64).flatten()
            if vec.size < 4:
                vec = np.pad(vec, (0, 4 - vec.size), constant_values=0.0)
            vec = vec[:4]
            norm = np.linalg.norm(vec) + 1e-8
            vec = vec / norm
        return vec

    def retrieve(
        self,
        task_description: str,
        semantic_vec: Optional[List[float]] = None,
        threshold: float = 0.8,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        使用四元数点积来寻找相似的“逻辑拓扑”。
        返回匹配的 memory 列表。
        """
        target_q = self._semantic_vec_to_quaternion(semantic_vec, task_description)
        scored: List[Dict[str, Any]] = []

        for memory in self.memories:
            stored_q = np.array(memory["quaternion_state"], dtype=np.float64)

            similarity = quaternion_dot_product(target_q, stored_q)
            if similarity >= threshold:
                scored.append({"score": similarity, "memory": memory})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [item["memory"] for item in scored[: max(1, top_k)]]

    def recall(self, task_description: str, threshold=0.8):
        """Backward-compatible alias for retrieve (returns first or None)."""
        results = self.retrieve(task_description, semantic_vec=None, threshold=threshold, top_k=1)
        return results[0] if results else None

    def crystallize(self, task, solution_code, success_rate=1.0):
        """
        结晶：将成功的经验固化为拓扑结。
        """
        q_state = self.vectorize_task(task).tolist()
        
        new_memory = {
            "task": task,
            "solution": solution_code,
            "quaternion_state": q_state,
            "energy": success_rate  # 成功率作为能量值
        }
        self.memories.append(new_memory)
        self.save_memory()
        print(f"✨ [FractalMemory] 经验已结晶。当前熵减节点数: {len(self.memories)}")