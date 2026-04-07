"""
H2Q Quantum Computing Layer
============================

数学基础：
  SU(2) ≅ S³ (单位四元数) — 单比特 Hilbert 空间
  C(3,0) ≅ G3 几何代数 — Pauli 门生成代数
  Fueter 解析函数空间 — 量子码字拓扑保护
  DAS Z₂ 层级 — 量子稳定子群层级
  Tribonacci SL(3,Z) — 离散量子幺正性

模块:
  hilbert_space  — 密度矩阵、Von Neumann 熵、量子纯度
  gate_algebra   — Pauli/Hadamard/CNOT/任意 SU(2) 量子门
  vqe_engine     — 变分量子本征值求解器
  quantum_agi    — 量子并行 AGI 自驱动进化引擎
  acceptance_test — 验收测试套件
"""

from h2q_project.quantum.hilbert_space import QuantumState, DensityMatrix
from h2q_project.quantum.gate_algebra import QuantumGateAlgebra
from h2q_project.quantum.vqe_engine import VQEEngine
from h2q_project.quantum.quantum_agi import QuantumParallelAGI

__all__ = [
    "QuantumState",
    "DensityMatrix",
    "QuantumGateAlgebra",
    "VQEEngine",
    "QuantumParallelAGI",
]
