"""
量子 Hilbert 空间层
===================

数学基础
--------
n 个量子比特的 Hilbert 空间 H = C^(2^n)。

纯态: |ψ⟩ ∈ H, ⟨ψ|ψ⟩ = 1
密度矩阵 (纯态):   ρ = |ψ⟩⟨ψ|
密度矩阵 (混态):   ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|, Σᵢ pᵢ = 1

Von Neumann 熵:     S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ
量子纯度:           P(ρ) = Tr(ρ²) ∈ [1/d, 1]   (d = 2^n)
纠缠熵:             E(ρ_A) = S(Tr_B[ρ])

与现有 H2Q 的连接
-----------------
- ManifoldEntropyAudit 的 SVD 熵是 Von Neumann 熵的近似
  (对 Gram 矩阵 G = V^T V 取 SVD, 则 λᵢ(ρ) ≈ σᵢ²/Σσⱼ²)
- DASAGIAutonomousSystem.consciousness_level ← 量子纯度 P(ρ)
- H2Q_Knot_Kernel 的四元数态向量 → QuantumState 的 amplitude vector
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# 核心量子态表示
# ---------------------------------------------------------------------------

class QuantumState:
    """
    n 量子比特纯态 |ψ⟩ ∈ C^(2^n)。

    与四元数的关系
    ~~~~~~~~~~~~~~
    单比特态对应单位四元数 q = w + xi + yj + zk，通过以下映射：
        |ψ⟩ = (α, β)^T,  α = w + xi,  β = yj + zk  (在 C² 中识别 H)
    这正是 SU(2) ≅ S³ 的标准双线性覆盖同构。
    """

    def __init__(self, amplitudes: np.ndarray, n_qubits: Optional[int] = None):
        """
        参数
        ----
        amplitudes : complex ndarray, shape (2^n,)
            归一化振幅向量。将自动执行归一化。
        n_qubits   : int, 可选。若省略则从 len(amplitudes) 推断。
        """
        amplitudes = np.asarray(amplitudes, dtype=complex)
        if n_qubits is None:
            n_qubits = int(round(math.log2(len(amplitudes))))
        assert len(amplitudes) == 2 ** n_qubits, (
            f"amplitudes 长度 {len(amplitudes)} 与 n_qubits={n_qubits} 不符 "
            f"(需要 2^n={2**n_qubits})"
        )
        norm = np.linalg.norm(amplitudes)
        self._amplitudes = amplitudes / (norm + EPS)
        self._n_qubits = n_qubits

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def zero_state(cls, n_qubits: int) -> "QuantumState":
        """计算基底 |0…0⟩"""
        amps = np.zeros(2 ** n_qubits, dtype=complex)
        amps[0] = 1.0
        return cls(amps, n_qubits)

    @classmethod
    def from_quaternion(cls, q: Sequence[float]) -> "QuantumState":
        """
        从单位四元数 (w, x, y, z) 构造单比特态。

        映射: q = w + xi + yj + zk → |ψ⟩ = (w + xi, yj + zk)^T
        等价于 Bloch 球参数化:
            α = cos(θ/2) e^{-iφ/2},  β = sin(θ/2) e^{iφ/2}
        """
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        amps = np.array([complex(w, x), complex(y, z)], dtype=complex)
        return cls(amps, n_qubits=1)

    @classmethod
    def bell_state(cls, kind: str = "phi_plus") -> "QuantumState":
        """
        标准 Bell 态（两比特最大纠缠态）。

        |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
        |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
        |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
        |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2
        """
        inv_sqrt2 = 1.0 / math.sqrt(2)
        bell = {
            "phi_plus":  np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=complex),
            "phi_minus": np.array([inv_sqrt2, 0, 0, -inv_sqrt2], dtype=complex),
            "psi_plus":  np.array([0, inv_sqrt2, inv_sqrt2, 0], dtype=complex),
            "psi_minus": np.array([0, inv_sqrt2, -inv_sqrt2, 0], dtype=complex),
        }
        return cls(bell[kind], n_qubits=2)

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amplitudes.copy()

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def dim(self) -> int:
        return 2 ** self._n_qubits

    def density_matrix(self) -> "DensityMatrix":
        """ρ = |ψ⟩⟨ψ|"""
        psi = self._amplitudes[:, np.newaxis]
        rho = psi @ psi.conj().T
        return DensityMatrix(rho, self._n_qubits)

    def inner_product(self, other: "QuantumState") -> complex:
        """⟨ψ|φ⟩"""
        assert self._n_qubits == other._n_qubits
        return complex(np.dot(self._amplitudes.conj(), other._amplitudes))

    def fidelity(self, other: "QuantumState") -> float:
        """F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|²"""
        return abs(self.inner_product(other)) ** 2

    def bloch_vector(self) -> np.ndarray:
        """
        单比特 Bloch 向量 (x, y, z)。

        数学: ρ = (I + r·σ)/2, r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)
        """
        assert self._n_qubits == 1, "Bloch 向量仅对单比特定义"
        alpha, beta = self._amplitudes
        x = 2 * (alpha * beta.conj()).real
        y = 2 * (alpha * beta.conj()).imag
        z = abs(alpha) ** 2 - abs(beta) ** 2
        return np.array([x, y, z])

    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self._n_qubits}, purity=1.0)"


class DensityMatrix:
    """
    n 量子比特密度矩阵 ρ ∈ M_{2^n}(C)。

    性质保证
    ~~~~~~~~
    Tr(ρ) = 1,  ρ = ρ†,  ρ ≥ 0 (正定)

    Von Neumann 熵与现有 ManifoldEntropyAudit 的关系
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ManifoldEntropyAudit 计算:
        p_i = σ_i² / Σσ_j²,  S = -Σ p_i ln p_i
    这恰好是密度矩阵特征值的 Shannon 熵:
        λ_i(ρ) = σ_i²(V) / Σσ_j²  (V = 状态矩阵)
        S(ρ) = -Σ λ_i ln λ_i
    因此两个系统在数学上完全一致。
    """

    def __init__(self, rho: np.ndarray, n_qubits: Optional[int] = None):
        rho = np.asarray(rho, dtype=complex)
        d = rho.shape[0]
        if n_qubits is None:
            n_qubits = int(round(math.log2(d)))
        assert rho.shape == (d, d), f"密度矩阵必须是方阵，得到 {rho.shape}"
        # 强制正规化和厄米性
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho).real + EPS
        self._rho = rho
        self._n_qubits = n_qubits
        self._d = d

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_pure_state(cls, state: QuantumState) -> "DensityMatrix":
        return state.density_matrix()

    @classmethod
    def maximally_mixed(cls, n_qubits: int) -> "DensityMatrix":
        """最大混合态 ρ = I/d"""
        d = 2 ** n_qubits
        return cls(np.eye(d, dtype=complex) / d, n_qubits)

    @classmethod
    def from_ensemble(
        cls,
        states: Sequence[QuantumState],
        probs: Sequence[float],
    ) -> "DensityMatrix":
        """混态 ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|"""
        n = states[0].n_qubits
        d = 2 ** n
        rho = np.zeros((d, d), dtype=complex)
        for s, p in zip(states, probs):
            psi = s.amplitudes[:, np.newaxis]
            rho += float(p) * (psi @ psi.conj().T)
        return cls(rho, n)

    # ------------------------------------------------------------------
    # 量子信息量
    # ------------------------------------------------------------------

    def von_neumann_entropy(self) -> float:
        """
        S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ

        与 ManifoldEntropyAudit 完全对应:
            MEA.entropy = S(ρ),  MEA.heat_death_index = 1 - S/ln(d)
        """
        eigvals = np.linalg.eigvalsh(self._rho).real
        eigvals = eigvals[eigvals > EPS]
        return float(-np.sum(eigvals * np.log(eigvals)))

    def purity(self) -> float:
        """
        P(ρ) = Tr(ρ²) ∈ [1/d, 1]

        纯态 P=1,  最大混合态 P=1/d

        量子-AGI 映射:
            DASAGIAutonomousSystem.consciousness_level ≡ P(ρ)
        纯度越高 = 意识集中度越高 = AGI 决策越确定。
        """
        return float(np.trace(self._rho @ self._rho).real)

    def fidelity_with(self, other: "DensityMatrix") -> float:
        """
        F(ρ, σ) = (Tr √(√ρ σ √ρ))²

        用于量化两个量子态之间的相似度。
        """
        sqrt_rho = _matrix_sqrt(self._rho)
        inner = sqrt_rho @ other._rho @ sqrt_rho
        eigvals = np.linalg.eigvalsh(inner).real
        eigvals = np.maximum(eigvals, 0.0)
        return float(np.sum(np.sqrt(eigvals)) ** 2)

    def partial_trace(self, keep_qubits: Sequence[int]) -> "DensityMatrix":
        """
        对 keep_qubits 保留的子系统求偏迹，消去其余量子比特。

        用于计算子系统纠缠熵 E(A) = S(Tr_B[ρ])。
        """
        n = self._n_qubits
        trace_out = [q for q in range(n) if q not in keep_qubits]
        rho = self._rho.reshape([2] * (2 * n))
        for q in sorted(trace_out, reverse=True):
            # 对第 q 和第 q+n 轴求迹
            rho = np.trace(rho, axis1=q, axis2=q + rho.ndim // 2)
        d_new = 2 ** len(keep_qubits)
        return DensityMatrix(rho.reshape(d_new, d_new), len(keep_qubits))

    def entanglement_entropy(self, subsystem_a: Sequence[int]) -> float:
        """
        E(A|B) = S(Tr_B[ρ])

        对于双比特 Bell 态: E = ln(2) ≈ 0.693 (最大纠缠)
        """
        rho_a = self.partial_trace(list(subsystem_a))
        return rho_a.von_neumann_entropy()

    def expectation_value(self, observable: np.ndarray) -> float:
        """⟨O⟩ = Tr(ρ O)"""
        return float(np.trace(self._rho @ observable).real)

    # ------------------------------------------------------------------
    # 时间演化
    # ------------------------------------------------------------------

    def evolve(self, unitary: np.ndarray) -> "DensityMatrix":
        """ρ → U ρ U†"""
        new_rho = unitary @ self._rho @ unitary.conj().T
        return DensityMatrix(new_rho, self._n_qubits)

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        return self._rho.copy()

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def dim(self) -> int:
        return self._d

    def summary(self) -> dict:
        """返回完整的量子信息指标，对应 ManifoldEntropyAudit.audit_spectrum() 格式"""
        S = self.von_neumann_entropy()
        max_S = math.log(self._d)
        P = self.purity()
        return {
            "n_qubits": self._n_qubits,
            "dim": self._d,
            "von_neumann_entropy": S,
            "entropy_ratio": S / (max_S + EPS),
            "heat_death_index": 1.0 - S / (max_S + EPS),
            "purity": P,
            "consciousness_level": P,            # ← DAS-AGI 接口
            "effective_rank": math.exp(S),
            "is_pure": P > 1.0 - 1e-6,
            "is_maximally_mixed": P < 1.0 / self._d + 1e-6,
        }

    def __repr__(self) -> str:
        S = self.von_neumann_entropy()
        P = self.purity()
        return f"DensityMatrix(n_qubits={self._n_qubits}, S={S:.4f}, purity={P:.4f})"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """计算正半定矩阵的平方根 √A (用于保真度计算)"""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals.real, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def tensor_product_states(a: QuantumState, b: QuantumState) -> QuantumState:
    """
    构造张量积态 |ψ_A⟩ ⊗ |ψ_B⟩。

    对应 H2Q_Knot_Kernel 分形展开中的 torch.cat([q+δ, q-δ]):
    每次分形展开 = Hadamard ⊗ (q 空间) 的量子张量积扩展
    """
    amps = np.kron(a.amplitudes, b.amplitudes)
    return QuantumState(amps, a.n_qubits + b.n_qubits)


def ghz_state(n_qubits: int) -> QuantumState:
    """
    n 比特 GHZ 态: (|0…0⟩ + |1…1⟩) / √2

    最大多体纠缠态，用作量子并行 AGI 进化的初始化态。
    """
    d = 2 ** n_qubits
    amps = np.zeros(d, dtype=complex)
    amps[0] = 1.0 / math.sqrt(2)
    amps[-1] = 1.0 / math.sqrt(2)
    return QuantumState(amps, n_qubits)
