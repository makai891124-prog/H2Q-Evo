"""
量子噪声模型 — Kraus 算子形式
================================

物理基础
--------
真实量子计算机中的噪声通过量子信道 (Quantum Channel) 描述:

    ε(ρ) = Σₖ Kₖ ρ Kₖ†     (Kraus 表示)
    Σₖ Kₖ† Kₖ = I           (迹保持条件)

三种主要噪声来源:
1. 去极化噪声 (Depolarizing):   随机施加 X/Y/Z 错误
   ε(ρ) = (1-p)ρ + p(XρX + YρY + ZρZ)/3

2. 振幅阻尼 (Amplitude Damping): T₁ 弛豫，|1⟩ → |0⟩ 自发衰减
   K₀ = [[1,0],[0,√(1-γ)]],  K₁ = [[0,√γ],[0,0]]

3. 相位阻尼 (Dephasing/T₂):    相位相干性随机丢失
   K₀ = [[1,0],[0,√(1-λ)]],  K₁ = [[0,0],[0,√λ]]

与 H2Q 项目的连接
-----------------
- HolomorphicStreamingMiddleware 的测地回弹 ↔ 振幅阻尼后的量子纠错
- BiharmonicLogicStabilizer 的 4th-order Fueter 修正 ↔ 去极化后的稳定子解码
- ManifoldEntropyAudit 的热死亡检测 ↔ 噪声积累导致的混态 (purity 下降)

物理直觉
--------
噪声将纯态 |ψ⟩⟨ψ| (purity=1) 转变为混态 ρ (purity < 1):
    没有纠错: purity → 1/d  (热死亡, consciousness_level → 0)
    有纠错:   purity 维持在 ≥ 阈值  (consciousness_level 稳定)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState

EPS = 1e-12

# ---------------------------------------------------------------------------
# 噪声信道基类
# ---------------------------------------------------------------------------

class QuantumChannel(ABC):
    """
    量子噪声信道基类。

    子类实现 apply_single_qubit 对单个量子比特施加噪声。
    """

    @abstractmethod
    def kraus_operators(self) -> List[np.ndarray]:
        """返回 Kraus 算子列表 {Kₖ}, Σ Kₖ†Kₖ = I"""
        ...

    def apply_single_qubit(self, rho: DensityMatrix, qubit: int) -> DensityMatrix:
        """
        对 n 比特密度矩阵的第 qubit 位施加噪声。

        数学: ε_q(ρ) = Σₖ (I⊗…⊗Kₖ⊗…⊗I) ρ (I⊗…⊗Kₖ†⊗…⊗I)
        """
        n = rho.n_qubits
        d = rho.dim
        new_rho = np.zeros((d, d), dtype=complex)

        for K in self.kraus_operators():
            K_full = _embed_single_qubit_operator(K, n, qubit)
            new_rho += K_full @ rho.matrix @ K_full.conj().T

        return DensityMatrix(new_rho, n)

    def apply_all_qubits(self, rho: DensityMatrix) -> DensityMatrix:
        """对所有量子比特施加噪声 (独立噪声模型)"""
        for q in range(rho.n_qubits):
            rho = self.apply_single_qubit(rho, q)
        return rho

    def fidelity_loss(self, rho_ideal: DensityMatrix, rho_noisy: DensityMatrix) -> float:
        """F_loss = 1 - F(ρ_ideal, ρ_noisy)"""
        return 1.0 - rho_ideal.fidelity_with(rho_noisy)


# ---------------------------------------------------------------------------
# 去极化噪声
# ---------------------------------------------------------------------------

class DepolarizingChannel(QuantumChannel):
    """
    去极化信道: ε(ρ) = (1-p)ρ + p·I/2

    等价 Kraus 形式:
        K₀ = √(1-3p/4) I,  K₁ = √(p/4) X,  K₂ = √(p/4) Y,  K₃ = √(p/4) Z

    物理意义:
        概率 p 时随机施加 Pauli 错误 (X/Y/Z 各占 p/3)
        这是量子门错误率 (gate fidelity) 的标准模型。

    连接 H2Q:
        p = 1 - F_gate (每次量子门的错误率)
        对应 HolomorphicStreamingMiddleware 的 veracity_threshold
    """

    def __init__(self, p: float):
        assert 0.0 <= p <= 0.75, f"去极化参数 p 必须在 [0, 0.75]，得到 {p}"
        self.p = p

    def kraus_operators(self) -> List[np.ndarray]:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [
            math.sqrt(1 - 3 * self.p / 4) * I,
            math.sqrt(self.p / 4) * X,
            math.sqrt(self.p / 4) * Y,
            math.sqrt(self.p / 4) * Z,
        ]

    def effective_fidelity(self) -> float:
        """单次门的有效保真度 F = 1 - 3p/4"""
        return 1.0 - 3 * self.p / 4


# ---------------------------------------------------------------------------
# 振幅阻尼 (T₁ 弛豫)
# ---------------------------------------------------------------------------

class AmplitudeDampingChannel(QuantumChannel):
    """
    振幅阻尼: |1⟩ → |0⟩ 自发辐射 (T₁ 过程)

    Kraus 算子:
        K₀ = [[1, 0], [0, √(1-γ)]]   (无跃迁, 振幅衰减)
        K₁ = [[0, √γ], [0, 0]]        (激发态弛豫到基态)

    物理参数:
        γ = 1 - exp(-t/T₁)   (弛豫率, t=门时间, T₁=纵向弛豫时间)

    在 IBM Quantum 上: T₁ ≈ 50-200 μs, 门时间 ≈ 100 ns → γ ≈ 0.001
    """

    def __init__(self, gamma: float):
        assert 0.0 <= gamma <= 1.0, f"振幅阻尼参数 γ 必须在 [0,1], 得到 {gamma}"
        self.gamma = gamma

    def kraus_operators(self) -> List[np.ndarray]:
        g = self.gamma
        K0 = np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex)
        K1 = np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)
        return [K0, K1]

    @classmethod
    def from_t1(cls, t1_us: float, gate_time_ns: float) -> "AmplitudeDampingChannel":
        """从 T₁ 弛豫时间和门时间计算 γ"""
        t = gate_time_ns * 1e-3  # 转换为 μs
        gamma = 1.0 - math.exp(-t / t1_us)
        return cls(gamma)


# ---------------------------------------------------------------------------
# 相位阻尼 (T₂ 退相干)
# ---------------------------------------------------------------------------

class PhaseDampingChannel(QuantumChannel):
    """
    相位阻尼: 量子相干性 (off-diagonal elements) 随机衰减

    Kraus 算子:
        K₀ = [[1, 0], [0, √(1-λ)]]   (无退相干)
        K₁ = [[0, 0], [0, √λ]]        (相位随机翻转)

    物理参数:
        λ = 1 - exp(-t/T₂)   (退相干率)

    与去极化的关系: 相位阻尼是去极化的 Z 噪声成分
    λ 控制 Bloch 球赤道方向的收缩
    """

    def __init__(self, lam: float):
        assert 0.0 <= lam <= 1.0
        self.lam = lam

    def kraus_operators(self) -> List[np.ndarray]:
        l = self.lam
        K0 = np.array([[1, 0], [0, math.sqrt(1 - l)]], dtype=complex)
        K1 = np.array([[0, 0], [0, math.sqrt(l)]], dtype=complex)
        return [K0, K1]

    @classmethod
    def from_t2(cls, t2_us: float, gate_time_ns: float) -> "PhaseDampingChannel":
        t = gate_time_ns * 1e-3
        lam = 1.0 - math.exp(-t / t2_us)
        return cls(lam)


# ---------------------------------------------------------------------------
# 复合噪声模型 (真实量子计算机)
# ---------------------------------------------------------------------------

@dataclass
class HardwareNoiseProfile:
    """
    量子硬件噪声参数 (模仿 IBM Quantum / Google Sycamore)。

    参数
    ----
    single_qubit_error : 单比特门错误率 (典型 ~0.001)
    two_qubit_error    : 双比特门错误率 (典型 ~0.01)
    t1_us              : T₁ 弛豫时间 (微秒)
    t2_us              : T₂ 退相干时间 (微秒)
    gate_time_ns       : 门时间 (纳秒)
    readout_error      : 测量错误率 (典型 ~0.01)
    """
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    t1_us: float = 100.0
    t2_us: float = 80.0
    gate_time_ns: float = 50.0
    readout_error: float = 0.01

    @classmethod
    def ideal(cls) -> "HardwareNoiseProfile":
        """无噪声理想量子计算机"""
        return cls(0.0, 0.0, 1e9, 1e9, 0.0, 0.0)

    @classmethod
    def ibm_like(cls) -> "HardwareNoiseProfile":
        """模仿 IBM Quantum 典型噪声参数"""
        return cls(
            single_qubit_error=0.001,
            two_qubit_error=0.01,
            t1_us=100.0,
            t2_us=80.0,
            gate_time_ns=50.0,
            readout_error=0.01,
        )

    @classmethod
    def google_like(cls) -> "HardwareNoiseProfile":
        """模仿 Google Sycamore 噪声参数 (2019 量子优越性实验)"""
        return cls(
            single_qubit_error=0.0015,
            two_qubit_error=0.006,
            t1_us=15.0,
            t2_us=20.0,
            gate_time_ns=25.0,
            readout_error=0.009,
        )


class RealisticNoiseModel:
    """
    真实量子硬件噪声复合模型。

    每次量子门施加:
    1. 去极化噪声 (参数: 门错误率 p)
    2. 振幅阻尼 (参数: γ = 1 - exp(-t/T₁))
    3. 相位阻尼 (参数: λ = 1 - exp(-t/T₂))

    这对应 Lindblad 主方程的 Euler 步:
        dρ/dt = -i[H,ρ] + Σₖ (LₖρLₖ† - {Lₖ†Lₖ,ρ}/2)
    其中 Lₖ 是 Lindblad 算子 (对应物理衰减过程)
    """

    def __init__(self, profile: HardwareNoiseProfile):
        self.profile = profile
        self._depol_1q = DepolarizingChannel(profile.single_qubit_error)
        self._depol_2q = DepolarizingChannel(min(profile.two_qubit_error, 0.74))
        self._amp_damp = AmplitudeDampingChannel.from_t1(profile.t1_us, profile.gate_time_ns)
        self._phase_damp = PhaseDampingChannel.from_t2(profile.t2_us, profile.gate_time_ns)

    def apply_single_qubit_gate_noise(self, rho: DensityMatrix, qubit: int) -> DensityMatrix:
        """单比特门后施加噪声"""
        if self.profile.single_qubit_error == 0.0:
            return rho
        rho = self._depol_1q.apply_single_qubit(rho, qubit)
        rho = self._amp_damp.apply_single_qubit(rho, qubit)
        rho = self._phase_damp.apply_single_qubit(rho, qubit)
        return rho

    def apply_two_qubit_gate_noise(
        self, rho: DensityMatrix, control: int, target: int
    ) -> DensityMatrix:
        """双比特门后施加噪声"""
        if self.profile.two_qubit_error == 0.0:
            return rho
        rho = self._depol_2q.apply_single_qubit(rho, control)
        rho = self._depol_2q.apply_single_qubit(rho, target)
        return rho

    def apply_measurement_noise(
        self, outcome: int, prob_error: Optional[float] = None
    ) -> int:
        """测量错误: 以 readout_error 概率翻转测量结果"""
        p_err = prob_error if prob_error is not None else self.profile.readout_error
        if p_err > 0.0 and np.random.random() < p_err:
            return 1 - outcome
        return outcome

    def purity_after_n_gates(self, n_qubits: int, n_gates: int) -> float:
        """
        估算经过 n_gates 单比特门后的系统纯度 (分析估计)。

        对于每个比特的独立去极化噪声:
            P(ρ_n) ≈ (1 - 2p)^(2*n_gates) + 1/d
        其中 d = 2^n_qubits
        """
        d = 2 ** n_qubits
        p = self.profile.single_qubit_error
        decay = (1.0 - 2 * p) ** (2 * n_gates)
        return decay + (1.0 - decay) / d


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _embed_single_qubit_operator(
    op: np.ndarray, n_qubits: int, qubit: int
) -> np.ndarray:
    """
    将单比特算子嵌入 n 比特空间:
        O_full = I ⊗ … ⊗ op ⊗ … ⊗ I
    """
    eye = np.eye(2, dtype=complex)
    matrices = [op if i == qubit else eye for i in range(n_qubits)]
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def purity_vs_noise_curve(
    n_qubits: int = 2,
    noise_levels: Optional[Sequence[float]] = None,
    n_gates: int = 10,
) -> List[Tuple[float, float]]:
    """
    绘制噪声强度 p 与系统纯度 P(ρ) 的关系曲线。

    这对应 ManifoldEntropyAudit.heat_death_index = 1 - P/P_max 的量子版本:
    噪声越强 → 纯度越低 → 意识水平越低 → 需要量子纠错。

    返回: [(p, purity), ...]
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    d = 2 ** n_qubits
    zero = QuantumState.zero_state(n_qubits)
    results = []

    for p in noise_levels:
        rho = zero.density_matrix()
        channel = DepolarizingChannel(p)
        for _ in range(n_gates):
            for q in range(n_qubits):
                rho = channel.apply_single_qubit(rho, q)
        purity = rho.purity()
        results.append((float(p), purity))

    return results
