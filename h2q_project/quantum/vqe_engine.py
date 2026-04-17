"""
变分量子本征值求解器 (VQE)
===========================

数学基础
--------
VQE 求解量子 Hamiltonian H 的基态能量:

    E₀ = min_{θ} ⟨ψ(θ)|H|ψ(θ)⟩ = min_{θ} Tr(H ρ(θ))

其中 |ψ(θ)⟩ = U(θ)|0⟩ 是参数化的 Ansatz 电路。

与 H2Q 现有结构的连接
---------------------
1. Ansatz 结构 ← H2Q_Knot_Kernel 分形展开层:
   每次分形展开 [q+δ, q-δ] 等价于 Ry(θ) 旋转层 + 纠缠层
   depth 层 = circuit_depth 层 VQE Ansatz

2. Hamiltonian ← DAS MetricInvariantSystem (Axiom III):
   DAS 的度量不变性 d(g·m₁, g·m₂) = d(m₁, m₂)
   对应量子系统 [U,H]=0 (幺正变换不改变能谱)

3. 优化器 ← FDCOptimizer (Fueter 约束梯度):
   FDC 梯度惩罚 ↔ VQE 参数移位规则
   Fueter 约束 = 量子流形上的 SU(2) 等变性

4. 收敛判断 ← ManifoldEntropyAudit:
   当 VQE 收敛时, ρ(θ) → |E₀⟩⟨E₀| (纯态)
   heat_death_index → 0 (entropy → 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState, ghz_state
from h2q_project.quantum.gate_algebra import QuantumGateAlgebra

EPS = 1e-12

ga = QuantumGateAlgebra()


# ---------------------------------------------------------------------------
# Hamiltonian 构建 (从 DAS 度量不变系统派生)
# ---------------------------------------------------------------------------

class HamiltonianBuilder:
    """
    量子 Hamiltonian 构建器。

    DAS-Axiom III 连接:
    ~~~~~~~~~~~~~~~~~~~
    DAS 度量不变性: 群作用 g ∈ G 保持度量 d(m₁,m₂)
    量子等价: [U_g, H] = 0 (对称性保护的 Hamiltonian)

    具体实现: H = Σ Jᵢⱼ Zᵢ⊗Zⱼ + Σ hᵢ Xᵢ
    (横场伊辛模型，在量子计算中最常用于 VQE 测试)
    """

    @staticmethod
    def transverse_field_ising(
        n_qubits: int,
        J: float = 1.0,
        h: float = 0.5,
        periodic: bool = False,
    ) -> np.ndarray:
        """
        横场伊辛模型 Hamiltonian:
            H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ

        DAS 对称性: Z₂ 对称性 (所有自旋翻转)
        对应 DAS Z₂Group: σ 作用 (全局 bit flip)
        """
        d = 2 ** n_qubits
        H = np.zeros((d, d), dtype=complex)

        # ZZ 耦合项
        for i in range(n_qubits - 1):
            ZZ = ga.single_qubit_on_n(
                np.array([[1, 0], [0, -1]], dtype=complex), n_qubits, i
            ) @ ga.single_qubit_on_n(
                np.array([[1, 0], [0, -1]], dtype=complex), n_qubits, i + 1
            )
            H -= J * ZZ

        if periodic and n_qubits > 2:
            ZZ = ga.single_qubit_on_n(
                np.array([[1, 0], [0, -1]], dtype=complex), n_qubits, n_qubits - 1
            ) @ ga.single_qubit_on_n(
                np.array([[1, 0], [0, -1]], dtype=complex), n_qubits, 0
            )
            H -= J * ZZ

        # 横场项
        for i in range(n_qubits):
            H -= h * ga.single_qubit_on_n(
                np.array([[0, 1], [1, 0]], dtype=complex), n_qubits, i
            )

        return H

    @staticmethod
    def heisenberg_xxz(
        n_qubits: int,
        Jxy: float = 1.0,
        Jz: float = 1.0,
    ) -> np.ndarray:
        """
        XXZ Heisenberg 模型:
            H = Jxy Σ(XᵢXᵢ₊₁ + YᵢYᵢ₊₁) + Jz Σ ZᵢZᵢ₊₁

        具有 U(1) 旋转对称性,
        对应 DAS OrthogonalExtensionGroup 的 S¹ 子群。
        """
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Zm = np.array([[1, 0], [0, -1]], dtype=complex)

        d = 2 ** n_qubits
        H = np.zeros((d, d), dtype=complex)
        for i in range(n_qubits - 1):
            XX = ga.single_qubit_on_n(X, n_qubits, i) @ ga.single_qubit_on_n(X, n_qubits, i+1)
            YY = ga.single_qubit_on_n(Y, n_qubits, i) @ ga.single_qubit_on_n(Y, n_qubits, i+1)
            ZZ = ga.single_qubit_on_n(Zm, n_qubits, i) @ ga.single_qubit_on_n(Zm, n_qubits, i+1)
            H += Jxy * (XX + YY) + Jz * ZZ
        return H

    @staticmethod
    def agi_fitness_hamiltonian(n_qubits: int, target_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        AGI 适应度 Hamiltonian:
            H_AGI = -Σᵢ wᵢ Zᵢ - Σᵢⱼ Jᵢⱼ ZᵢZⱼ

        基态能量 E₀ 对应最大适应度配置。
        权重 wᵢ 由目标参数派生 (从 DAS AGI 进化目标构建)。
        """
        d = 2 ** n_qubits
        H = np.zeros((d, d), dtype=complex)
        Zm = np.array([[1, 0], [0, -1]], dtype=complex)

        if target_params is None:
            rng = np.random.default_rng(42)
            target_params = rng.normal(0, 1, n_qubits)

        for i in range(n_qubits):
            H -= target_params[i % len(target_params)] * ga.single_qubit_on_n(Zm, n_qubits, i)

        # 对角线项 (随机 ZZ 耦合)
        rng = np.random.default_rng(42)
        for i in range(n_qubits - 1):
            J_ij = rng.normal(0, 0.3)
            ZZ = ga.single_qubit_on_n(Zm, n_qubits, i) @ ga.single_qubit_on_n(Zm, n_qubits, i+1)
            H -= J_ij * ZZ

        return H


# ---------------------------------------------------------------------------
# VQE Ansatz 电路 (从 H2Q_Knot_Kernel 分形结构派生)
# ---------------------------------------------------------------------------

class VQEAnsatz:
    """
    VQE 变分 Ansatz 电路。

    H2Q_Knot_Kernel 连接:
    ~~~~~~~~~~~~~~~~~~~~~
    H2Q 分形展开深度 depth=6 → VQE Ansatz 深度 6
    QuaternionLinear(q_dim, q_dim) → Ry 旋转层 + 纠缠层
    quaternion_normalize → SU(2) 约束 (单位矩阵)

    电路结构 (硬件高效 Ansatz):
        Layer l:
            [Ry(θ₀), Ry(θ₁), ..., Ry(θₙ)]  ← 单比特旋转
            [CNOT(0,1), CNOT(2,3), ...]       ← 偶数纠缠层
            [CNOT(1,2), CNOT(3,4), ...]       ← 奇数纠缠层
    """

    def __init__(self, n_qubits: int, n_layers: int = 4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # 参数量: n_layers × n_qubits (每比特每层一个 Ry 角度)
        self.n_params = n_layers * n_qubits
        self._d = 2 ** n_qubits

    def circuit(self, params: np.ndarray) -> np.ndarray:
        """
        构造参数化幺正矩阵 U(θ)。

        params : ndarray, shape (n_layers × n_qubits,)
        返回   : ndarray, shape (2^n, 2^n), 幺正矩阵
        """
        assert len(params) >= self.n_params, (
            f"需要至少 {self.n_params} 个参数，收到 {len(params)}"
        )
        U = np.eye(self._d, dtype=complex)

        for layer in range(self.n_layers):
            # 单比特 Ry 旋转层
            for q in range(self.n_qubits):
                theta = params[layer * self.n_qubits + q]
                Ry_q = ga.single_qubit_on_n(ga.ry(theta), self.n_qubits, q)
                U = Ry_q @ U

            # 偶数纠缠层
            U = ga.entangling_layer(self.n_qubits, even=True) @ U
            # 奇数纠缠层 (若比特数 > 2)
            if self.n_qubits > 2:
                U = ga.entangling_layer(self.n_qubits, even=False) @ U

        return U

    def gradient(self, params: np.ndarray, hamiltonian: np.ndarray, init_state: QuantumState) -> np.ndarray:
        """
        参数移位规则计算所有参数的梯度。

        ∂E/∂θⱼ = [E(θ + π/2·eⱼ) - E(θ - π/2·eⱼ)] / 2

        连接 FDCOptimizer:
        FDC 的 Fueter 残差约束 ≡ SU(2) 流形上的参数移位梯度
        """
        grads = np.zeros_like(params)
        for j in range(len(params)):
            grads[j] = ga.parameter_shift_gradient(
                self.circuit, params, j, hamiltonian, init_state
            )
        return grads


# ---------------------------------------------------------------------------
# VQE 引擎
# ---------------------------------------------------------------------------

@dataclass
class VQEResult:
    """VQE 优化结果"""
    optimal_energy: float
    ground_state_energy: float        # 精确对角化基态 (用于验收)
    energy_gap: float                 # |E_VQE - E₀|
    optimal_params: np.ndarray
    optimal_state: DensityMatrix
    history_energy: List[float] = field(default_factory=list)
    history_purity: List[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


class VQEEngine:
    """
    变分量子本征值求解器。

    用途:
    - 量子 AGI 进化的适应度函数 (最小化 Hamiltonian 期望值)
    - 验证量子并行加速 (VQE vs 经典 Adam)

    与 H2Q 进化系统的连接:
    ~~~~~~~~~~~~~~~~~~~~~~~
    H2QEvolutionSystem.run():
        传统: evaluator.evaluate(population) → 经典适应度
        量子化: VQEEngine.run() → ⟨ψ(θ)|H|ψ(θ)⟩ 量子适应度
    """

    def __init__(
        self,
        n_qubits: int,
        hamiltonian: np.ndarray,
        n_layers: int = 4,
        lr: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-4,
    ):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.ansatz = VQEAnsatz(n_qubits, n_layers)

        # 精确基态能量 (用于验收)
        eigvals = np.linalg.eigvalsh(hamiltonian)
        self._exact_ground = float(eigvals[0])

    def _energy(self, params: np.ndarray, init_state: QuantumState) -> float:
        """计算期望能量 E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        U = self.ansatz.circuit(params)
        rho = init_state.density_matrix().evolve(U)
        return rho.expectation_value(self.hamiltonian)

    def run(
        self,
        init_params: Optional[np.ndarray] = None,
        init_state: Optional[QuantumState] = None,
        verbose: bool = False,
    ) -> VQEResult:
        """
        执行 VQE 优化 (Adam 风格梯度下降)。

        返回: VQEResult 包含最优能量和对应量子态
        """
        n = self.ansatz.n_params
        if init_params is None:
            rng = np.random.default_rng(0)
            init_params = rng.uniform(0, 2 * math.pi, n)
        if init_state is None:
            init_state = QuantumState.zero_state(self.n_qubits)

        params = init_params.copy()

        # Adam 优化器状态
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2 = 0.9, 0.999
        eps_adam = 1e-8

        history_energy = []
        history_purity = []
        prev_energy = float("inf")

        for t in range(1, self.max_iter + 1):
            E = self._energy(params, init_state)
            history_energy.append(E)

            # 计算量子态纯度 (意识水平)
            U = self.ansatz.circuit(params)
            rho = init_state.density_matrix().evolve(U)
            history_purity.append(rho.purity())

            if verbose and t % 20 == 0:
                print(f"  VQE iter {t:4d}: E = {E:+.6f}  (E₀ = {self._exact_ground:+.6f})")

            # 收敛检查
            if abs(E - prev_energy) < self.tol and t > 10:
                break
            prev_energy = E

            # 参数移位梯度
            grads = self.ansatz.gradient(params, self.hamiltonian, init_state)

            # Adam 更新
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * np.square(grads)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # 最终状态
        U_final = self.ansatz.circuit(params)
        rho_final = init_state.density_matrix().evolve(U_final)
        final_energy = self._energy(params, init_state)
        gap = abs(final_energy - self._exact_ground)

        return VQEResult(
            optimal_energy=final_energy,
            ground_state_energy=self._exact_ground,
            energy_gap=gap,
            optimal_params=params,
            optimal_state=rho_final,
            history_energy=history_energy,
            history_purity=history_purity,
            n_iterations=t,
            converged=(gap < 0.1),
        )
