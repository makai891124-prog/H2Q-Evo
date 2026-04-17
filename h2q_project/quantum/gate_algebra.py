"""
量子门代数 — 从四元数/G3 几何代数派生
========================================

数学基础
--------
G3 几何代数的 Pauli 矩阵生成关系:

    e₂₃ = iX,  e₃₁ = iY,  e₁₂ = iZ
    X = σ_x = [[0,1],[1,0]]
    Y = σ_y = [[0,-i],[i,0]]
    Z = σ_z = [[1,0],[0,-1]]

SU(2) 旋转 ↔ 四元数旋子:

    U(n̂, θ) = exp(-iθ n̂·σ/2)
             = cos(θ/2) I - i sin(θ/2)(n_x X + n_y Y + n_z Z)
    ↔ q = cos(θ/2) + sin(θ/2)(n_x i + n_y j + n_z k)

Hadamard 的四元数表示:

    H = (X + Z)/√2 = [[1,1],[1,-1]]/√2
    对应四元数: q_H = (1+i)/√2 (在 xy 平面的 45° 旋子)

CNOT 的 DAS Z₂⊗Z₂ 推导:

    CNOT |c,t⟩ = |c, c⊕t⟩  (XOR = Z₂ 加法)
    Z₂ 作用: σ·|t⟩ = |t⊕1⟩ (bit flip)
    DAS OrthogonalExtension: 控制比特条件作用 ↔ Z₂⊗Z₂ 群作用

与 automorphic_dde.py 的连接
-----------------------------
apply_lie_group_action(q) = g·q·ḡ  (Lie 群自同构)
等价于量子门共轭: U ρ U†
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState

EPS = 1e-12


# ---------------------------------------------------------------------------
# Pauli 矩阵 (从 G3 bivectors 派生)
# ---------------------------------------------------------------------------

# G3 中: e₂₃ = iX, e₃₁ = iY, e₁₂ = iZ
# 因此 X = -i·e₂₃, Y = -i·e₃₁, Z = -i·e₁₂
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)    # e₂₃ bivector (π旋转)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex) # e₃₁ bivector (π旋转)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)   # e₁₂ bivector (π旋转)
H_GATE = (X + Z) / math.sqrt(2)                  # Hadamard: (X+Z)/√2

# Phase 门 S = √Z
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
# T 门 T = Z^(1/4)
T_GATE = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)


def _kron(*mats: np.ndarray) -> np.ndarray:
    """多矩阵 Kronecker 积"""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


class QuantumGateAlgebra:
    """
    量子门代数，提供单/多比特量子门。

    所有门均从四元数/G3 几何代数推导，与 H2Q 项目数学基础完全一致。
    """

    # ------------------------------------------------------------------
    # 单比特门 — 从 SU(2) 四元数参数化
    # ------------------------------------------------------------------

    @staticmethod
    def pauli_x() -> np.ndarray:
        """X 门 (NOT 门) — 对应 G3 e₂₃ bivector π 旋转"""
        return X.copy()

    @staticmethod
    def pauli_y() -> np.ndarray:
        """Y 门 — 对应 G3 e₃₁ bivector π 旋转"""
        return Y.copy()

    @staticmethod
    def pauli_z() -> np.ndarray:
        """Z 门 — 对应 G3 e₁₂ bivector π 旋转"""
        return Z.copy()

    @staticmethod
    def hadamard() -> np.ndarray:
        """
        Hadamard 门 H = (X+Z)/√2

        四元数表示: q = (1+i)/√2
        作用: |0⟩ → (|0⟩+|1⟩)/√2,  |1⟩ → (|0⟩-|1⟩)/√2

        连接到 H2Q_Knot_Kernel 分形展开:
        每个展开步 [q+δ, q-δ] 等价于在当前维度应用 Hadamard
        """
        return H_GATE.copy()

    @staticmethod
    def phase() -> np.ndarray:
        """S 门 (π/2 相位门)"""
        return S_GATE.copy()

    @staticmethod
    def t_gate() -> np.ndarray:
        """T 门 (π/4 相位门) — 通用量子计算所需"""
        return T_GATE.copy()

    @staticmethod
    def su2_rotation(theta: float, phi: float, lam: float) -> np.ndarray:
        """
        任意 SU(2) 旋转门 U₃(θ, φ, λ):

            U₃ = [[cos(θ/2),        -e^{iλ} sin(θ/2)],
                  [e^{iφ} sin(θ/2),  e^{i(φ+λ)} cos(θ/2)]]

        对应四元数参数化:
            q = cos(θ/2) + sin(θ/2)(sin(φ)i + sin(λ)j + cos(φ+λ)/2 k)

        连接到 automorphic_dde.py 的 LieGroupActionModule:
            apply_lie_group_action(q) = g·q·ḡ ↔ U₃(θ,φ,λ)|ψ⟩

        参数
        ----
        theta : 极角 (Bloch 球 θ) ∈ [0, π]
        phi   : 方位角 φ ∈ [0, 2π)
        lam   : 辅助相位 λ ∈ [0, 2π)
        """
        cos_h = math.cos(theta / 2)
        sin_h = math.sin(theta / 2)
        return np.array([
            [cos_h,                          -np.exp(1j * lam) * sin_h],
            [np.exp(1j * phi) * sin_h,        np.exp(1j * (phi + lam)) * cos_h],
        ], dtype=complex)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """Rₓ(θ) = exp(-iθX/2) — 绕 Bloch 球 x 轴旋转"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """R_y(θ) = exp(-iθY/2) — 绕 Bloch 球 y 轴旋转"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """R_z(θ) = exp(-iθZ/2) — 绕 Bloch 球 z 轴旋转"""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0,                       np.exp(1j * theta / 2)],
        ], dtype=complex)

    @staticmethod
    def from_quaternion(q: Sequence[float]) -> np.ndarray:
        """
        从单位四元数 (w, x, y, z) 构造 SU(2) 矩阵。

        精确同构: q → U = wI - i(xX + yY + zZ)

        这与 H2Q QuaternionLinear 的前向传播相同，
        只是在矩阵表示下显式写出。
        """
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        return np.array([
            [complex(w, -z),  complex(-y, -x)],
            [complex(y, -x),  complex(w, z)],
        ], dtype=complex)

    # ------------------------------------------------------------------
    # 多比特门 — 从 DAS Z₂ 层级构造
    # ------------------------------------------------------------------

    @staticmethod
    def cnot(n_qubits: int, control: int, target: int) -> np.ndarray:
        """
        n 比特系统中的 CNOT 门。

        DAS 推导:
            Z₂ 作用 σ: |t⟩ → |t⊕1⟩ (bit flip)
            DAS OrthogonalExtension Z₂⊗Z₂:
                当控制比特 = 1 时, 对目标应用 Z₂ flip
            CNOT_{c,t} = (I + Z_c)/2 ⊗ I_t + (I - Z_c)/2 ⊗ X_t
        """
        d = 2 ** n_qubits
        U = np.zeros((d, d), dtype=complex)
        for state in range(d):
            bit_c = (state >> (n_qubits - 1 - control)) & 1
            if bit_c == 1:
                flipped = state ^ (1 << (n_qubits - 1 - target))
                U[flipped, state] = 1.0
            else:
                U[state, state] = 1.0
        return U

    @staticmethod
    def cz(n_qubits: int, control: int, target: int) -> np.ndarray:
        """受控 Z 门 (CZ)"""
        d = 2 ** n_qubits
        U = np.eye(d, dtype=complex)
        for state in range(d):
            bit_c = (state >> (n_qubits - 1 - control)) & 1
            bit_t = (state >> (n_qubits - 1 - target)) & 1
            if bit_c == 1 and bit_t == 1:
                U[state, state] = -1.0
        return U

    @staticmethod
    def single_qubit_on_n(gate: np.ndarray, n_qubits: int, qubit: int) -> np.ndarray:
        """
        将单比特门作用于 n 比特系统的第 qubit 位。

        构造: I ⊗ … ⊗ gate ⊗ … ⊗ I
        """
        mats = [gate if i == qubit else I2 for i in range(n_qubits)]
        return _kron(*mats)

    @staticmethod
    def entangling_layer(n_qubits: int, even: bool = True) -> np.ndarray:
        """
        纠缠层: 对相邻比特对作用 CNOT，构建多体纠缠。

        用于 VQE Ansatz 和量子并行 AGI 初始化。
        """
        ga = QuantumGateAlgebra
        d = 2 ** n_qubits
        U = np.eye(d, dtype=complex)
        start = 0 if even else 1
        for i in range(start, n_qubits - 1, 2):
            U = ga.cnot(n_qubits, i, i + 1) @ U
        return U

    # ------------------------------------------------------------------
    # 量子电路执行
    # ------------------------------------------------------------------

    @staticmethod
    def apply(gate: np.ndarray, state: QuantumState) -> QuantumState:
        """|ψ'⟩ = U|ψ⟩"""
        new_amps = gate @ state.amplitudes
        return QuantumState(new_amps, state.n_qubits)

    @staticmethod
    def apply_density(gate: np.ndarray, rho: DensityMatrix) -> DensityMatrix:
        """ρ' = U ρ U†"""
        return rho.evolve(gate)

    @staticmethod
    def measure_z(state: QuantumState, qubit: int) -> Tuple[int, QuantumState, float]:
        """
        在 Z 基测量第 qubit 位。

        返回: (结果 0/1, 塌缩后态, 概率)

        数学:
            P(0) = Σ_{k: bit_qubit(k)=0} |αₖ|²
            P(1) = 1 - P(0)
            |ψ'⟩ = P_outcome |ψ⟩ / √P(outcome)
        """
        n = state.n_qubits
        amps = state.amplitudes
        probs = np.zeros(2)
        indices = [[], []]
        for k in range(2 ** n):
            bit = (k >> (n - 1 - qubit)) & 1
            probs[bit] += abs(amps[k]) ** 2
            indices[bit].append(k)

        outcome = int(np.random.choice(2, p=probs / (probs.sum() + EPS)))
        new_amps = np.zeros_like(amps)
        for k in indices[outcome]:
            new_amps[k] = amps[k]
        collapsed = QuantumState(new_amps / (math.sqrt(probs[outcome]) + EPS), n)
        return outcome, collapsed, float(probs[outcome])

    # ------------------------------------------------------------------
    # 参数移位规则 (用于 VQE 梯度计算)
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_shift_gradient(
        circuit_fn,
        params: np.ndarray,
        param_idx: int,
        observable: np.ndarray,
        init_state: QuantumState,
    ) -> float:
        """
        参数移位规则: ∂E/∂θⱼ = [E(θ + π/2·eⱼ) - E(θ - π/2·eⱼ)] / 2

        这是 VQE 中量子梯度的精确计算方法，
        等价于 FDC 优化器在四元数流形上的 Fueter 约束梯度。
        """
        shift = math.pi / 2
        params_plus  = params.copy(); params_plus[param_idx]  += shift
        params_minus = params.copy(); params_minus[param_idx] -= shift

        U_plus  = circuit_fn(params_plus)
        U_minus = circuit_fn(params_minus)

        rho_init = init_state.density_matrix()
        E_plus  = rho_init.evolve(U_plus).expectation_value(observable)
        E_minus = rho_init.evolve(U_minus).expectation_value(observable)
        return (E_plus - E_minus) / 2.0


# ---------------------------------------------------------------------------
# CHSH 算子 (对接 das_gqs/chsh_validation.py)
# ---------------------------------------------------------------------------

def chsh_operator(a: float, a_prime: float, b: float, b_prime: float) -> np.ndarray:
    """
    CHSH 算子 C = A⊗B - A⊗B' + A'⊗B + A'⊗B'

    其中 A = cos(a)Z + sin(a)X (Bloch 球赤道方向测量)

    经典界: |⟨C⟩| ≤ 2
    量子 Tsirelson 界: |⟨C⟩| ≤ 2√2 ≈ 2.828

    与 das_gqs/chsh_validation.py 的关系:
        geometric_correlation(axis_a, axis_b) = E(a,b) = -cos(a-b)
        chsh_validation 的 S = |E(ab) - E(ab') + E(a'b) + E(a'b')|
    """
    def meas_op(angle: float) -> np.ndarray:
        return math.cos(angle) * Z + math.sin(angle) * X

    A  = meas_op(a)
    Ap = meas_op(a_prime)
    B  = meas_op(b)
    Bp = meas_op(b_prime)

    return np.kron(A, B) - np.kron(A, Bp) + np.kron(Ap, B) + np.kron(Ap, Bp)


def tsirelson_violation(state: QuantumState) -> dict:
    """
    计算给定态的 CHSH 违背值 S，验证量子纠缠。

    对 |Φ⁺⟩ 的最优 Bell 角度: a=0, a'=π/2, b=π/4, b'=3π/4
    → E(a,b) = cos(a-b), S = 2√2 ≈ 2.828
    """
    C_op = chsh_operator(0, math.pi / 2, math.pi / 4, 3 * math.pi / 4)
    rho = state.density_matrix()
    S = abs(rho.expectation_value(C_op))
    return {
        "S": S,
        "classical_bound": 2.0,
        "tsirelson_bound": 2.0 * math.sqrt(2),
        "violates_bell": bool(S > 2.0 + 1e-6),
        "tsirelson_error": abs(S - 2.0 * math.sqrt(2)),
    }
