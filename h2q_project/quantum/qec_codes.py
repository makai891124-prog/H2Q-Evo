"""
量子纠错码 (QEC) — 稳定子形式
=================================

数学基础
--------
量子纠错的核心是将 k 个逻辑量子比特编码到 n 个物理量子比特中:

    [[n, k, d]] 码: n 物理比特, k 逻辑比特, 距离 d

稳定子群 S:
    S = ⟨g₁, g₂, …, g_{n-k}⟩ ⊂ Pauli^n
    码空间: C = {|ψ⟩: g|ψ⟩ = |ψ⟩ for all g ∈ S}

错误综合征:
    测量稳定子 gᵢ → 结果 ±1
    +1: 码字空间  -1: 错误发生

纠错条件 (Knill-Laflamme):
    ⟨ψᵢ|E†ₐEᵦ|ψⱼ⟩ = Cₐᵦ δᵢⱼ

与 H2Q 的连接
-------------
- BiharmonicLogicStabilizer 的 Fueter 条件 ↔ 量子稳定子约束 g|ψ⟩ = +|ψ⟩
- HolomorphicStreamingMiddleware 测地回弹 ↔ 稳定子解码 + 逻辑运算
- DAS Z₂ 层级 ↔ 量子稳定子群 (Pauli 群的 Z₂ 子群)

实现的码
--------
1. [[3,1,1]] 比特翻转重复码 (Bit-Flip Repetition Code)
   逻辑 |0⟩_L = |000⟩,  逻辑 |1⟩_L = |111⟩
   稳定子: Z₁Z₂, Z₂Z₃
   可纠正: 单比特 X 错误

2. [[5,1,3]] 完美码 (Perfect Code) — 理论构建
   可纠正任意单比特错误

3. 简化表面码 (Distance-3 Surface Code) — 原理验证
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState
from h2q_project.quantum.gate_algebra import QuantumGateAlgebra, I2, X, Y, Z
from h2q_project.quantum.noise_model import DepolarizingChannel, RealisticNoiseModel

ga = QuantumGateAlgebra()
EPS = 1e-12

# ---------------------------------------------------------------------------
# Pauli 群工具
# ---------------------------------------------------------------------------

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = X.copy()
PAULI_Y = Y.copy()
PAULI_Z = Z.copy()

PAULI_MAP = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


def pauli_string(ops: str, n_qubits: int) -> np.ndarray:
    """
    从字符串构造多比特 Pauli 算子。

    例: pauli_string("ZZI", 3) = Z ⊗ Z ⊗ I

    用于构造稳定子生成元。
    """
    assert len(ops) == n_qubits
    mats = [PAULI_MAP[c] for c in ops.upper()]
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def measure_stabilizer(rho: DensityMatrix, stabilizer: np.ndarray) -> Tuple[int, DensityMatrix]:
    """
    测量稳定子算子 g (本征值 ±1)。

    结果:
        +1: 态在码字空间中 (无错误)
        -1: 错误综合征 (需要纠错)
    投影后密度矩阵相应塌缩。
    """
    d = rho.dim
    # 投影算子: P± = (I ± g) / 2
    proj_plus  = (np.eye(d, dtype=complex) + stabilizer) / 2
    proj_minus = (np.eye(d, dtype=complex) - stabilizer) / 2

    prob_plus = float(np.trace(proj_plus @ rho.matrix).real)
    prob_minus = 1.0 - prob_plus

    rng = np.random.default_rng()
    if prob_plus > 1.0 - EPS or rng.random() < prob_plus:
        outcome = +1
        new_rho_mat = proj_plus @ rho.matrix @ proj_plus / (prob_plus + EPS)
    else:
        outcome = -1
        new_rho_mat = proj_minus @ rho.matrix @ proj_minus / (prob_minus + EPS)

    return outcome, DensityMatrix(new_rho_mat, rho.n_qubits)


# ---------------------------------------------------------------------------
# [[3,1,1]] 比特翻转重复码
# ---------------------------------------------------------------------------

class BitFlipRepetitionCode:
    """
    [[3,1,1]] 比特翻转重复码。

    编码:
        |0⟩_L = |000⟩
        |1⟩_L = |111⟩
        |+⟩_L = (|000⟩ + |111⟩) / √2

    稳定子生成元:
        g₁ = Z₁Z₂I₃  (检查比特0和1是否相同)
        g₂ = I₁Z₂Z₃  (检查比特1和2是否相同)

    逻辑算子:
        X̄ = X₁X₂X₃  (逻辑翻转)
        Z̄ = Z₁      (逻辑相位)

    纠错能力:
        可纠正 1 个 X 错误 (3 个物理比特中的任意一个)
        对 Z 错误无保护 (需要相位翻转重复码 [[3,1,1]] 配合)

    与 H2Q DAS Z₂ 的连接:
        Z₂ 对称性 (所有比特翻转) = 逻辑 X̄
        DAS Z₂Group.apply(σ) ↔ X̄ 逻辑门
    """

    def __init__(self):
        self.n_physical = 3
        self.n_logical = 1

        # 稳定子生成元
        self.stabilizers = [
            pauli_string("ZZI", 3),  # g₁ = Z₁Z₂
            pauli_string("IZZ", 3),  # g₂ = Z₂Z₃
        ]

        # 逻辑算子
        self.logical_x = pauli_string("XXX", 3)
        self.logical_z = pauli_string("ZII", 3)

        # 错误综合征解码表
        # (g₁ 结果, g₂ 结果) → 错误位置 (0=无错误, 1/2/3=翻转物理比特 0/1/2)
        self._syndrome_table: Dict[Tuple[int,int], Optional[np.ndarray]] = {
            (+1, +1): None,                          # 无错误
            (-1, +1): pauli_string("XII", 3),        # 比特 0 翻转
            (-1, -1): pauli_string("IXI", 3),        # 比特 1 翻转
            (+1, -1): pauli_string("IIX", 3),        # 比特 2 翻转
        }

    def encode(self, logical_state: QuantumState) -> QuantumState:
        """
        将单比特逻辑态编码为 3 比特物理态。

        |0⟩ → |000⟩,  |1⟩ → |111⟩
        α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩

        数学:
            CNOT(0→1) · CNOT(0→2) · |ψ⟩ ⊗ |00⟩

        与 H2Q_Knot_Kernel 连接:
            分形展开 [q+δ, q-δ] 产生纠错冗余
            类似于在重复码中将单比特信息分散到多个比特
        """
        assert logical_state.n_qubits == 1
        d3 = 8  # 2^3

        alpha, beta = logical_state.amplitudes
        encoded_amps = np.zeros(d3, dtype=complex)
        encoded_amps[0b000] = alpha  # |000⟩
        encoded_amps[0b111] = beta   # |111⟩
        return QuantumState(encoded_amps, n_qubits=3)

    def decode(self, physical_state: QuantumState) -> QuantumState:
        """
        从 3 比特物理态恢复逻辑态。

        多数投票: 提取 |000⟩ 和 |111⟩ 振幅
        """
        assert physical_state.n_qubits == 3
        amps = physical_state.amplitudes
        alpha = amps[0b000]
        beta  = amps[0b111]
        return QuantumState(np.array([alpha, beta]), n_qubits=1)

    def measure_syndrome(self, rho: DensityMatrix) -> Tuple[Tuple[int,int], DensityMatrix]:
        """
        测量错误综合征 (g₁, g₂)。

        返回: ((s₁, s₂), 测量后密度矩阵)
        """
        s1, rho = measure_stabilizer(rho, self.stabilizers[0])
        s2, rho = measure_stabilizer(rho, self.stabilizers[1])
        return (s1, s2), rho

    def correct(self, rho: DensityMatrix, syndrome: Tuple[int,int]) -> DensityMatrix:
        """
        根据综合征施加纠错操作。
        """
        correction = self._syndrome_table.get(syndrome)
        if correction is None:
            return rho
        return rho.evolve(correction)

    def encode_correct_decode(
        self,
        logical_state: QuantumState,
        noise_model: Optional[RealisticNoiseModel] = None,
        noise_channel: Optional[DepolarizingChannel] = None,
    ) -> Tuple[QuantumState, Dict]:
        """
        完整的编码 → 噪声 → 纠错 → 解码流程。

        使用平均恢复映射 (Averaged Recovery Map):
            R(ρ) = Σ_s  C_s · P_s · ρ_noisy · P_s · C_s†

        其中:
            P_s  = 综合征 s 对应的投影算子
            C_s  = 对应的纠错幺正算子

        这是正确的量子纠错密度矩阵操作方式,
        不依赖随机采样, 保留完整的量子相干性。

        返回: (恢复的逻辑态, 指标字典)
        """
        d = 8  # 2^3

        # 1. 编码
        physical = self.encode(logical_state)
        rho = physical.density_matrix()

        # 2. 施加噪声
        if noise_model is not None:
            for q in range(3):
                rho = noise_model.apply_single_qubit_gate_noise(rho, q)
        elif noise_channel is not None:
            rho = noise_channel.apply_all_qubits(rho)

        purity_before = rho.purity()

        # 3. 平均恢复映射: R(ρ) = Σ_s C_s P_s ρ P_s C_s†
        rho_recovered = np.zeros((d, d), dtype=complex)
        detected_count = 0

        for s1 in [+1, -1]:
            for s2 in [+1, -1]:
                # 联合综合征投影算子
                P1 = (np.eye(d, dtype=complex) + s1 * self.stabilizers[0]) / 2
                P2 = (np.eye(d, dtype=complex) + s2 * self.stabilizers[1]) / 2
                P_s = P1 @ P2

                # 对应纠错算子
                syndrome = (s1, s2)
                C = self._syndrome_table.get(syndrome)
                C_mat = C if C is not None else np.eye(d, dtype=complex)

                # R_s(ρ) = C_s P_s ρ P_s C_s†
                projected = P_s @ rho.matrix @ P_s
                corrected = C_mat @ projected @ C_mat.conj().T
                rho_recovered += corrected

                if syndrome != (+1, +1):
                    prob_s = float(np.trace(projected).real)
                    if prob_s > 1e-6:
                        detected_count += 1

        rho_corr = DensityMatrix(rho_recovered, n_qubits=3)
        purity_after = rho_corr.purity()

        # 4. 正确解码: 从物理密度矩阵提取逻辑密度矩阵
        #    |0_L⟩ = |000⟩ (index 0),  |1_L⟩ = |111⟩ (index 7)
        #    ρ_L[i,j] = ⟨i_L|ρ_phys|j_L⟩
        R = rho_corr.matrix
        rho_logical_mat = np.array([
            [R[0, 0], R[0, 7]],
            [R[7, 0], R[7, 7]],
        ], dtype=complex)
        # 归一化 (可能因噪声轻微偏离)
        trace_l = rho_logical_mat[0, 0].real + rho_logical_mat[1, 1].real
        if trace_l > 1e-10:
            rho_logical_mat /= trace_l
        rho_logical = DensityMatrix(rho_logical_mat, n_qubits=1)

        # 5. 保真度计算
        ideal_rho = logical_state.density_matrix()
        fidelity = ideal_rho.fidelity_with(rho_logical)

        return QuantumState(
            np.array([
                complex(rho_logical.matrix[0, 0]) ** 0.5,
                complex(rho_logical.matrix[1, 1]) ** 0.5,
            ]),
            n_qubits=1,
        ), {
            "syndrome": (detected_count > 0),
            "purity_before_correction": purity_before,
            "purity_after_correction": purity_after,
            "logical_fidelity": fidelity,
            "error_detected": detected_count > 0,
        }


# ---------------------------------------------------------------------------
# [[5,1,3]] 完美量子纠错码 (理论验证)
# ---------------------------------------------------------------------------

class PerfectFiveQubitCode:
    """
    [[5,1,3]] 完美码 — 最小可纠正任意单比特错误的量子码。

    稳定子生成元 (4 个独立生成元):
        g₁ = XZZXI
        g₂ = IXZZX
        g₃ = XIXZZ
        g₄ = ZXIXZ

    逻辑算子:
        X̄ = XXXXX
        Z̄ = ZZZZZ

    参数: [[5, 1, 3]]
        5 物理比特, 1 逻辑比特, 距离 3 (可纠正 1 个任意 Pauli 错误)

    纠错完备性 (Knill-Laflamme 条件满足):
        对于任意 P ∈ {I,X,Y,Z}⁵, |wt(P)| ≤ 1:
        ⟨0̄|P̄|0̄⟩ = 0  (错误正交于码字)

    实现: 验证编码和稳定子约束 (完整实现需要 2^5=32 维希尔伯特空间)
    """

    def __init__(self):
        self.n_physical = 5
        self.n_logical = 1

        # 稳定子生成元 (4 个)
        self.stabilizers = [
            pauli_string("XZZXI", 5),
            pauli_string("IXZZX", 5),
            pauli_string("XIXZZ", 5),
            pauli_string("ZXIXZ", 5),
        ]

        # 逻辑 X̄, Z̄
        self.logical_x = pauli_string("XXXXX", 5)
        self.logical_z = pauli_string("ZZZZZ", 5)

    def compute_code_space(self) -> np.ndarray:
        """
        计算码字空间 (所有稳定子 +1 本征空间的交)。

        返回: 码字空间的标准正交基, shape (32, 2)
              两列分别为 |0̄⟩ 和 |1̄⟩
        """
        d = 2 ** 5
        P = np.eye(d, dtype=complex)  # 开始为完整希尔伯特空间投影

        for g in self.stabilizers:
            P = P @ (np.eye(d, dtype=complex) + g) / 2  # +1 本征空间投影

        # 提取码字空间基
        eigvals, eigvecs = np.linalg.eigh(P)
        code_basis = eigvecs[:, eigvals > 0.5]
        return code_basis

    def verify_stabilizers(self) -> Dict[str, bool]:
        """
        验证稳定子生成元的对易关系。

        对于有效稳定子码, 所有生成元两两对易: [gᵢ, gⱼ] = 0
        """
        n = len(self.stabilizers)
        results = {}
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = self.stabilizers[i], self.stabilizers[j]
                commutator = gi @ gj - gj @ gi
                commutes = bool(np.allclose(commutator, 0, atol=1e-10))
                results[f"g{i+1}_g{j+1}_commute"] = commutes
        return results

    def code_distance(self) -> int:
        """
        [[5,1,3]] 码的理论距离为 3。
        (需要至少 3 个 Pauli 错误才能无法区分)
        """
        return 3


# ---------------------------------------------------------------------------
# QEC 基准 — 有/无纠错的保真度对比
# ---------------------------------------------------------------------------

@dataclass
class QECBenchmarkResult:
    """量子纠错基准测试结果"""
    noise_level: float
    fidelity_no_qec: float    # 无纠错时的逻辑保真度
    fidelity_with_qec: float  # 有纠错时的逻辑保真度
    qec_improvement: float    # fidelity_with_qec - fidelity_no_qec
    syndrome_detection_rate: float  # 错误被检测到的比率


def run_qec_benchmark(
    noise_levels: Optional[List[float]] = None,
    n_shots: int = 100,
    logical_state: Optional[QuantumState] = None,
) -> List[QECBenchmarkResult]:
    """
    在不同噪声级别下测试量子纠错效果。

    验收目标:
        对于 p ≤ 0.05: fidelity_with_qec > fidelity_no_qec + 0.1
        这表明量子纠错在阈值以下有明显改善效果。

    阈值定理 (Threshold Theorem):
        若单比特错误率 p < p_threshold ≈ 0.01-0.05 (取决于码),
        则量子纠错可以任意降低逻辑错误率。
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.03, 0.05, 0.08, 0.10]
    if logical_state is None:
        logical_state = QuantumState(
            np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]), n_qubits=1
        )

    code = BitFlipRepetitionCode()
    results = []

    for p in noise_levels:
        channel = DepolarizingChannel(p)
        fid_no_qec_list = []
        fid_qec_list = []
        detected_count = 0

        for _ in range(n_shots):
            # 无纠错: 直接对逻辑态施加噪声
            rho_direct = logical_state.density_matrix()
            rho_noisy = channel.apply_single_qubit(rho_direct, 0)
            fid_no_qec_list.append(logical_state.density_matrix().fidelity_with(rho_noisy))

            # 有纠错: 编码 → 噪声 → 纠错 → 解码
            _, metrics = code.encode_correct_decode(logical_state, noise_channel=channel)
            fid_qec_list.append(metrics["logical_fidelity"])
            if metrics["error_detected"]:
                detected_count += 1

        results.append(QECBenchmarkResult(
            noise_level=p,
            fidelity_no_qec=float(np.mean(fid_no_qec_list)),
            fidelity_with_qec=float(np.mean(fid_qec_list)),
            qec_improvement=float(np.mean(fid_qec_list) - np.mean(fid_no_qec_list)),
            syndrome_detection_rate=detected_count / n_shots,
        ))

    return results
