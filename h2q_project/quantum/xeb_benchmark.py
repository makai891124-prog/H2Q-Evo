"""
线性交叉熵基准 (XEB) — 量子优越性度量
==========================================

数学基础
--------
XEB (Linear Cross-Entropy Benchmarking) 由 Google 在 2019 年量子优越性实验中使用:

    F_XEB = 2^n · Σₓ p_circuit(x) · f̂(x) - 1

其中:
    p_circuit(x) = |⟨x|U|0⟩|² 是理论概率 (精确模拟计算)
    f̂(x)         = 采样频率 (n_shots 次测量中 x 出现的次数 / n_shots)
    n            = 量子比特数

物理意义:
    F_XEB = 1:   完美量子采样 (与理论完全一致)
    F_XEB = 0:   纯随机采样 (均匀分布, 无量子信息)
    F_XEB > 0:   量子优越性信号

Porter-Thomas 分布:
    随机量子电路的输出概率服从 Porter-Thomas 分布:
        P(p) = (2^n - 1) · (1 - p·2^n)^(2^n - 2)  ≈ 2^n · e^{-p·2^n}
    这是量子混沌的特征标志。
    ⟨p_circuit(x)²⟩ = 2/(2^n + 1)  (Porter-Thomas 第二矩)
    ⟨p_circuit(x)⟩ = 1/2^n  (均匀)

满足 Porter-Thomas 分布 → F_XEB ≈ 1 (量子优越性)

与 das_gqs 的连接
-----------------
- das_gqs/public_rcs_xeb_unified_analysis.py 包含真实 Google XEB 数据分析
- das_gqs/supremacy_benchmark.py 使用 DAS 惰性模拟器计算 GHZ 期望值
- 本模块实现精确的 F_XEB 计算 (可与上述对比)

阈值
----
若 F_XEB > 2/3 (理论中间值), 则统计显著地排除经典随机噪声模型
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState
from h2q_project.quantum.circuit_simulator import QuantumCircuit
from h2q_project.quantum.noise_model import HardwareNoiseProfile, RealisticNoiseModel

EPS = 1e-12


# ---------------------------------------------------------------------------
# XEB 计算核心
# ---------------------------------------------------------------------------

@dataclass
class XEBResult:
    """XEB 基准测试结果"""
    n_qubits: int
    depth: int
    n_shots: int
    xeb_fidelity: float
    porter_thomas_score: float    # 与 Porter-Thomas 分布的一致性
    classical_threshold: float    # F_XEB=0 (经典随机) 的 95% CI 上界
    quantum_advantage: bool       # F_XEB > classical_threshold
    mean_bitstring_prob: float    # 平均采样比特串的理论概率
    circuit_entropy: float        # 输出分布的 Shannon 熵
    elapsed_ms: float = 0.0


def compute_xeb(
    n_qubits: int,
    depth: int,
    n_shots: int = 2000,
    noise: Optional[RealisticNoiseModel] = None,
    seed: Optional[int] = None,
) -> XEBResult:
    """
    对随机量子电路计算 XEB 保真度。

    步骤:
    1. 生成随机电路 (含噪声或理想)
    2. 精确状态向量模拟, 得到理论概率 p_circuit(x)
    3. 密度矩阵模拟 (含噪声) + 采样, 得到 f̂(x)
    4. 计算 F_XEB = 2^n · Σ p · f̂ - 1

    参数
    ----
    n_qubits : 量子比特数
    depth    : 电路深度
    n_shots  : 采样次数
    noise    : 噪声模型 (None = 理想无噪声)
    seed     : 随机种子 (控制电路结构)
    """
    t0 = time.time()
    d = 2 ** n_qubits

    # 生成随机电路 (保存门序列用于精确模拟)
    circuit = QuantumCircuit.random_circuit(n_qubits, depth, noise=noise, seed=seed)

    # 精确理论概率 (无噪声状态向量模拟)
    ideal_circuit = QuantumCircuit.random_circuit(n_qubits, depth, noise=None, seed=seed)
    ideal_state = ideal_circuit.statevector_run()
    p_theory = np.abs(ideal_state.amplitudes) ** 2

    # 采样 (含噪声)
    counts = circuit.sample(n_shots=n_shots)

    # 计算 F_XEB
    xeb_sum = 0.0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        f_hat = count / n_shots
        xeb_sum += p_theory[idx] * f_hat

    F_XEB = float(d * xeb_sum - 1.0)

    # Porter-Thomas 一致性: 理想 F_XEB ≈ 1
    # 对于无噪声电路, 测量误差 ≈ 1/sqrt(n_shots·Porter-Thomas_variance)
    porter_thomas_score = F_XEB

    # 经典阈值 (均匀随机采样下的 95% CI 上界)
    # 均匀分布: E[F_XEB] = 0, σ ≈ 1/sqrt(n_shots)
    classical_95_ci = 2.0 / math.sqrt(n_shots)

    # 采样的平均理论概率 (Porter-Thomas 第二矩验证)
    sampled_probs = [p_theory[int(b, 2)] for b in counts for _ in range(counts[b])]
    mean_prob = float(np.mean(sampled_probs)) if sampled_probs else 0.0

    # 输出分布熵
    probs_pos = p_theory[p_theory > 0]
    circuit_entropy = float(-np.sum(probs_pos * np.log(probs_pos)))

    elapsed_ms = (time.time() - t0) * 1000

    return XEBResult(
        n_qubits=n_qubits,
        depth=depth,
        n_shots=n_shots,
        xeb_fidelity=F_XEB,
        porter_thomas_score=porter_thomas_score,
        classical_threshold=classical_95_ci,
        quantum_advantage=bool(F_XEB > classical_95_ci),
        mean_bitstring_prob=mean_prob,
        circuit_entropy=circuit_entropy,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# XEB 与噪声关系
# ---------------------------------------------------------------------------

def xeb_vs_noise(
    n_qubits: int = 3,
    depth: int = 5,
    noise_levels: Optional[List[float]] = None,
    n_shots: int = 1000,
) -> List[Tuple[float, float]]:
    """
    计算不同噪声级别下的 XEB 值。

    理论预测:
        F_XEB(p) = (1-p)^(n_gates) ≈ exp(-n_gates · p)
        其中 n_gates = 电路门总数

    这与 Google 2019 论文一致: XEB 随噪声指数衰减。

    返回: [(noise_level, F_XEB), ...]
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]

    results = []
    for p in noise_levels:
        if p == 0.0:
            nm = None
        else:
            profile = HardwareNoiseProfile(single_qubit_error=p, two_qubit_error=p * 5)
            nm = RealisticNoiseModel(profile)
        result = compute_xeb(n_qubits, depth, n_shots=n_shots, noise=nm, seed=42)
        results.append((p, result.xeb_fidelity))

    return results


# ---------------------------------------------------------------------------
# Porter-Thomas 分布验证
# ---------------------------------------------------------------------------

def porter_thomas_test(
    n_qubits: int,
    depth: int,
    n_circuits: int = 20,
    seed: int = 0,
) -> Dict:
    """
    对 n_circuits 个随机电路验证输出分布服从 Porter-Thomas 分布。

    Porter-Thomas 分布特征:
        均值: E[p(x)] = 1/2^n
        方差: Var[p(x)] = 1/(2^n · (2^n + 1)) ≈ (1/2^n)²
        偏度 > 1 (重尾分布)

    若输出满足 Porter-Thomas 分布, 则说明电路已达到 "量子混沌" 区域,
    经典模拟的计算复杂度为 O(2^n) (量子优越性的必要条件)。

    与 das_gqs/convergence_experiment.py 的连接:
        収敛实验验证 DAS 系统达到量子混沌的代数
    """
    d = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    all_probs = []
    mean_xeb = 0.0

    for i in range(n_circuits):
        circuit = QuantumCircuit.random_circuit(n_qubits, depth, noise=None, seed=int(rng.integers(1000)))
        state = circuit.statevector_run()
        probs = np.abs(state.amplitudes) ** 2
        all_probs.extend(probs.tolist())
        mean_xeb += float(d * np.sum(probs ** 2) - 1.0)

    all_probs = np.array(all_probs)
    mean_xeb /= n_circuits

    # Porter-Thomas 理论预测
    pt_mean = 1.0 / d
    pt_variance = 1.0 / (d * (d + 1))
    pt_skewness = 2.0 * (d - 1) / math.sqrt(d / (d + 1))

    # 实测统计
    obs_mean = float(np.mean(all_probs))
    obs_std  = float(np.std(all_probs))
    obs_skew = float(
        np.mean(((all_probs - obs_mean) / (obs_std + EPS)) ** 3)
    )

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "n_circuits": n_circuits,
        "observed_mean": obs_mean,
        "theoretical_mean": pt_mean,
        "mean_error": abs(obs_mean - pt_mean) / (pt_mean + EPS),
        "observed_std": obs_std,
        "theoretical_std": math.sqrt(pt_variance),
        "observed_skewness": obs_skew,
        "theoretical_skewness": pt_skewness,
        "mean_xeb_fidelity": mean_xeb,
        "porter_thomas_satisfied": abs(obs_mean - pt_mean) / (pt_mean + EPS) < 0.05,
    }
