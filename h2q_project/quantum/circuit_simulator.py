"""
量子电路模拟器 — 完整门级模拟
================================

数学基础
--------
量子电路是一系列幺正算子的复合:

    U_circuit = U_d · U_{d-1} · … · U₁   (从右到左依次作用)

状态演化:
    |ψ(t)⟩ = U_circuit |ψ(0)⟩
    ρ(t) = U_circuit ρ(0) U_circuit†

含噪声的量子信道:
    Λ(ρ) = Σₖ Kₖ ρ Kₖ†   (每个门后施加对应噪声信道)

测量后状态 (Born 规则):
    P(k) = Tr(Pₖ ρ),  ρ' = Pₖ ρ Pₖ / P(k)

随机电路采样 (RCS) 的量子优越性度量:
    线性交叉熵基准 (XEB):
        F_XEB = 2^n ⟨P(x)⟩_samples - 1
    理想量子: F_XEB → 1
    随机经典: F_XEB → 0
    当 F_XEB > 0 时证明量子优越性

与 H2Q 项目的连接
-----------------
- 量子电路深度 ↔ H2Q_Knot_Kernel 的分形展开深度
- 测量后态 ↔ DAS AGI 的决策 (观测塌缩到特定选择)
- RCS 电路 ↔ das_gqs/supremacy_benchmark.py 的基准测试
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from h2q_project.quantum.hilbert_space import DensityMatrix, QuantumState
from h2q_project.quantum.gate_algebra import QuantumGateAlgebra, I2
from h2q_project.quantum.noise_model import RealisticNoiseModel, HardwareNoiseProfile

ga = QuantumGateAlgebra()
EPS = 1e-12


# ---------------------------------------------------------------------------
# 量子门指令集
# ---------------------------------------------------------------------------

@dataclass
class GateInstruction:
    """
    单条量子门指令。

    name    : 门名称 ('H', 'X', 'Y', 'Z', 'CNOT', 'Rx', 'Ry', 'Rz', 'S', 'T', 'CZ')
    qubits  : 作用的量子比特 (单比特: [q], 双比特: [control, target])
    params  : 旋转角等参数 (仅旋转门需要)
    """
    name: str
    qubits: List[int]
    params: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# 量子电路构建器
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """
    量子电路构建器和模拟器。

    支持:
    - 任意单比特门 (H, X, Y, Z, S, T, Rx, Ry, Rz)
    - 双比特纠缠门 (CNOT, CZ)
    - 含噪声模拟
    - 测量 (Z 基和任意基)
    - 随机电路采样 (RCS)

    连接 das_gqs/supremacy_benchmark.py:
        DASLazyGHZSimulator.build_ghz() ↔ QuantumCircuit.ghz()
        DASLazyGHZSimulator.apply_hadamard() ↔ circuit.h(q)
        DASLazyGHZSimulator.apply_cnot_link() ↔ circuit.cnot(c, t)
    """

    def __init__(self, n_qubits: int, noise: Optional[RealisticNoiseModel] = None):
        self.n_qubits = n_qubits
        self.noise = noise
        self._instructions: List[GateInstruction] = []

    # ------------------------------------------------------------------
    # 门添加接口
    # ------------------------------------------------------------------

    def h(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("H", [q]))
        return self

    def x(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("X", [q]))
        return self

    def y(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("Y", [q]))
        return self

    def z(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("Z", [q]))
        return self

    def s(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("S", [q]))
        return self

    def t(self, q: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("T", [q]))
        return self

    def rx(self, q: int, theta: float) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("Rx", [q], [theta]))
        return self

    def ry(self, q: int, theta: float) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("Ry", [q], [theta]))
        return self

    def rz(self, q: int, theta: float) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("Rz", [q], [theta]))
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("CNOT", [control, target]))
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction("CZ", [control, target]))
        return self

    def barrier(self) -> "QuantumCircuit":
        """标记层边界 (用于可视化, 不影响计算)"""
        self._instructions.append(GateInstruction("BARRIER", []))
        return self

    # ------------------------------------------------------------------
    # 电路工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def ghz(cls, n_qubits: int, noise: Optional[RealisticNoiseModel] = None) -> "QuantumCircuit":
        """
        GHZ 态制备电路: H(0) · CNOT(0,1) · … · CNOT(0,n-1)

        连接 DASLazyGHZSimulator.build_ghz():
            DAS 用惰性求值, 此处用精确状态向量模拟
        """
        qc = cls(n_qubits, noise)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cnot(0, i)
        return qc

    @classmethod
    def bell_pair(cls, noise: Optional[RealisticNoiseModel] = None) -> "QuantumCircuit":
        """Bell 态制备: H(0) · CNOT(0,1)"""
        qc = cls(2, noise)
        qc.h(0).cnot(0, 1)
        return qc

    @classmethod
    def random_circuit(
        cls,
        n_qubits: int,
        depth: int,
        noise: Optional[RealisticNoiseModel] = None,
        seed: Optional[int] = None,
    ) -> "QuantumCircuit":
        """
        随机量子电路 (用于 RCS 量子优越性基准)。

        结构: 交替的随机单比特门层 + 固定纠缠层
              模仿 Google Sycamore 的 RCS 电路结构

        数学:
            每层 = [Rnd_1q_gates] + [CNOT层]
            总深度 d 层

        与 das_gqs/supremacy_benchmark.py 的连接:
            该文件中的 RCS 基准也使用相似结构
        """
        rng = np.random.default_rng(seed)
        qc = cls(n_qubits, noise)

        # 单比特随机门集 (SU(2) 采样)
        single_q_gates = ["H", "Rx", "Ry", "Rz", "S", "T", "X", "Y", "Z"]

        for layer in range(depth):
            # 随机单比特门层
            for q in range(n_qubits):
                gate = single_q_gates[rng.integers(len(single_q_gates))]
                if gate in ("Rx", "Ry", "Rz"):
                    angle = float(rng.uniform(0, 2 * math.pi))
                    qc._instructions.append(GateInstruction(gate, [q], [angle]))
                else:
                    qc._instructions.append(GateInstruction(gate, [q]))

            # 纠缠层 (偶/奇交替)
            start = 0 if layer % 2 == 0 else 1
            for q in range(start, n_qubits - 1, 2):
                qc.cnot(q, q + 1)

        return qc

    # ------------------------------------------------------------------
    # 电路执行
    # ------------------------------------------------------------------

    def _apply_instruction(
        self, inst: GateInstruction, rho: DensityMatrix
    ) -> DensityMatrix:
        """执行单条门指令并可选施加噪声"""
        n = self.n_qubits
        name = inst.name
        qubits = inst.qubits

        if name == "BARRIER":
            return rho

        # 单比特门
        if name == "H":
            gate = ga.hadamard()
            U = ga.single_qubit_on_n(gate, n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name in ("X", "Y", "Z"):
            gate = {"X": ga.pauli_x(), "Y": ga.pauli_y(), "Z": ga.pauli_z()}[name]
            U = ga.single_qubit_on_n(gate, n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "S":
            U = ga.single_qubit_on_n(ga.phase(), n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "T":
            U = ga.single_qubit_on_n(ga.t_gate(), n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "Rx":
            U = ga.single_qubit_on_n(ga.rx(inst.params[0]), n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "Ry":
            U = ga.single_qubit_on_n(ga.ry(inst.params[0]), n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "Rz":
            U = ga.single_qubit_on_n(ga.rz(inst.params[0]), n, qubits[0])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_single_qubit_gate_noise(rho, qubits[0])

        elif name == "CNOT":
            U = ga.cnot(n, qubits[0], qubits[1])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_two_qubit_gate_noise(rho, qubits[0], qubits[1])

        elif name == "CZ":
            U = ga.cz(n, qubits[0], qubits[1])
            rho = rho.evolve(U)
            if self.noise:
                rho = self.noise.apply_two_qubit_gate_noise(rho, qubits[0], qubits[1])

        return rho

    def run(
        self,
        init_state: Optional[QuantumState] = None,
    ) -> DensityMatrix:
        """
        执行量子电路, 返回最终密度矩阵。

        参数
        ----
        init_state : 初始量子态 (默认: |0…0⟩)

        返回
        ----
        DensityMatrix: 最终量子态 (含噪声时为混态)
        """
        if init_state is None:
            init_state = QuantumState.zero_state(self.n_qubits)

        rho = init_state.density_matrix()
        for inst in self._instructions:
            rho = self._apply_instruction(inst, rho)
        return rho

    def sample(
        self,
        n_shots: int = 1024,
        init_state: Optional[QuantumState] = None,
    ) -> Dict[str, int]:
        """
        对电路执行多次采样测量, 返回比特串计数字典。

        Born 规则: P(bitstring x) = |⟨x|ψ⟩|²

        用于 XEB 基准测试: 采样结果与理论概率分布的交叉熵。
        """
        rho = self.run(init_state)
        probs = np.diag(rho.matrix).real
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum() + EPS

        rng = np.random.default_rng()
        d = 2 ** self.n_qubits
        samples = rng.choice(d, size=n_shots, p=probs)

        counts: Dict[str, int] = {}
        for s in samples:
            key = format(s, f"0{self.n_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def statevector_run(self, init_state: Optional[QuantumState] = None) -> QuantumState:
        """
        纯态模拟 (忽略噪声) — 用于 XEB 理论概率计算。
        """
        if init_state is None:
            amps = np.zeros(2 ** self.n_qubits, dtype=complex)
            amps[0] = 1.0
        else:
            amps = init_state.amplitudes.copy()

        for inst in self._instructions:
            if inst.name == "BARRIER":
                continue
            n = self.n_qubits
            name = inst.name
            qubits = inst.qubits

            if name == "H":
                U = ga.single_qubit_on_n(ga.hadamard(), n, qubits[0])
            elif name == "X":
                U = ga.single_qubit_on_n(ga.pauli_x(), n, qubits[0])
            elif name == "Y":
                U = ga.single_qubit_on_n(ga.pauli_y(), n, qubits[0])
            elif name == "Z":
                U = ga.single_qubit_on_n(ga.pauli_z(), n, qubits[0])
            elif name == "S":
                U = ga.single_qubit_on_n(ga.phase(), n, qubits[0])
            elif name == "T":
                U = ga.single_qubit_on_n(ga.t_gate(), n, qubits[0])
            elif name == "Rx":
                U = ga.single_qubit_on_n(ga.rx(inst.params[0]), n, qubits[0])
            elif name == "Ry":
                U = ga.single_qubit_on_n(ga.ry(inst.params[0]), n, qubits[0])
            elif name == "Rz":
                U = ga.single_qubit_on_n(ga.rz(inst.params[0]), n, qubits[0])
            elif name == "CNOT":
                U = ga.cnot(n, qubits[0], qubits[1])
            elif name == "CZ":
                U = ga.cz(n, qubits[0], qubits[1])
            else:
                continue
            amps = U @ amps

        return QuantumState(amps, self.n_qubits)

    @property
    def depth(self) -> int:
        """电路深度 (门层数, 不计 BARRIER)"""
        return sum(1 for inst in self._instructions if inst.name != "BARRIER")

    @property
    def gate_count(self) -> Dict[str, int]:
        """各类门的数量统计"""
        counts: Dict[str, int] = {}
        for inst in self._instructions:
            if inst.name != "BARRIER":
                counts[inst.name] = counts.get(inst.name, 0) + 1
        return counts

    def __len__(self) -> int:
        return self.depth

    def __repr__(self) -> str:
        gc = self.gate_count
        return (
            f"QuantumCircuit(n_qubits={self.n_qubits}, "
            f"depth={self.depth}, gates={gc}, "
            f"noise={'yes' if self.noise else 'no'})"
        )
