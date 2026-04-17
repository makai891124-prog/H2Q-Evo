"""
pytest 兼容的量子层单元测试。

运行: pytest h2q_project/tests/test_quantum_layer.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from h2q_project.quantum.hilbert_space import (
    DensityMatrix, QuantumState, ghz_state, tensor_product_states
)
from h2q_project.quantum.gate_algebra import (
    QuantumGateAlgebra, tsirelson_violation
)
from h2q_project.quantum.vqe_engine import HamiltonianBuilder, VQEEngine

ga = QuantumGateAlgebra()


class TestHilbertSpace:
    def test_zero_state_normalized(self):
        state = QuantumState.zero_state(2)
        assert abs(np.linalg.norm(state.amplitudes) - 1.0) < 1e-9

    def test_bell_state_purity_is_one(self):
        bell = QuantumState.bell_state("phi_plus")
        rho = bell.density_matrix()
        assert abs(rho.purity() - 1.0) < 1e-9

    def test_bell_state_entropy_zero(self):
        bell = QuantumState.bell_state("phi_plus")
        S = bell.density_matrix().von_neumann_entropy()
        assert S < 1e-9

    def test_bell_entanglement_entropy(self):
        """Entanglement entropy of Bell state partial trace = ln(2)"""
        bell = QuantumState.bell_state("phi_plus")
        rho = bell.density_matrix()
        E = rho.entanglement_entropy([0])
        assert abs(E - math.log(2)) < 1e-9

    def test_maximally_mixed_purity(self):
        n = 2
        rho = DensityMatrix.maximally_mixed(n)
        expected = 1.0 / (2 ** n)
        assert abs(rho.purity() - expected) < 1e-9

    def test_von_neumann_matches_mea_approach(self):
        """S_VN from eigenvalues == S_MEA from SVD of raw state matrix."""
        d = 4
        rng = np.random.default_rng(0)
        V = rng.normal(0, 1, (d, d)) + 1j * rng.normal(0, 1, (d, d))
        _, sigma_V, _ = np.linalg.svd(V)
        s2 = sigma_V ** 2
        p = s2 / s2.sum()
        S_MEA = float(-np.sum(p * np.log(p + 1e-14)))

        rho = DensityMatrix(V @ V.conj().T, n_qubits=2)
        S_VN = rho.von_neumann_entropy()
        assert abs(S_VN - S_MEA) < 1e-9


class TestGateAlgebra:
    def test_pauli_unitarity(self):
        for gate in [ga.pauli_x(), ga.pauli_y(), ga.pauli_z(), ga.hadamard()]:
            assert np.allclose(gate @ gate.conj().T, np.eye(2), atol=1e-10)

    def test_hadamard_creates_superposition(self):
        zero = QuantumState.zero_state(1)
        h_state = ga.apply(ga.hadamard(), zero)
        probs = np.abs(h_state.amplitudes) ** 2
        assert abs(probs[0] - 0.5) < 1e-9 and abs(probs[1] - 0.5) < 1e-9

    def test_cnot_entangles(self):
        """CNOT on |+0> creates Bell state |Φ+>"""
        zero = QuantumState.zero_state(2)
        # Apply H to qubit 0
        H2 = ga.single_qubit_on_n(ga.hadamard(), 2, 0)
        state_after_h = ga.apply(H2, zero)
        cnot = ga.cnot(2, 0, 1)
        bell_approx = ga.apply(cnot, state_after_h)
        # Should be (|00> + |11>)/sqrt(2)
        expected = QuantumState.bell_state("phi_plus")
        fidelity = bell_approx.fidelity(expected)
        assert fidelity > 0.999

    def test_chsh_bell_violation(self):
        bell = QuantumState.bell_state("phi_plus")
        result = tsirelson_violation(bell)
        assert result["violates_bell"]
        assert abs(result["S"] - 2 * math.sqrt(2)) < 1e-6

    def test_su2_from_quaternion_unitarity(self):
        q = [math.cos(0.3), math.sin(0.3), 0, 0]
        U = ga.from_quaternion(q)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


class TestVQEEngine:
    def test_vqe_ising_converges(self):
        H = HamiltonianBuilder.transverse_field_ising(2, J=1.0, h=0.5)
        vqe = VQEEngine(n_qubits=2, hamiltonian=H, n_layers=3, lr=0.1, max_iter=100)
        result = vqe.run(verbose=False)
        gap_ratio = result.energy_gap / abs(result.ground_state_energy)
        assert gap_ratio < 0.15, f"VQE gap ratio {gap_ratio:.3f} exceeds 15%"

    def test_vqe_energy_monotone(self):
        """VQE energy should generally decrease (first vs last 10 iters)"""
        H = HamiltonianBuilder.transverse_field_ising(2, J=1.0, h=0.5)
        vqe = VQEEngine(n_qubits=2, hamiltonian=H, n_layers=2, lr=0.1, max_iter=60)
        result = vqe.run(verbose=False)
        first_avg = np.mean(result.history_energy[:5])
        last_avg = np.mean(result.history_energy[-5:])
        assert last_avg <= first_avg + 0.5, "VQE energy did not decrease"
