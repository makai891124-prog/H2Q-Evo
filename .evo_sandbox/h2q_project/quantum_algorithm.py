import numpy as np

class QuantumAlgorithm:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex64)  # Reduced precision to complex64
        self.state[0] = 1.0

    def apply_hadamard(self, qubit_index):
        for i in range(2**self.num_qubits):
            if (i >> qubit_index) & 1 == 0:
                amp0 = self.state[i]
                amp1 = self.state[i + (1 << qubit_index)]
                self.state[i] = (amp0 + amp1) / np.sqrt(2, dtype=np.float32) # Reduced precision
                self.state[i + (1 << qubit_index)] = (amp0 - amp1) / np.sqrt(2, dtype=np.float32) # Reduced precision

    def apply_cnot(self, control_qubit, target_qubit):
        for i in range(2**self.num_qubits):
            if (i >> control_qubit) & 1 == 1:
                target_index = i ^ (1 << target_qubit)
                self.state[i], self.state[target_index] = self.state[target_index], self.state[i]

    def measure(self):
        probabilities = np.abs(self.state)**2
        outcome = np.random.choice(2**self.num_qubits, p=probabilities.astype(np.float32)) # Reduced precision
        return outcome