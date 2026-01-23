from h2q_project.quantum_algorithm import QuantumAlgorithm

def run_simulation(num_qubits, num_iterations):
    algorithm = QuantumAlgorithm(num_qubits)

    for _ in range(num_iterations):
        for i in range(num_qubits):
            algorithm.apply_hadamard(i)
        algorithm.apply_cnot(0, 1)

    outcome = algorithm.measure()
    return outcome

if __name__ == "__main__":
    num_qubits = 4
    num_iterations = 10
    result = run_simulation(num_qubits, num_iterations)
    print(f"Simulation result: {result}")