import numpy as np
from circuit import QuantumCircuit
import time

EPSILON = 1e-16
QUBITS_TO_TEST_QFT = 10

def test_qft_matrix_output():
    start_time = time.perf_counter()
    # Test 2: Compare QFT matrices and output matrices for 2 to number of qubits to test

    for num_qubits in range(2, QUBITS_TO_TEST_QFT):
        circuit = QuantumCircuit(num_qubits)
        circuit.load_qft_preset()
        qft_matrix = circuit.get_circuit_operator_matrix()

        # Compute the expected QFT matrix
        expected_qft_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                N = 2**num_qubits
                expected_qft_matrix[i,j] = (1/np.sqrt(N)) * np.exp(2j * np.pi * i * j / (N))

        assert np.allclose(qft_matrix, expected_qft_matrix, atol=EPSILON)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"======== QFT Test runtime: {elapsed_time:.2f} seconds =========")
    print("=============== QFT tests passed! ===============")

if __name__ == "__main__":
    test_qft_matrix_output()
    print("=============== All tests passed! ===============")