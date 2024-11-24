import numpy as np
from circuit import QuantumCircuit
import time

EPSILON = 1e-16
QUBITS_TO_TEST_QFT = 12

def test_qft_matrix_output():
    start_time = time.perf_counter()
    # Test 2: Compare QFT matrices and output matrices for 2 to number of qubits to test

    for num_qubits in range(2, QUBITS_TO_TEST_QFT + 1):
        # Start measurement of computing quantum circuit.
        start_time_qft = time.perf_counter()

        circuit = QuantumCircuit(num_qubits)
        circuit.load_qft_preset()
        qft_matrix = circuit.get_circuit_operator_matrix()

        # Calculate runtime of the computing the circuit
        end_time_qft = time.perf_counter()
        qft_elapsed_time = end_time_qft - start_time_qft
        print(f"===== QFT test with {num_qubits} qubits runtime: {qft_elapsed_time:.2f} ======")

        # Compute the expected QFT matrix
        expected_qft_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                N = 2**num_qubits
                expected_qft_matrix[i,j] = (1/np.sqrt(N)) * np.exp(2j * np.pi * i * j / (N))

        assert np.allclose(qft_matrix, expected_qft_matrix, atol=EPSILON)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"==== Total QFT Test runtime: {elapsed_time:.2f} seconds =======")
    print("=============== QFT tests passed! ===============")

if __name__ == "__main__":
    test_qft_matrix_output()
    print("=============== All tests passed! ===============")