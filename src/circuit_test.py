import numpy as np
from circuit import QuantumCircuit
from multi_qubit import MultiQubit
import time

EPSILON = 1e-16
QUBITS_TO_TEST_QFT = 9


def qft_on_sine(number_of_qubits):
    start_time = time.perf_counter()
    x = np.linspace(-2 * np.pi,2 * np.pi,2**number_of_qubits)
    amplitudes = np.sin(x)
    probabilities = np.abs(amplitudes) ** 2 
    norm_constant  = np.sum(probabilities)
    amplitudes /= np.sqrt(norm_constant)

    mt = MultiQubit(amplitudes)
    # mt.plot_probabilities()
    mt.plot_probabilities()
    mt.plot_amplitudes()

    circuit = QuantumCircuit(number_of_qubits)
    # Load the qft circuit
    circuit.load_qft_preset()
    result = circuit.apply_circuit(mt)
    result.plot_probabilities()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"=== The runtime of QFT on a sine function is: {elapsed_time:0.2f} === ")

def normal_pdf(x, mu=0, sigma=1):
    """
    Custom implementation of normal probability density function.
    
    Args:
    x (float): Input value
    mu (float): Mean of the distribution
    sigma (float): Standard deviation of the distribution
    
    Returns:
    float: Probability density
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def qft_on_gaussian(number_of_qubits,mu=0,sigma=1):

    start_time = time.perf_counter()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 2**number_of_qubits)
    
    # Calculate probabilities using custom normal PDF
    probabilities = normal_pdf(x, mu, sigma)
    
    # Normalize probabilities to sum to 1
    probabilities /= np.sum(probabilities)

    amplitudes = np.sqrt(probabilities)

    mt = MultiQubit(amplitudes)
    # mt.plot_probabilities()
    mt.plot_measurements()
    circuit = QuantumCircuit(number_of_qubits)
    # Load the qft circuit
    circuit.load_qft_preset()
    result = circuit.apply_circuit(mt)
    result.plot_measurements()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"=== The runtime of QFT on a gaussian distrubition is: {elapsed_time:0.2f} === ")

def test_qft_matrix_output(qubits_to_test = 3):
    start_time = time.perf_counter()
    # Test 2: Compare QFT matrices and output matrices for 2 to number of qubits to test

    for num_qubits in range(2, qubits_to_test + 1):
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
    # test_qft_matrix_output(QUBITS_TO_TEST_QFT)
    qft_on_gaussian(9)
    # qft_on_sine(6)
    print("=============== All tests passed! ===============")