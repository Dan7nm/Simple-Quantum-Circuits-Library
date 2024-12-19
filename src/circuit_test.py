import numpy as np
from circuit import QuantumCircuit
from multi_qubit import MultiQubit
from qubit import Qubit
import time
import random

EPSILON = 1e-16
QUBITS_TO_TEST = 2

def qft_on_sine(number_of_qubits: int) -> None:
    """
    Applies the Quantum Fourier Transform (QFT) to a quantum state with amplitudes based on a sine function and compares the resulting amplitudes with the expected amplitudes using numpy's FFT method.
    The function also plots the input states's amplitudes and the measurement plot of the resulting state.

    Parameters
    ----------
    number_of_qubits : int
        The number of qubits in the quantum state.

    Returns
    -------
    None
    """
    start_time = time.perf_counter()
    
    # Create normalized state with amplitudes of a sine function.
    x = np.linspace(-2 * np.pi, 2 * np.pi, 2**number_of_qubits)
    amplitudes = np.sin(x)
    probabilities = np.abs(amplitudes) ** 2 
    norm_constant = np.sum(probabilities)
    amplitudes /= np.sqrt(norm_constant)
    mt = MultiQubit(amplitudes)

    # Plot the input state amplitudes.
    mt.plot_amplitudes(plot_type='line')

    # Load the QFT circuit and apply it.
    circuit = QuantumCircuit(number_of_qubits)
    circuit.load_qft_preset()
    result = circuit.run_circuit(mt)

    # Plot the amplitudes after QFT.
    result.plot_measurements()

    # Compare with expected amplitudes using numpy's FFT.
    expected_amplitudes = np.fft.fft(amplitudes)
    expected_amplitudes /= np.sqrt(np.sum(np.abs(expected_amplitudes)**2))  # Normalize
    # Check if the output is as expected.
    assert np.allclose(np.abs(result.get_tensor_vector()), np.abs(expected_amplitudes), atol=EPSILON), \
        "QFT results do not match expected FFT amplitudes!"
    print("=== QFT Test on a Sine function passed ===")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"=== The runtime of QFT on a sine function is: {elapsed_time:0.2f} seconds ===\n")


def normal_pdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """
    Calculates the probability density of the normal (Gaussian) distribution.

    Parameters
    ----------
    x : np.ndarray
        The input values where the PDF is evaluated.
    mu : float, optional
        The mean of the Gaussian distribution (default is 0).
    sigma : float, optional
        The standard deviation of the Gaussian distribution (default is 1).

    Returns
    -------
    np.ndarray
        The probability density function evaluated at each point in `x`.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def qft_on_gaussian(number_of_qubits: int, mu: float = 0, sigma: float = 1) -> None:
    """
    A Applies the Quantum Fourier Transform (QFT) to a quantum state with amplitudes based on a Gaussian function and compares the resulting amplitudes with the expected amplitudes using numpy's FFT method.
    The function also plots the input states's amplitudes and the measurement plot of the resulting state.

    Parameters
    ----------
    number_of_qubits : int
        The number of qubits in the quantum state.
    mu : float, optional
        The mean of the Gaussian distribution (default is 0).
    sigma : float, optional
        The standard deviation of the Gaussian distribution (default is 1).

    Returns
    -------
    None
    """
    start_time = time.perf_counter()

    # Create normalized state with amplitudes of a Gaussian function.
    x = np.linspace(mu - 100 * sigma, mu + 100 * sigma, 2**number_of_qubits)
    amplitudes = normal_pdf(x, mu, sigma)
    probabilities = np.abs(amplitudes) ** 2
    norm_constant = np.sum(probabilities)
    amplitudes /= np.sqrt(norm_constant)

    mt = MultiQubit(amplitudes)

    # Plot the input state amplitudes.
    mt.plot_amplitudes(plot_type="line")

    # Load the QFT circuit and apply it.
    circuit = QuantumCircuit(number_of_qubits)
    circuit.load_qft_preset()
    result = circuit.run_circuit(mt)

    # Plot the amplitudes after QFT.
    result.plot_probabilities()

    # Compare with expected amplitudes using numpy's FFT.
    expected_amplitudes = np.fft.fft(amplitudes)
    expected_amplitudes /= np.sqrt(np.sum(np.abs(expected_amplitudes)**2))  # Normalize

    assert np.allclose(np.abs(result.get_tensor_vector()), np.abs(expected_amplitudes), atol=EPSILON), \
        "QFT results do not match expected FFT amplitudes!"
    print("=== QFT Test on a Gaussian function passed ===")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"=== The runtime of QFT on a Gaussian distribution is: {elapsed_time:0.2f} seconds ===\n")


def test_qft_matrix_output(qubits_to_test: int = 3) -> None:
    """
    Tests the correctness of the QFT circuit by comparing its matrix representation to the expected QFT matrix.

    Parameters
    ----------
    qubits_to_test : int, optional
        The maximum number of qubits to test (default is 3).

    Returns
    -------
    None
    """
    start_time = time.perf_counter()

    for num_qubits in range(2, qubits_to_test + 1):
        start_time_qft = time.perf_counter()

        # Load the QFT circuit and get its matrix representation.
        circuit = QuantumCircuit(num_qubits)
        circuit.load_qft_preset()
        qft_matrix = circuit.get_circuit_operator_matrix()

        end_time_qft = time.perf_counter()
        qft_elapsed_time = end_time_qft - start_time_qft
        print(f"===== QFT test with {num_qubits} qubits runtime: {qft_elapsed_time:.2f} ======")

        # Compute the expected QFT matrix.
        expected_qft_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                N = 2**num_qubits
                expected_qft_matrix[i, j] = (1/np.sqrt(N)) * np.exp(2j * np.pi * i * j / N)

        assert np.allclose(qft_matrix, expected_qft_matrix, atol=EPSILON), \
            "Circuit QFT matrix does not match expected QFT matrix!"

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"==== Total QFT Test runtime: {elapsed_time:.2f} seconds =======")
    print("=============== QFT tests passed! ===============")

def test_tracing_out_qubit(qubits_to_test: int,print_amplitudes:bool=False) -> None:
    """
    Tests the process of tracing out a qubit in a multi-qubit quantum state by comparing 
    the amplitudes of each qubit before and after the operation. The function generates 
    a random quantum state for multiple qubits, traces out each qubit one by one, and 
    verifies if the original amplitudes are preserved.

    Parameters
    ----------
    qubits_to_test : int
        The number of qubits in the quantum state to test. Each qubit will be initialized
        with random amplitudes, and the test will check that the amplitudes are correctly 
        traced out for each qubit.
    print_amplitudes : bool
        A boolean value to print or not to print the lists for the original and traced amplitudes

    Returns
    -------
    None
        This function does not return any values. It performs an assertion to verify that 
        the traced out qubits' amplitudes match the original amplitudes and prints confirmation.

    Notes
    -----
    The function assumes that the `MultiQubit` class has methods like `add_qubit`, `get_qubit`, 
    and `print_tensor_form` for handling the quantum state, and that each `Qubit` has methods 
    like `get_alpha` and `get_beta` for retrieving the amplitudes of individual qubits.
    
    The test checks that the amplitudes of each qubit remain unchanged after tracing out each qubit 
    by comparing them to the original values stored before the tracing out operation.
    
    Example
    -------
    >>> test_tracing_out_qubit(3)
    Prints the original amplitudes and confirms that tracing out each qubit works correctly.
    """

    q_state = MultiQubit()
    qubits_original_ampitudes = []
    for qubit in range(qubits_to_test):
        alpha = random.uniform(0,1)
        beta = random.uniform(0,1)
        norm_factor = abs(alpha) ** 2 + abs(beta) ** 2
        alpha /= np.sqrt(norm_factor)
        beta /= np.sqrt(norm_factor) 
        q_state.add_qubit(Qubit(alpha,beta))
        qubits_original_ampitudes.append((alpha,beta))
    
    traced_out_amplitudes = []
    for qubit_index in range(qubits_to_test):
        qubit_to_check = q_state.get_qubit(qubit_index)
        alpha = qubit_to_check.get_alpha()
        beta = qubit_to_check.get_beta()
        traced_out_amplitudes.append((alpha,beta))
        
        # Check if the original amplitudes of the qubit are the same as the traced out qubit:
        assert np.isclose(alpha, qubits_original_ampitudes[qubit_index][0], atol=EPSILON)
        assert np.isclose(beta, qubits_original_ampitudes[qubit_index][1], atol=EPSILON)
    
    # Print the amplitudes and traced out amplitudes:
    if print_amplitudes:
        print(f" Original qubits amplitudes {qubits_original_ampitudes}")
        print(f" Traced out qubits amplitudes {traced_out_amplitudes}")

    print("==== The traced out qubits real amplitudes are the same as their original real amplitudes. ==== ")    
    

if __name__ == "__main__":
    # Run tests.
    # qft_on_sine(QUBITS_TO_TEST)
    # qft_on_gaussian(QUBITS_TO_TEST,mu=0,sigma=0.1)
    # test_qft_matrix_output(QUBITS_TO_TEST)
    test_tracing_out_qubit(QUBITS_TO_TEST)
    print("=============== All tests passed! ===============")
