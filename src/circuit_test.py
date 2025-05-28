from c_register import ClassicalRegister
import numpy as np
from circuit import QuantumCircuit
from multi_qubit import MultiQubit
from qubit import Qubit
import time
import random
import matplotlib.pyplot as plt
from typing import Dict

### Constants ###
EPSILON = 1e-16
QUBITS_TO_TEST = 6
NUM_MEASUREMENTS_DELTA = 100
MAX_MEASURE_NUM = 5000
NUM_OF_RUNS = 1000

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
    circuit = QuantumCircuit(mt)
    circuit.load_qft_preset()
    result = circuit.run_circuit()

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
    circuit = QuantumCircuit(mt)
    circuit.load_qft_preset()
    result = circuit.run_circuit()

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
        vector = np.zeros(2**num_qubits)
        vector[0] = 1
        mt = MultiQubit(vector)

        # Load the QFT circuit and get its matrix representation.
        circuit = QuantumCircuit(mt)
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

def cross_entropy(expected_state: MultiQubit ,predicted_state: MultiQubit) -> float:
    """
    Calculate the cross-entropy between two quantum states.

    Parameters
    ----------
    expected_state : MultiQubit
        The MultiQubit object representing the expected probability distribution.
    predicted_state : MultiQubit
        The MultiQubit object representing the predicted probability distribution.

    Returns
    -------
    float
        The cross-entropy value between the two quantum states.
        Returns ``float('inf')`` if the predicted distribution has zero
        probability (or a probability less than EPSILON) for a state with a
        non-zero probability (greater than EPSILON) in the expected distribution.

    Notes
    -----
    The cross-entropy H(p, q) between two probability distributions p and q
    is defined as:

    .. math::
        H(p, q) = - \sum_{x} p(x) \log(q(x))

    where p(x) is the probability of state x in the expected distribution,
    and q(x) is the probability of state x in the predicted distribution.
    A small epsilon value is used to handle potential log(0) errors.

    """
    expected_probs: Dict[str, float] = expected_state.get_probabilities()
    predicted_probs: Dict[str, float] = predicted_state.get_probabilities()
    cross_entropy_value = 0.0

    for state, expected_prob in expected_probs.items():
        if state in predicted_probs:
            predicted_prob = predicted_probs[state]
            if predicted_prob > EPSILON:
                cross_entropy_value -= expected_prob * np.log(predicted_prob)
            elif expected_prob > EPSILON:
                return float('inf') 
        elif expected_prob > EPSILON:
            return float('inf') 

    return cross_entropy_value

def cmp_dynamic_qft(qubits_num:int) -> None:
    rand_state = MultiQubit(qubits_num=qubits_num)
    classical_reg = ClassicalRegister(num_bits=qubits_num)
    regular_circuit = QuantumCircuit(input_state=rand_state,classical_register=classical_reg)
    dynamic_circuit = QuantumCircuit(input_state=rand_state,classical_register=classical_reg)
    regular_circuit.load_qft_preset()
    dynamic_circuit.load_dynamic_qft_preset()
    cross_entropy_lst = []
    run_number_lst = []
    reg_output_state = regular_circuit.run_circuit()
    for run_number in range(10, NUM_OF_RUNS + 1, 50):
        run_number_lst.append(run_number)
        print(run_number)
        dyn_output_state = dynamic_circuit.run_many(run_number)
        cross_entropy_val = cross_entropy(reg_output_state,dyn_output_state)
        cross_entropy_lst.append(cross_entropy_val)

    plot_cross_entropy(run_number_lst,cross_entropy_lst,qubits_num)

def plot_cross_entropy(run_number_lst,cross_entropy_lst,qubits_num):
    plt.figure(figsize=(10, 6))
    plt.plot(run_number_lst, cross_entropy_lst, marker='o')
    plt.xlabel('Number of Runs for Dynamic Circuit')
    plt.ylabel('Cross-Entropy')
    plt.title(f'Cross-Entropy vs. Number of Runs ({qubits_num} Qubits)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cmp_dynamic_qft(3)
    print("=============== All tests passed! ===============")
