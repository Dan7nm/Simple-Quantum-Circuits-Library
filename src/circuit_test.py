from c_register import ClassicalRegister
import numpy as np
from circuit import QuantumCircuit
from multi_qubit import MultiQubit
from qubit import Qubit
import time
import random
import matplotlib.pyplot as plt
from typing import Dict
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter

### Constants ###
QUBITS_TO_TEST = 6
NUM_MEASUREMENTS_DELTA = 100
MAX_MEASURE_NUM = 5000
NUM_OF_RUNS = 5000
EPSILON = 1e-10

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
    epsilon = 1e-16

    expected_probs: Dict[str, float] = expected_state.get_probabilities()
    predicted_probs: Dict[str, float] = predicted_state.get_probabilities()
    cross_entropy_value = 0.0

    for state, expected_prob in expected_probs.items():
        predicted_prob = predicted_probs.get(state, 0.0)
        predicted_prob = max(predicted_prob, epsilon)
        cross_entropy_value -= expected_prob * np.log(predicted_prob)

    return cross_entropy_value    

def cmp_states(qubits_num:int,number_of_runs: int):
    rand_state = MultiQubit(qubits_num=qubits_num)
    cross_entropy_lst = []
    run_number_lst = list(range(1, number_of_runs, 100)) 
    for run_number in run_number_lst:
        print(f"Run number: {run_number}")
        measured_state = rand_state.measure_multiple(run_number)
        cross_entropy_val = cross_entropy(rand_state,measured_state)
        cross_entropy_lst.append(cross_entropy_val)

    print(f"Number of runs: {len(run_number_lst)}")

    # Calculate true entropy (cross-entropy of expected state with itself)
    min_entropy_val = cross_entropy(rand_state, rand_state)
    
    plot_cross_entropy(run_number_lst, cross_entropy_lst, qubits_num, min_entropy_val)

def cmp_dynamic_qft(qubits_num:int,number_of_runs: int,step: int) -> None:
    rand_state = MultiQubit(qubits_num=qubits_num)
    classical_reg = ClassicalRegister(num_bits=qubits_num)
    regular_circuit = QuantumCircuit(input_state=rand_state,classical_register=classical_reg)
    dynamic_circuit = QuantumCircuit(input_state=rand_state,classical_register=classical_reg)
    regular_circuit.load_qft_preset()
    dynamic_circuit.load_dynamic_qft_preset()
    cross_entropy_lst = []
    run_number_lst = []
    reg_output_state = regular_circuit.run_circuit()
    for run_number in range(10, number_of_runs, step):
        run_number_lst.append(run_number)
        print(f"Run number: {run_number}")
        dyn_output_state = dynamic_circuit.run_many(run_number)
        cross_entropy_val = cross_entropy(reg_output_state,dyn_output_state)
        cross_entropy_lst.append(cross_entropy_val)

    # Calculate true entropy for the dynamic QFT case
    min_entropy_val = cross_entropy(reg_output_state, reg_output_state)

    plot_cross_entropy(run_number_lst, cross_entropy_lst, qubits_num, min_entropy_val)

    plot_cross_entropy(run_number_lst, cross_entropy_lst, qubits_num, min_entropy_val,log_scale=True)

def plot_cross_entropy(run_number_lst, cross_entropy_lst, qubits_num, min_entropy_val, log_scale: bool = False):
    plt.figure(figsize=(12, 8))
    
    runs = np.array(run_number_lst)
    cross_entropies = np.array(cross_entropy_lst) - min_entropy_val
    
    # Plot measured cross-entropy
    plt.plot(runs, cross_entropies, 'o-', markersize=8, linewidth=2, 
             label='Measured Cross-Entropy', color='blue', alpha=0.8)
    
    # Fit 1/N function: cross_entropy = A/N + min_entropy_val
    def inverse_n_func(n, A):
        return A / n
    
    # Use scipy curve_fit
    popt, _ = curve_fit(inverse_n_func, runs, cross_entropies, p0=[100.0])
    A_coefficient = popt[0]
    
    # Calculate R-squared
    predicted = inverse_n_func(runs, A_coefficient)
    ss_res = np.sum((cross_entropies - predicted) ** 2)
    ss_tot = np.sum((cross_entropies - np.mean(cross_entropies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Plot fitted curve
    runs_smooth = np.linspace(runs[0], runs[-1], 200)
    fitted_curve = inverse_n_func(runs_smooth, A_coefficient)
    plt.plot(runs_smooth, fitted_curve, '--', linewidth=3, color='orange',
             label=f'fit: {A_coefficient:.1f}/N (R^2 = {r_squared:.2f})')
    
    print(f"1/N fit: A = {A_coefficient:.3f}, R² = {r_squared:.4f}")
    
    # Mark minimum achieved entropy
    min_idx = np.argmin(cross_entropies)
    plt.scatter([runs[min_idx]], [cross_entropies[min_idx]], color='green', s=200, 
                label=f'Best: {cross_entropies[min_idx]:.4f} (Run {runs[min_idx]})')
    
    plt.xlabel('Number of Runs', fontsize=12)
    plt.ylabel('Cross-Entropy', fontsize=12)
    plt.title(f'Cross-Entropy Difference vs. Number of Runs ({qubits_num} Qubits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if log_scale:
        plt.xscale('log',base=np.e)
        plt.yscale('log',base=np.e)

        # Round the axes to 2 decimal places
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.savefig(f"cross_entropy_{qubits_num}_qubits_log.png", dpi=200, bbox_inches='tight')
    else:
        plt.savefig(f"cross_entropy_{qubits_num}_qubits.png", dpi=200, bbox_inches='tight')
    plt.show()

def cmp_qft_results_prob_distr(qubits_num: int, number_of_runs: int) -> None:
    """
    Compare regular QFT and dynamic QFT results by overlaying their probability distributions in a single plot.
    
    This function creates a random quantum state, applies both regular and dynamic QFT circuits,
    and plots the resulting probability distributions overlaid on the same graph for easy comparison.
    
    Parameters
    ----------
    qubits_num : int
        The number of qubits for the quantum state and circuits.
    number_of_runs : int
        The number of runs for the dynamic QFT circuit to estimate probabilities.
        
    Returns
    -------
    None
    """
    print(f"Comparing Regular QFT vs Dynamic QFT with {qubits_num} qubits and {number_of_runs} runs...")
    
    # Create a random quantum state
    rand_state = MultiQubit(qubits_num=qubits_num)
    
    # Create classical register for dynamic circuit
    classical_reg = ClassicalRegister(num_bits=qubits_num)
    
    # Set up regular QFT circuit
    regular_circuit = QuantumCircuit(input_state=rand_state, classical_register=classical_reg)
    regular_circuit.load_qft_preset()
    
    # Set up dynamic QFT circuit  
    dynamic_circuit = QuantumCircuit(input_state=rand_state, classical_register=classical_reg)
    dynamic_circuit.load_dynamic_qft_preset()
    
    # Run both circuits
    print("Running regular QFT circuit...")
    reg_output_state = regular_circuit.run_circuit()
    
    print("Running dynamic QFT circuit...")
    dyn_output_state = dynamic_circuit.run_many(number_of_runs)
    
    # Get probability data for plotting
    states_list = [format(state, f"0{qubits_num}b") for state in range(2 ** qubits_num)]
    reg_probs = [abs(amplitude)**2 for amplitude in reg_output_state.get_tensor_vector()]
    dyn_probs = [abs(amplitude)**2 for amplitude in dyn_output_state.get_tensor_vector()]
    
    # Create single overlay plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions for side-by-side comparison
    bar_width = 0.35
    x_pos = np.arange(len(states_list))
    
    # Plot both results with slight offset for better visibility
    bars1 = ax.bar(x_pos - bar_width/2, reg_probs, bar_width, 
                   color='blue', alpha=0.7, label='Regular QFT')
    bars2 = ax.bar(x_pos + bar_width/2, dyn_probs, bar_width, 
                   color='red', alpha=0.7, label=f'Dynamic QFT ({number_of_runs} runs)')
    
    # Customize the plot
    ax.set_xlabel('Quantum States', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'QFT Comparison: Regular vs Dynamic ({qubits_num} Qubits)', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(states_list, rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"qft_comparison_{qubits_num}_qubits_{number_of_runs}_runs.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print similarity metrics
    cross_entropy_val = cross_entropy(reg_output_state, dyn_output_state)
    print(f"Cross-entropy between regular and dynamic QFT: {cross_entropy_val:.6f}")
    
    # Calculate fidelity (overlap between probability distributions)
    fidelity = np.sum(np.sqrt(np.array(reg_probs) * np.array(dyn_probs)))
    print(f"Fidelity between regular and dynamic QFT: {fidelity:.6f}")
    
    print("QFT comparison completed!")

def test_qft_phase_error_cross_entropy(number_of_qubits: int = 5, num_runs: int = 10, 
                                        phi_max: float = 0.5, num_phi_points: int = 20,measurement_num: int = 1000) -> None:
    """
    Test that compares cross entropy between regular QFT (no errors) and QFT with phase errors.
    Plots cross_entropy vs error magnitude phi.

    Parameters
    ----------
    number_of_qubits : int, optional
        Number of qubits for the QFT circuit (default is 5).
    num_runs : int, optional
        Number of runs to average over for each phi value (default is 10).
    phi_max : float, optional
        Maximum phase error magnitude to test (default is 0.5 radians).
    num_phi_points : int, optional
        Number of phi values to test (default is 20).
    meaasurement_num : int, optional
        Number of measurements to perform for each run (default is 1000).

    Returns
    -------
    None
    """
    from cell import QuantumCircuitCell
    
    print(f"Testing QFT Phase Error Cross Entropy ({number_of_qubits} qubits)")
    print("=" * 60)
    
    # Create test phi values from 0 to phi_max
    phi_values = np.linspace(0, phi_max, num_phi_points)
    cross_entropy_values = []
    cross_entropy_std = []
    cross_entropies_values_dyn = []
    cross_entropy_std_dyn = []

    # Get reference QFT result (no errors)
    print("Computing reference QFT result (no phase errors)...")
    QuantumCircuitCell.disable_phase_error()
    input_state = MultiQubit(qubits_num=number_of_qubits)
    reference_circuit = QuantumCircuit(input_state)
    reference_circuit.load_qft_preset()
    reference_result = reference_circuit.run_circuit()
    
    print(f"Testing {num_phi_points} phase error values from 0 to {phi_max:.3f} radians...")
    
    for i, phi in enumerate(phi_values):
        print(f"Progress: {i+1}/{num_phi_points} (phi = {phi:.3f})", end="\r")
        
        # Set phase error magnitude
        if phi == 0:
            QuantumCircuitCell.disable_phase_error()
        else:
            QuantumCircuitCell.set_phase_error(phi)
        
        # Run multiple trials for this phi value
        cross_entropies_for_phi = []
        cross_entropies_for_phi_dyn = []
        
        for run in range(num_runs):
            # Create regular qft circuit
            test_circuit = QuantumCircuit(input_state)
            test_circuit.load_qft_preset()
            test_result = test_circuit.run_circuit()
            sampled_result = test_result.measure_multiple(num_of_measurements=measurement_num)

            # create a dynamic qft circuit
            classical_reg = ClassicalRegister(num_bits=number_of_qubits)
            dynamic_circuit = QuantumCircuit(input_state, classical_register=classical_reg)
            dynamic_circuit.load_dynamic_qft_preset()
            dynamic_result = dynamic_circuit.run_many(num_of_runs=measurement_num)
            
            # Calculate cross entropy between reference and error-affected result
            ce_dyn = cross_entropy(reference_result, dynamic_result)
            ce = cross_entropy(reference_result, sampled_result)
            cross_entropies_for_phi.append(ce)
            cross_entropies_for_phi_dyn.append(ce_dyn)
        
        # Store mean and standard deviation
        cross_entropy_values.append(np.mean(cross_entropies_for_phi))
        cross_entropy_std.append(np.std(cross_entropies_for_phi))

        cross_entropies_values_dyn.append(np.mean(cross_entropies_for_phi_dyn))
        cross_entropy_std_dyn.append(np.std(cross_entropies_for_phi_dyn))
    
    print("\nCompleted all phase error tests.")
    
    # Disable phase errors after testing
    QuantumCircuitCell.disable_phase_error()
    
    # Convert to numpy arrays for plotting
    phi_values = np.array(phi_values)
    cross_entropy_values = np.array(cross_entropy_values)
    cross_entropy_std = np.array(cross_entropy_std)
    cross_entropies_values_dyn = np.array(cross_entropies_values_dyn)
    cross_entropy_std_dyn = np.array(cross_entropy_std_dyn)

    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot regular QFT cross entropy with error bars
    plt.errorbar(phi_values, cross_entropy_values, yerr=cross_entropy_std, 
                fmt='o-', markersize=6, linewidth=2, capsize=5, 
                label=f'Regular QFT Cross-Entropy (avg over {num_runs} runs)', 
                color='blue', alpha=0.8)
    # Plot dynamic QFT cross entropy with error bars
    plt.errorbar(phi_values, cross_entropies_values_dyn, yerr=cross_entropy_std_dyn,
                fmt='s--', markersize=6, linewidth=2, capsize=5,
                label=f'Dynamic QFT Cross-Entropy (avg over {num_runs} runs)',
                color='orange', alpha=0.8)

    plt.xlabel('Phase Error Magnitude φ (radians)', fontsize=12)
    plt.ylabel('Cross Entropy', fontsize=12)
    plt.title(f'Cross Entropy vs Phase Error Magnitude\n'
              f'QFT with {number_of_qubits} qubits, averaged over {num_runs} runs\n'
              f'Measurements of circuit: {measurement_num}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'cross_entropy_vs_phase_error_{number_of_qubits}_qubits_mes_{measurement_num}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Number of qubits: {number_of_qubits}")
    print(f"Runs per phi value: {num_runs}")
    print(f"Phase error range: 0 to {phi_max:.3f} radians")
    print(f"Cross entropy at phi=0: {cross_entropy_values[0]:.6f} ± {cross_entropy_std[0]:.6f}")
    print(f"Cross entropy at phi={phi_max:.3f}: {cross_entropy_values[-1]:.6f} ± {cross_entropy_std[-1]:.6f}")
    print(f"Maximum cross entropy: {np.max(cross_entropy_values):.6f}")
    print("Test completed successfully!")

if __name__ == "__main__":
    
    # Test QFT with phase errors 
    test_qft_phase_error_cross_entropy(number_of_qubits=5, num_runs=10, phi_max=0.3, num_phi_points=15,measurement_num=500)

    print("=============== All tests passed! ===============")
