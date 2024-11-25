import numpy as np
from qubit import Qubit
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import random

EPSILON = 1e-10

INV_VEC = "The input vector is invalid. Vector should be normalized, meaning the sum of all cells squared should sum to 1."
INV_POS_VAL = "The value is invalid. The value should be a positive non zero integer."

class MultiQubit:
    """
    :ivar __number_of_qubits: Number of qubits in the tensor product.
    :vartype __number_of_qubits: int
    :ivar __tensor_vector: Vector representation of the quantum state.
    :vartype __tensor_vector: list[complex]
    
    A class representing tensor products of many quantum bits (qubits).

    This class implements the tensor product operation for quantum states,
    allowing for the construction and manipulation of multi-qubit systems.
    It maintains both the vector representation and the quantum state notation.

    Example
    -------
    >>> mt = MultiQubit()
    >>> q0 = Qubit(1,0)
    >>> q1 = Qubit(0,1)
    >>> mt.add_qubit(q0)
    >>> mt.add_qubit(q1)
    >>> mt.print_tensor_form()
    Tensor product in basis state form: |01⟩
    """

    def __init__(self, vector: NDArray[np.complex128]=np.array([])) -> None:
        """
        Initialize an empty QubitTensor object.

        The tensor product starts with no qubits and an empty tensor vector.
        Qubits can be added using the add_qubit method.

        :param __tensor_vector: Vector representation of the quantum state.
        :type __tensor_vector: NDArray[np.complex128]
        """
        # Check if the given vector is normalized:
        self.__valid_amplitudes(vector)
        self.__tensor_vector = vector
        vector_len = len(self.__tensor_vector)
        self.__number_of_qubits = int(np.log2(vector_len) if vector_len > 0 else 0)

    def add_qubit(self, new_qubit: Qubit) -> None:
        """
        Add a new qubit to the tensor product and update the state vector.

        This method adds a new qubit to the system and updates the tensor
        product state vector accordingly.

        :param new_qubit: The qubit to be added to the tensor product
        :type new_qubit: Qubit
        :raises TypeError: If new_qubit is not a Qubit object
        
        Example
        -------
        >>> mt = MultiQubit()
        >>> q0 = Qubit(1,0)  # |0⟩ state
        >>> mt.add_qubit(q0)
        """
        self.__compute_tensor_vector(new_qubit)
        self.__number_of_qubits += 1

    def __compute_tensor_vector(self, new_qubit: Qubit) -> None:
        """
        Compute the new tensor product vector after adding a qubit.

        Implements the tensor product computation between the current state
        vector and the new qubit's state vector.

        :param new_qubit: The qubit to be tensored with the current state
        :type new_qubit: Qubit

        Note
        ----
        The computation follows these steps:
        1. Calculate the phases for :math:`|0⟩` and :math:`|1⟩` states
        2. Get the amplitude coefficients (α and β)
        3. Perform the tensor product operation
        """
        num_of_elements = len(self.__tensor_vector)

        new_qubit_vector = new_qubit.get_vector()
        self.__tensor_vector = self.__tensor_product(self.__tensor_vector,new_qubit_vector,num_of_elements)

    def print_vector_form(self) -> None:
        """
        Print the vector representation of the quantum state.

        Displays the coefficients of the quantum state in the computational
        basis as a list of complex numbers.
        """
        print("The vector of the tensor product is:", self.__tensor_vector)

    def print_tensor_form(self) -> None:
        """
        Print the quantum state in Dirac notation.

        Displays the quantum state as a sum of computational basis states
        with their corresponding complex coefficients.
        """
        if self.__number_of_qubits == 0:
            print("Empty tensor product")
            return
            
        tensor_str = ""
        num_states = 2 ** self.__number_of_qubits
        
        for i in range(num_states):
            coeff = self.__tensor_vector[i]
            
            if abs(coeff) < EPSILON:
                continue
                
            if tensor_str and coeff.real >= 0:
                tensor_str += " + "
            elif tensor_str and coeff.real < 0:
                tensor_str += " - "
                coeff = -coeff
                
            basis_state = format(i, f'0{self.__number_of_qubits}b')
            
            if abs(coeff.imag) < EPSILON:
                coeff_str = f"{coeff.real:.4g}"
            elif abs(coeff.real) < EPSILON:
                coeff_str = f"{coeff.imag:.4g}j"
            else:
                coeff_str = f"({coeff.real:.4g}{coeff.imag:+.4g}j)"
                
            if abs(coeff - 1) < EPSILON:
                tensor_str += f"|{basis_state}⟩"
            else:
                tensor_str += f"{coeff_str}|{basis_state}⟩"
            
        if not tensor_str:
            tensor_str = "0"
            
        print("Tensor product in basis state form:", tensor_str)
    
    def get_number_of_qubits(self) -> int:
        """
        Get the number of qubits in the tensor product.

        :return: The number of qubits in the system
        :rtype: int
        """
        return self.__number_of_qubits
    
    def get_tensor_vector(self) -> NDArray[np.complex128]:
        """
        Get the tensor product vector.

        :return: Tensor product vector
        :rtype: np.ndarray
        """
        return self.__tensor_vector
    
    def __tensor_product(self,vec1: NDArray[np.complex128],vec2: NDArray[np.complex128],num_of_elems: int) -> NDArray[np.complex128]:
        """
        Compute the tensor product of two vectors.

        This method calculates the tensor product between `vec1` and `vec2`. We assume that vec is of dimension of 2.

        :param vec1: The first vector to use in the tensor product calculation.
        :type vec1: np.ndarray
        :param vec2: The second vector to use in the tensor product calculation. It should contain exactly two elements.
        :type vec2: np.ndarray
        :param num_of_elems: The number of elements from `vec1` to use in the tensor product calculation. If zero, only `vec2` is used.
        :type num_of_elems: int
        :return: Tensor product vector
        :rtype: np.ndarray
        """
        new_tensor_vec = []
        if(num_of_elems != 0):
            for i in range(num_of_elems):
                new_tensor_vec.append(vec1[i] * vec2[0])
                new_tensor_vec.append(vec1[i] * vec2[1])
        else:
            new_tensor_vec.append(vec2[0])
            new_tensor_vec.append(vec2[1])

        return np.array(new_tensor_vec)


    def __valid_amplitudes(self,vector: NDArray[np.complex128]) -> None:
        """
        This method checks if the input vector respects the normalization condition.

        Parameters
        ----------
        vector : NDArray[np.complex128]
            The input vector to check normalization.

        Raises
        ------
        ValueError
            If the vector doesn't meet the the normalization condition then a ValueError will be raised
        """
        sum = 0
        if len(vector) == 0:
            return 
        for amplitude in vector:
            probability = abs(amplitude) ** 2
            sum += probability
        if not abs(sum - 1) <= EPSILON:
            raise ValueError(INV_VEC)
        
    def __init_state_intervals(self) -> dict:
        """
        This method returns a dictionary with the intervals and it's states.
        Using those intervals we can now to which state the measurement collapsed.

        Returns
        -------
        out : dict
            The dictionary with keys as intervals and it corresponding amplitudes 
        """
        interval_dict = {}
        lower_bound = 0
        upper_bound = 0
        for state_index,amplitude in enumerate(self.__tensor_vector):
            prob = abs(amplitude) ** 2
            # We could have prob that are not zero but close to zero to floating inaccuracies. This should solve this problem.
            if prob > EPSILON:
                upper_bound+=prob
                state = format(state_index,f'0{self.__number_of_qubits}b')
                interval_dict.update({(lower_bound,upper_bound):state})
            lower_bound = upper_bound
        
        return interval_dict

    def measure(self) -> str:
        """
        This function measures the state one time and returns the state to which the mesurement collapsed to.

        Returns
        -------
        out : str
           The state to which the measurement collapsed to.
        """
        interval_dict = self.__init_state_intervals()
        rand_num = random.uniform(0,1)
        for interval,state in interval_dict.items():
            if interval[0]<= rand_num <= interval[1]:
                return state

    def plot_measurements(self,num_of_measurements: int = 10000) -> None:
        """
        This function measures a specified number of times. Then plots in a graph the number of times we got each state divided by the number of measurements thus resulting in the probabilities.

        Parameters
        ----------
        number_of_measurements : int
            The number of times to measure the entire circuit.
        """
        self.__valid_pos_val(value= num_of_measurements)

        # Initialize dictionary to stores the states and number of times we collapsed to that state.
        states_dict = {format(state_index, f"0{self.__number_of_qubits}b"): 0
               for state_index in range(2**self.__number_of_qubits)}
    
        for measurements in range(num_of_measurements):
            collapsed_state = self.measure()
            states_dict[collapsed_state] += 1
        
        states_list = list(states_dict.keys())
        probs_list = np.array(list(states_dict.values())) / num_of_measurements

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(states_list, probs_list, color='blue', alpha=0.7)

        # Add labels and title
        plt.xlabel('Quantum States', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Probability Distribution of Measured Quantum States\n(Number of Measurments: {num_of_measurements})', fontsize=14)

        # Show grid and the plot
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    def __valid_pos_val(self,value:int) -> None:
        """
        This method check if the argument is non zero positive integer.
        Parameters
        ----------
        values : int 
            The value to check
        
        Raises
        ------
        ValueError
            Raises a value error if the the given value is not a positive integer 
        """
        if not isinstance(value,int) and value < 1:
            raise ValueError(INV_POS_VAL)
    
    def plot_probabilities(self) -> None:
        """
        This function plots directly the probabilities for the the state with measuring.
        """
        states_list = [format(state,f"0{self.__number_of_qubits}b") for state in range(2 ** self.__number_of_qubits)]
        probs_list = [abs(amplitude)**2 for amplitude in self.__tensor_vector]
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(states_list, probs_list, color='blue', alpha=0.7)

        # Add labels and title
        plt.xlabel('Quantum States', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Probability Distribution of a Quantum State\n', fontsize=14)

        # Show grid and the plot
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()


    def plot_amplitudes(self) -> None:
        """
        This function plots directly the probabilities for the the state with measuring.
        """
        states_list = [format(state,f"0{self.__number_of_qubits}b") for state in range(2 ** self.__number_of_qubits)]
        amplitude_list = [amplitude for amplitude in self.__tensor_vector]
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(states_list, amplitude_list, color='blue', alpha=0.7)

        # Add labels and title
        plt.xlabel('Quantum States', fontsize=12)
        plt.ylabel('Amplitudes', fontsize=12)
        plt.title(f'Amplitudes Distribution of a Quantum State\n', fontsize=14)

        # Show grid and the plot
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()