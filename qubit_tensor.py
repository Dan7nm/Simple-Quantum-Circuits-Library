import numpy as np

class QubitTensor:
    """
    A class representing tensor products of quantum bits (qubits).

    This class implements the tensor product operation for quantum states,
    allowing for the construction and manipulation of multi-qubit systems.
    It maintains both the vector representation and the quantum state notation.

    :ivar __number_of_qubits: Number of qubits in the tensor product
    :type __number_of_qubits: int
    :ivar __qubits: List storing the Qubit objects
    :type __qubits: list[Qubit]
    :ivar __tensor_vector: Vector representation of the quantum state
    :type __tensor_vector: list[complex]

    Example
    -------
    >>> qt = QubitTensor()
    >>> q1 = Qubit(1/np.sqrt(2), 1/np.sqrt(2), 0, 0)  # |+⟩ state
    >>> qt.add_qubit(q1)
    >>> qt.print_tensor_form()
    Tensor product in basis state form:
    0.7071|0⟩ + 0.7071|1⟩
    """

    def __init__(self):
        """
        Initialize an empty QubitTensor object.

        The tensor product starts with no qubits and an empty tensor vector.
        Qubits can be added using the add_qubit method.
        """
        self.__number_of_qubits = 0
        self.__qubits = []
        self.__tensor_vector = []

    def add_qubit(self, new_qubit):
        """
        Add a new qubit to the tensor product and update the state vector.

        This method adds a new qubit to the system and updates the tensor
        product state vector accordingly.

        :param new_qubit: The qubit to be added to the tensor product
        :type new_qubit: Qubit
        :raises TypeError: If new_qubit is not a Qubit object
        
        Example
        -------
        >>> qt = QubitTensor()
        >>> q1 = Qubit(1, 0, 0, 0)  # |0⟩ state
        >>> qt.add_qubit(q1)
        """
        self.__qubits.append(new_qubit)
        self.compute_tensor_vector(new_qubit)
        self.__number_of_qubits += 1

    def compute_tensor_vector(self, new_qubit):
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
        new_tensor_vec = []
        num_of_elements = self.__number_of_qubits * 2 

        zero_state_phase = np.exp(1j * np.pi * new_qubit.get_rel_ph_zero())
        one_state_phase = np.exp(1j * np.pi * new_qubit.get_rel_ph_one())
        first_elem = new_qubit.get_alpha() 
        second_elem = new_qubit.get_beta()

        if(num_of_elements != 0):
            for i in range(num_of_elements):
                new_tensor_vec.append(self.__tensor_vector[i] * first_elem)
                new_tensor_vec.append(self.__tensor_vector[i] * second_elem)
        else:
            new_tensor_vec.append(first_elem)
            new_tensor_vec.append(second_elem)

        self.__tensor_vector = new_tensor_vec

    def print_vector_form(self):
        """
        Print the vector representation of the quantum state.

        Displays the coefficients of the quantum state in the computational
        basis as a list of complex numbers.

        Example
        -------
        >>> qt = QubitTensor()
        >>> q1 = Qubit(1/np.sqrt(2), 1/np.sqrt(2), 0, 0)
        >>> qt.add_qubit(q1)
        >>> qt.print_vector_form()
        The vector of the tensor product is: [0.7071+0j, 0.7071+0j]
        """
        print("The vector of the tensor product is:", self.__tensor_vector)

    def print_tensor_form(self):
        """
        Print the quantum state in Dirac notation.

        Displays the quantum state as a sum of computational basis states
        with their corresponding complex coefficients.

        Format
        ------
        The output follows the form:
        a|000⟩ + b|001⟩ + c|010⟩ + ... + h|111⟩
        where a, b, c, ..., h are complex coefficients

        Note
        ----
        - Terms with zero coefficients are omitted
        - Complex coefficients are displayed in (a+bj) format
        - Real coefficients are displayed without parentheses
        - Coefficient 1 is omitted for clarity

        Example
        -------
        >>> qt = QubitTensor()
        >>> q1 = Qubit(1/np.sqrt(2), 1/np.sqrt(2), 0, 0)
        >>> q2 = Qubit(1, 0, 0, 0)
        >>> qt.add_qubit(q1)
        >>> qt.add_qubit(q2)
        >>> qt.print_tensor_form()
        Tensor product in basis state form:
        0.7071|00⟩ + 0.7071|10⟩
        """
        if self.__number_of_qubits == 0:
            print("Empty tensor product")
            return
            
        tensor_str = ""
        num_states = 2 ** self.__number_of_qubits
        
        for i in range(num_states):
            coeff = self.__tensor_vector[i]
            
            if abs(coeff) < 1e-10:
                continue
                
            if tensor_str and coeff.real >= 0:
                tensor_str += " + "
            elif tensor_str and coeff.real < 0:
                tensor_str += " - "
                coeff = -coeff
                
            basis_state = format(i, f'0{self.__number_of_qubits}b')
            
            if abs(coeff.imag) < 1e-10:
                coeff_str = f"{coeff.real:.4g}"
            elif abs(coeff.real) < 1e-10:
                coeff_str = f"{coeff.imag:.4g}j"
            else:
                coeff_str = f"({coeff.real:.4g}{coeff.imag:+.4g}j)"
                
            if abs(coeff - 1) < 1e-10:
                tensor_str += f"|{basis_state}⟩"
            else:
                tensor_str += f"{coeff_str}|{basis_state}⟩"
            
        if not tensor_str:
            tensor_str = "0"
            
        print("Tensor product in basis state form:")
        print(tensor_str)
    
    def get_number_of_qubits(self):
        """
        Get the number of qubits in the tensor product.

        :return: The number of qubits in the system
        :rtype: int

        Example
        -------
        >>> qt = QubitTensor()
        >>> q1 = Qubit(1, 0, 0, 0)
        >>> qt.add_qubit(q1)
        >>> print(qt.get_number_of_qubits())
        1
        """
        return self.__number_of_qubits