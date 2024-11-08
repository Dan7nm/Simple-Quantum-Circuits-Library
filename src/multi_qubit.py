import numpy as np
from qubit import Qubit
from typing import List

EPSILON = 1e-10
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

    def __init__(self, vector: List[complex]=[], number_of_qubits: int = 0) -> None:
        """
        Initialize an empty QubitTensor object.

        The tensor product starts with no qubits and an empty tensor vector.
        Qubits can be added using the add_qubit method.

        :param __number_of_qubits: Number of qubits in the tensor product.
        :type __number_of_qubits: int
        :param __tensor_vector: Vector representation of the quantum state.
        :type __tensor_vector: list[complex]
        """
        self.__number_of_qubits = number_of_qubits
        self.__tensor_vector = vector

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
        new_tensor_vec = []
        num_of_elements = self.__number_of_qubits * 2 

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
    
    def get_tensor_vector(self) -> np.ndarray:
        """
        Get the tensor product vector.

        :return: Tensor product vector
        :rtype: np.ndarray
        """
        return self.__tensor_vector
    
    def __tensor_product(self,vec1: np.ndarray,vec2: np.ndarray,num_of_elems: int) -> np.ndarray:
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
