import numpy as np
from qubit import Qubit
from numpy.typing import NDArray

# Constants
INV_SINGLE_GATE_TYP = "Invalid single qubit gate type. Must be one of: "
INV_INDEX = "Control and target qubit index should be non-negative."
NOT_CTRL = "The action is invalid because the gate is a non-control gate."
EPSILON = 1e-10

class QuantumCircuitCell:
    """
    A class to represent a quantum circuit cell, with options for single-qubit gates, controlled gates, and swap gates, measurments.
    
    The gate can be specified as a single qubit gate (e.g., I, X, Y, Z, H, SWAP,M), a controlled qubit gate, 
    or a swap gate, allowing for versatile operations on qubits in a quantum system.

    Example
    --------
    >>> gate = QuantumCircuitCell()
    >>> gate.set_controlled_qubit_gate(1, 2, "P", np.pi/2)
    >>> gate.print_matrix()
    >>> print(gate.get_control_index())
    >>> print(gate.get_target_index())
    >>> print(gate.is_control_gate())
    Output:
    [ 1.00 0.00 ]
    [ 0.00 0.00+1.00j ]
    1
    2
    True
    """

    # Define gate matrices as class constants
    __gate_matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    }

    def __init__(self) -> None:
        """
        Initialize the Gate object with an identity gate.
        
        The initial configuration represents the identity gate, with no control or target qubits set.
        """
        self.__gate_matrix = None
        self.__gate_type = "I"
        self.__target_qubit = None
        self.__control_qubit = None
        self.__is_control_gate = False
        self.__is_swap_gate = False
        self.__is_measure_gate = False
        self.__is_conditional_gate = False
        self.__is_classical_bit = False

    def set_single_qubit_gate(self, gate_type: str = 'I', phi: float = 0.0) -> None:
        """
        Set the gate matrix for a specified single qubit gate type or phase gate.

        :param gate_type: Type of the gate, should be one of ('I', 'X', 'Y', 'Z', 'H', 'P').
        :type gate_type: str
        :param phi: Rotation angle in radians, used only if gate_type is 'P' (phase gate).
        :type phi: float
        :raises ValueError: If the gate type is invalid.
        """
        self.__gate_matrix = self.__get_gate_matrix(gate_type, phi)
        self.__gate_type = gate_type

    def set_controlled_qubit_gate(self, control_qubit: int, target_qubit: int, gate_type: str = 'I', phi: float = 0.0) -> None:
        """
        Set up a controlled gate with the specified control and target qubits and gate type.

        :param control_qubit: Index of the control qubit.
        :type control_qubit: int
        :param target_qubit: Index of the target qubit.
        :type target_qubit: int
        :param gate_type: Type of the gate, should be one of ('I', 'X', 'Y', 'Z', 'H', 'P').
        :type gate_type: str
        :param phi: Rotation angle in radians, used only if gate_type is 'P' (phase gate).
        :type phi: float
        :raises ValueError: If indices are negative or if the gate type is invalid.
        """
        self.__validate_indices(control_qubit, target_qubit)
        self.__control_qubit, self.__target_qubit = control_qubit, target_qubit
        self.__is_control_gate = True
        self.__gate_matrix = self.__get_gate_matrix(gate_type, phi)
        self.__gate_type = gate_type

    def set_swap_gate(self, first_qubit: int, second_qubit: int) -> None:
        """
        Configure a swap gate with the specified first and second qubits.

        :param first_qubit: Index of the first qubit to swap with.
        :type first_qubit: int
        :param second_qubit: Index of the second qubit to swap with.
        :type second_qubit: int
        :raises ValueError: If indices are negative.
        """
        self.__validate_indices(first_qubit, second_qubit)
        self.__control_qubit, self.__target_qubit = first_qubit, second_qubit
        self.__is_swap_gate = True
        self.__gate_type = "SWAP"

    def get_matrix(self) -> NDArray[np.complex128]:
        """
        Return the matrix representation of the gate.

        Returns
        -------
        matrix : NDArray[np.complex128]
            The matrix of the current get.
        """
        return self.__gate_matrix
    
    def get_control_index(self) -> int:
        """
        Return the control qubit index if this gate is a control or swap gate.

        :return: The index of the control qubit.
        :rtype: int
        :raises ValueError: If the gate is not a control or swap gate.
        """
        if not self.__is_control_gate and not self.__is_swap_gate: 
            raise ValueError(NOT_CTRL)
        else:
            return self.__control_qubit
        
    def get_target_index(self) -> int:
        """
        Return the target qubit index if this gate is a control or swap gate.

        :return: The index of the target qubit.
        :rtype: int
        :raises ValueError: If the gate is not a control or swap gate.
        """
        if not self.__is_control_gate and not self.__is_swap_gate:
            raise ValueError(NOT_CTRL)
        else:
            return self.__target_qubit
        
    def apply_matrix(self, qubit: Qubit) -> Qubit:
        """
        Apply the gate matrix to a qubit and return the resulting qubit.

        :param qubit: The input qubit to which the gate matrix is applied.
        :type qubit: Qubit
        :return: The resulting qubit after applying the gate.
        :rtype: Qubit
        """
        result_vector = np.dot(self.__gate_matrix, qubit.get_vector())
        return Qubit(*result_vector)

    def print_matrix(self) -> None:
        """
        Print the gate matrix with formatted complex values.
        
        Values close to zero are rounded to zero for better readability.
        """
        for row in self.__gate_matrix:
            formatted_row = ['[']
            for elem in row:
                real_part = 0 if abs(elem.real) < EPSILON else elem.real
                imag_part = 0 if abs(elem.imag) < EPSILON else elem.imag
                formatted_row.append(f"{real_part:.2f}{imag_part:+.2f}j" if imag_part else f"{real_part:.2f}")
            formatted_row.append(']')
            print(" ".join(formatted_row))

    def __get_gate_matrix(self, gate_type: str, phi: float) -> NDArray[np.complex128]:
        """
        Retrieve the gate matrix based on the specified type, including phase gate support.

        :param gate_type: Type of the gate, such as 'I', 'X', 'Y', 'Z', 'H', or 'P' for phase gate.
        :type gate_type: str
        :param phi: Rotation angle in radians, used only if gate_type is 'P' (phase gate).
        :type phi: float
        :return: The matrix for the specified gate type.
        :rtype: NDArray[np.complex128]
        :raises ValueError: If the gate type is invalid.
        """
        if gate_type == 'P':
            return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
        if gate_type not in QuantumCircuitCell.__gate_matrices:
            raise ValueError(f"{INV_SINGLE_GATE_TYP} {', '.join(QuantumCircuitCell.__gate_matrices.keys()) + ['P']}")
        return QuantumCircuitCell.__gate_matrices[gate_type].copy()

    def __validate_indices(self, control_qubit: int, target_qubit: int) -> None:
        """
        Validate that the control and target qubit indices are non-negative.

        :param control_qubit: Index of the control qubit.
        :type control_qubit: int
        :param target_qubit: Index of the target qubit.
        :type target_qubit: int
        :raises ValueError: If any of the indices are negative.
        """
        if control_qubit < 0 or target_qubit < 0:
            raise ValueError(INV_INDEX)

    def is_control_gate(self) -> bool:
        """
        Returns true or false if the gate is a control gate.

        :return: Boolean value if the gate is a control gate or not.
        :rtype: Bool
        """
        return self.__is_control_gate
    
    def is_swap_gate(self) -> bool:
        """
        Returns true or false if the gate is a swap gate.

        :return: Boolean value if the gate is a swap gate or not.
        :rtype: Bool
        """
        return self.__is_swap_gate
    
    def get_gate_type(self) -> str:
        """
        This function returns the gate type of this gate (e.g., I, X, Y, Z, H, SWAP)
        :return: The gate type
        :rtype: str
        """
        return self.__gate_type
    
    def is_single_qubit_gate(self) -> bool:
        """
        This method returns if the gate is a single qubit gate or not.

        :return: Returns boolean value if the gate is a single qubit gate.
        :rtype: bool
        """
        if not self.__is_control_gate and not self.__is_swap_gate:
            return True
        else:
            return False
        
    def is_measure_gate(self) -> bool:
        """
        This methods returns a boolean value if the gate is a measure gate or not.

        Returns
        -------
        return_val : bool
            True if the gate is a measure gate and false otherwiss.
        """
        return self.__is_measure_gate
    
    def set_measure_gate(self) -> None:
        """
        This method set this gate to be a measurement gate.
        """
        self.__is_measure_gate = True
        self.__gate_type = 'M'

    def is_conditional_gate(self) -> bool:
        """
        This methods returns a boolean value if the gate is a conditional gate or not.
        
        A conditional gate is a gate that acts on one qubits with an unitary operation if the classical bit input is one. Other wise the gate is just an identity gate. 

        Returns
        -------
        return_val : bool
            True if the gate is a measure gate and false otherwiss.
        """
        return self.__is_conditional_gate
    
    def set_conditional_gate(self) -> None:
        """
        This method sets this gate to be a a conditional gate.

        A conditional gate is a gate that acts on one qubits with an unitary operation if the classical bit input is one. Other wise the gate is just an identity gate. 
        """
        self.__is_measure_gate = True

    def conditional_gate_input(self,classical_bit:int,gate_type:str,phi:float=0.0) -> None:
        """
        This method recieves an input classical bit. If the input bit is one we apply the desired unitary operation otherwise we don't won't to change the qubit state applying the identity matrix.

        Parameters
        ----------
        classical_bit : int
            The input classical bit.
        gate_type : str
            The desired unitary opertion gate type.
        """
        if classical_bit:
            self.set_single_qubit_gate(gate_type,phi)
        else:
            self.set_single_qubit_gate("I")

    def set_classical_bit(self) -> None:
        """
        Set this cell to represent a classical bit.
        """
        self.__is_classical_bit = True
        self.__gate_matrix = None  # Classical bits don't have a gate matrix
        self.__gate_type = None   # Classical bits don't have a gate type
        self.__is_control_gate = False
        self.__is_swap_gate = False
        self.__is_measure_gate = False
        self.__is_conditional_gate = False

    def is_classical_bit(self) -> bool:
        """
        Check if this cell represents a classical bit.

        Returns
        -------
        return_val : bool
            True if the cell is a classical bit, False otherwise.
        """
        return self.__is_classical_bit