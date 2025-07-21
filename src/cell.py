import numpy as np
import random
from qubit import Qubit
from numpy.typing import NDArray
from c_register import ClassicalRegister
from multi_qubit import MultiQubit
from typing import NewType
from scipy.linalg import expm  # for exponentiation of Pauli generators

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

    def __init__(self) -> None:
        """
        Initialize the Gate object with an identity gate.
        
        The initial configuration represents the identity gate, with no control or target qubits set.
        """
        self.__gate_matrix = None
        self.__gate_type = "I"
        self.__phi = 0.0  # Store phase parameter for conditional gates
        self.__target_qubit = None
        self.__control_qubit = None
        self.__is_control_gate = False
        self.__is_swap_gate = False
        self.__is_measure_gate = False
        self.__is_conditional_gate = False
        self.__is_classical_bit = False

        self.__c_reg = None
        self.__c_reg_index = 0
        # store phase error magnitude for gate
        self.__error_magnitude = 0.0
    
    def set_single_qubit_gate(self, gate_type: str = 'I', phi: float = 0.0, error_magnitude: float = 0.0) -> None:
        """
        Set the gate matrix for a specified single qubit gate type or phase gate.

        :param gate_type: Type of the gate, should be one of ('I', 'X', 'Y', 'Z', 'H', 'P').
        :type gate_type: str
        :param phi: Rotation angle in radians, used only if gate_type is 'P' (phase gate).
        :type phi: float
        :param error_magnitude: Magnitude of phase error noise applied to the gate.
        :type error_magnitude: float
        :raises ValueError: If the gate type is invalid.
        """
        self.__gate_matrix = self.__get_gate_matrix(gate_type, phi, error_magnitude)
        self.__gate_type = gate_type
        self.__error_magnitude = error_magnitude

    def set_controlled_qubit_gate(self, control_qubit: int, target_qubit: int, gate_type: str = 'I', phi: float = 0.0, error_magnitude: float = 0.0) -> None:
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
        :param error_magnitude: Magnitude of phase error noise applied to the gate.
        :type error_magnitude: float
        :raises ValueError: If indices are negative or if the gate type is invalid.
        """
        self.__validate_indices(control_qubit, target_qubit)
        self.__control_qubit, self.__target_qubit = control_qubit, target_qubit
        self.__is_control_gate = True
        self.__gate_matrix = self.__get_gate_matrix(gate_type, phi, error_magnitude)
        self.__gate_type = gate_type
        self.__error_magnitude = error_magnitude

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
        return self.__gate_matrix.copy()
    
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

    def __get_gate_matrix(self, gate_type: str, phi: float=0.0,error_magnitude: float = 0.0) -> NDArray[np.complex128]:
        """
        Generate a single-qubit gate matrix via Pauli exponentials.

        Parameters
        ----------
        gate_type : {'I', 'X', 'Y', 'Z', 'H', 'P'}
        phi : float, optional
            Phase angle in radians for P gate. Default is 0.

        Returns
        -------
        NDArray[np.complex128]
            2×2 unitary matrix implementing the specified gate.
        """
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)

        # Apply phase error based on error magnitude
        noisy_matrix = self.__get_error_matrix(error_magnitude) if error_magnitude > 0 else identity

        # Gate mappings via exponentials
        if gate_type == 'I':
            return identity
        elif gate_type == 'X':
            return expm(-1j * np.pi/2 * sigma_x + noisy_matrix)
        elif gate_type == 'Y':
            return expm(-1j * np.pi/2 * sigma_y + noisy_matrix)
        elif gate_type == 'Z':
            return expm(-1j * np.pi/2 * sigma_z + noisy_matrix)
        elif gate_type == 'H':
            # Hadamard: rotation around (σ_x + σ_z)/√2 by π/2
            return expm(-1j * np.pi/2 * (identity + (sigma_x + sigma_z)/ np.sqrt(2)) + noisy_matrix)
        elif gate_type == 'P':
            # Phase gate P(phi): diag([1, e^{i phi}]) via (I - σ_z)/2 projector
            return expm((1j * phi * (identity - sigma_z) / 2) + noisy_matrix)
        else:
            raise ValueError(f"{INV_SINGLE_GATE_TYP} {', '.join(['I','X','Y','Z','H','P'])}")

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
    
    def set_measure_gate(self,target_qubit:int,c_reg: ClassicalRegister,c_reg_index: int) -> None:
        """
        The method sets this gate to be a measurement gate.

        Parameters
        ----------
        target_qubit : int
            The qubit index on which the measurement is performed.
        c_reg : ClassicalRegister
            The classical register object to which the classical bit will be stored.
        c_reg_int : int
            The index corresponding to the exact classical bit index in the classical register.

        Raises
        ------
        ValueError
            If the provided bit index is invalid. Should be between 0 and number of classical bit in the register
        """
        self.__is_measure_gate = True
        self.__gate_type = 'M'
        # We set the measure gate as an idetity cause we perform the measurement using another method and we do not change the resulting state.
        # Initialize measurement gate as identity with existing error magnitude
        self.__gate_matrix = self.__get_gate_matrix('I', 0.0, 0.0)
        self.__c_reg = c_reg
        self.__c_reg_index = c_reg_index
        self.__target_qubit = target_qubit

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
    
    def set_conditional_gate(self, gate_type: str, phi: float, c_reg: ClassicalRegister, c_reg_index: int, error_magnitude: float = 0.0) -> None:
        """
        This method sets this gate to be a a conditional gate.

        A conditional gate is a gate that acts on one qubit with an unitary operation if the classical bit is one in the specified classical register. Otherwise the gate will act as an ordinary identity gate. 

        Parameters
        ----------
        c_reg : ClassicalRegister
            The classical register object to which the classical bit will be stored.
        c_reg_int : int
            The index corresponding to the exact classical bit index in the classical register.
        error_magnitude: float
            Magnitude of phase error noise applied to the gate.

        """
        self.__gate_matrix = self.__get_gate_matrix(gate_type, phi, error_magnitude)
        self.__c_reg = c_reg
        self.__c_reg_index = c_reg_index
        self.__gate_type = gate_type
        self.__phi = phi  # Store the phase parameter
        self.__is_conditional_gate = True
        self.__error_magnitude = error_magnitude

    def set_classical_bit(self) -> None:
        """
        Set this cell to represent a classical bit.
        """
        self.__is_classical_bit = True
        # We set the classical bit cell to be an idenity matrix because we to not want to change the quantum state if the specified qubit.
        # Initialize classical bit as identity with existing error magnitude
        self.__gate_matrix = self.__get_gate_matrix('I', 0.0, 0.0)
        self.__gate_type = "I"   
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
    
    def measure(self,input_state: MultiQubit) -> MultiQubit:
        """
        The method performs a measurment on the current qubit and saves the collapsed state in the classical register specified in measure gate initialization

        Parameters
        ----------
        input_state : MultiQubit
            The input quantum state on which the measurement will be performed.

        Returns
        -------
        MultiQubit
            The multi qubit quantum state of the whole system. We expect the non measured qubits to be intact.
        
        Raises
        ------
        ValueError
            If the current gate is not a measurement gate.
        """
        if not self.__is_measure_gate:
            raise ValueError("Invalid command. Can not perform a measurement on a non measurment gate.")
        
        collapsed_state ,bit = input_state.measure_qubit(self.__target_qubit)

        # We save the collapsed state of the specific measured qubit in the classical register
        self.__c_reg[self.__c_reg_index] = bit

        return collapsed_state

    def apply_conditional_gate(self) -> None:
        """
        This method updates the gate matrix depending on the value of the classical bit connected to this gate. If the classical bit is 1 we keep the matrix of the specified unitary gate otherwise the gate will not be applied and will act as if a identity gate.  
        """
        bit = self.__c_reg[self.__c_reg_index]

        # If the bit is zero we revert the gate to be an identity gate otherwise the bit is one and we apply the desired unitary.
        if not bit:
            self.__gate_matrix = self.__get_gate_matrix('I', 0.0)
        else:
            # When bit is 1, restore the original gate matrix with the correct phase and error
            self.__gate_matrix = self.__get_gate_matrix(self.__gate_type, self.__phi, self.__error_magnitude)
 
    def __get_error_matrix(self,error_magnitude: float) -> NDArray[np.complex128]:
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        random_error = random.uniform(-error_magnitude, error_magnitude)

        return (-1j * random_error * (sigma_x + sigma_y + sigma_z)) / np.sqrt(3)