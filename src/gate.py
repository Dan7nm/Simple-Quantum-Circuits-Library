import numpy as np
from qubit import Qubit
from numpy.typing import NDArray

INV_SINGLE_GATE_TYP = "Invalid single qubit gate type. Must be one of: "
EPSILON = 1e-10

class Gate:
    """
    Class to represent quantum gate. The user can choose a single qubit gate or a controlled qubit gate
    """
    # Define gate matrices as class constants
    GATE_MATRICES = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    }

    def __init__(self) -> None:
        """
        Initialize the Gate object.

        """
        # Initialize a two-dimensional identity matrix to represent each gate with complex type
        self.__gate_matrix = np.identity(2, dtype=complex)
        # Initialize target and control qubit for the control qates:
        self.__target_qubit = None
        self.__control_qubit = None
        self.__is_control_gate = False
        
        
    def set_single_qubit_gate(self, gate_type: str ='I', phi: float = 0.0) -> None:
        """
        Set the quantum gate matrix based on the specified gate type.
        
        """
        if gate_type not in self.GATE_MATRICES and gate_type != 'P':
            valid_gates = list(self.GATE_MATRICES.keys()) + ['P']
            raise ValueError( f"{INV_SINGLE_GATE_TYP} {', '.join(valid_gates)}")
        if gate_type == 'P':
            self._set_phase_gate(phi)
        else:
            self.__gate_matrix = self.GATE_MATRICES[gate_type].copy()
            
    def _set_phase_gate(self, phi: float) -> None:
        """
        Set the phase gate matrix with the given rotation angle.
        
        Args:
            phi: Rotation angle in radians
        """
        self.__gate_matrix = np.array([
            [1, 0],
            [0, np.exp(1j * phi)]
        ], dtype=complex)
        
    def get_matrix(self) -> NDArray[np.complex128]:
        """
        Get the gate matrix.

        :return: The matrix representation of the gate
        :rtype: numpy.ndarray
        """
        return self.__gate_matrix

    def print_matrix(self):
        """
        Print the gate matrix with proper formatting, removing apostrophes.
        The matrix is displayed with 2 decimal precision, and values close to zero are rounded.
        """
        for row in self.__gate_matrix:
            formatted_row = ['[']
            for elem in row:
                real_val = elem.real
                imag_val = elem.imag
                # If the real part or imaginary part are close to zero, round to zero
                if abs(real_val) < EPSILON:
                    real_val = 0
                if abs(imag_val) < EPSILON:
                    imag_val = 0
                # Print the element directly, formatted as needed
                if imag_val == 0:
                    formatted_row.append(f"{real_val:.2f}")
                elif real_val == 0:
                    formatted_row.append(f"{imag_val:.2f}j")
                else:
                    formatted_row.append(f"{real_val:.2f}{imag_val:+.2f}j")
            formatted_row.append(']')
            # Join the formatted row and print without apostrophes
            print(" ".join(formatted_row))

    def apply_matrix(self, qubit):
        """
        Apply the gate matrix to a qubit and return the resulting qubit.

        :param qubit: The input qubit
        :type qubit: Qubit
        :return: The resulting qubit after applying the gate
        :rtype: Qubit
        """
        qubit_vector = qubit.get_vector()
        result_vector = np.dot(self.__gate_matrix, qubit_vector)
        alpha = result_vector[0]
        beta = result_vector[1]

        result_qubit = Qubit(alpha, beta)
        return result_qubit
