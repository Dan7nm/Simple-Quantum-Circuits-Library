import numpy as np
from qubit import Qubit

INV_GATE_TYP = "Invalid gate type."
EPSILON = 1e-10

class SingleQubitGate:
    """
    Class to represent single-qubit gates and apply them to qubits.
    It provides initialization for common gates like I, X, Y, Z, H and rotation gates.

    :ivar gate_type: Type of gate ('I', 'X', 'Y', 'Z', 'H', or 'P')
    :vartype gate_type: str
    :ivar phi: Angle of rotation (only used for the 'P' gate)
    :vartype phi: float
    """

    def __init__(self, gate_type='I', phi=0.0):
        """
        Initialize the SingleQubitGate object.

        :param gate_type: Type of gate ('I', 'X', 'Y', 'Z', 'H', or 'P')
        :type gate_type: str
        :param phi: Angle of rotation (only used for the 'P' gate)
        :type phi: float
        :raises ValueError: If an invalid gate type is provided
        """
        # Initialize a two-dimensional matrix to represent each gate
        self.__gate_matrix = np.zeros((2, 2), dtype=complex)
        # Dictionary mapping each gate type to its corresponding method
        matrix_type = {
            "I": self.__set_I,
            "X": self.__set_X,
            "Y": self.__set_Y,
            "Z": self.__set_Z,
            "H": self.__set_H,
            "P": self.__set_P
        }
        if gate_type in matrix_type:
            if gate_type == "P":
                matrix_type[gate_type](phi)
            else:
                matrix_type[gate_type]()
        else:
            raise ValueError(INV_GATE_TYP)

    def __complex_to_euler(self, z):
        """
        Convert a complex number to its Euler form, r * e^(iθ).
        
        :param z: Complex number to convert
        :type z: complex
        :return: Magnitude and phase (r, theta)
        :rtype: tuple
        """
        r = abs(z)
        theta = np.angle(z)

        # If the number is really small, round the value
        if r < EPSILON:
            r = 0
        if theta < EPSILON:
            theta = 0
        return r, theta

    def __set_I(self):
        """Set the gate matrix to the identity matrix (I)."""
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = 1

    def __set_X(self):
        """Set the gate matrix to the Pauli-X (NOT) gate."""
        self.__gate_matrix[0][1] = 1
        self.__gate_matrix[1][0] = 1

    def __set_Y(self):
        """Set the gate matrix to the Pauli-Y gate."""
        self.__gate_matrix[0][1] = -1j
        self.__gate_matrix[1][0] = 1j

    def __set_Z(self):
        """Set the gate matrix to the Pauli-Z gate."""
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = -1

    def __set_H(self):
        """Set the gate matrix to the Hadamard gate."""
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[0][1] = 1
        self.__gate_matrix[1][0] = 1
        self.__gate_matrix[1][1] = -1
        self.__gate_matrix *= (1 / np.sqrt(2))

    def __set_P(self, phi):
        """
        Set the gate matrix to a rotation gate with angle phi.

        :param phi: Angle of rotation
        :type phi: float
        """
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = np.exp(1j * phi)

    def get_matrix(self):
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
        first_elem = result_vector[0]
        second_elem = result_vector[1]

        alpha, rel_ph_zero = self.__complex_to_euler(first_elem)
        beta, rel_ph_one = self.__complex_to_euler(second_elem)

        result_qubit = Qubit(alpha, beta, rel_ph_zero, rel_ph_one)
        return result_qubit

# class TwoQubitGate:
#     """
#     Class to represent two-qubit gates and apply them to qubits.
#     It provides initialization for common gates like I, X, Y, Z, H, and rotation gates.

#     :ivar gate_type: Type of gate ('CP', 'CX', 'CZ')
#     :vartype gate_type: str
#     :ivar phi: Angle of rotation (only used for the 'CP' gate)
#     :vartype phi: float
#     """

#     def __init__(self, gate_type='I', phi=0.0):
#         """
#         Initialize the SingleQubitGate object.

#         :param gate_type: Type of gate ('CP', 'CX', 'CZ')
#         :type gate_type: str
#         :param phi: Angle of rotation (only used for the 'CP' gate)
#         :type phi: float
#         :raises ValueError: If an invalid gate type is provided
#         """
#         # Initialize a four-dimensional matrix to represent each gate
#         self.__gate_matrix = np.zeros((4, 4), dtype=complex)
#         # Dictionary mapping each gate type to its corresponding method
#         matrix_type = {
#             "CP": self.__set_CP,
#             "CX": self.__set_CX,
#             "CZ": self.__set_CZ,
#         }
#         if gate_type in matrix_type:
#             if gate_type == "CP":
#                 matrix_type[gate_type](phi)
#             else:
#                 matrix_type[gate_type]()
#         else:
#             raise ValueError(INV_GATE_TYP)
