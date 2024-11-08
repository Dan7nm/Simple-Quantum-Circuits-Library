import numpy as np
from gate import SingleQubitGate
from circuit import Circuit

class QFT:
    """
    :ivar __qubit_tensor: The input quantum state to transform
    :vartype __qubit_tensor: MultiQubit
    :ivar __number_of_qubits: Number of qubits in the system
    :vartype __number_of_qubits: int
    :ivar __matrix: The full QFT transformation matrix
    :vartype __matrix: numpy.ndarray
    :ivar __result_qubit_tensor: The result of applying QFT to the input state
    :vartype __result_qubit_tensor: MultiQubit
    
    Quantum Fourier Transform implementation.
    
    This class implements the Quantum Fourier Transform (QFT) algorithm, which is a quantum
    version of the discrete Fourier transform. The QFT is a crucial component in many quantum
    algorithms including Shor's factoring algorithm and quantum phase estimation.

    The QFT transform is defined as:

    .. math::
        |j⟩ → \\frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{i\\frac{2\pi jk}{N}} |k⟩

    where N = 2^n for n qubits.

    Example
    -------
    >>> from qubit import Qubit
    >>> from qubit_tensor import MultiQubit
    >>> # Create a 2-qubit state |00⟩
    >>> q1 = Qubit(1, 0)
    >>> q2 = Qubit(1, 0)
    >>> qt = MultiQubit()
    >>> qt.add_qubit(q1)
    >>> qt.add_qubit(q2)
    >>> # Apply QFT
    >>> qft = QFT(qt)
    >>> result = qft.get_result()
    >>> result.print_tensor_form()

    
    """

    def __init__(self, qubit_tensor):
        """
        Initialize the QFT transform with an input quantum state.

        :param qubit_tensor: The quantum state to transform
        :type qubit_tensor: MultiQubit
        """
        self.__qubit_tensor = qubit_tensor
        self.__number_of_qubits = qubit_tensor.get_number_of_qubits()
        self.__matrix = np.identity(2**self.__number_of_qubits)
        self.__result_qubit_tensor = self.__compute_qft()

    def __compute_qft(self):
        """
        Compute the Quantum Fourier Transform.

        This private method implements the QFT algorithm using the following steps:
        1. Apply Hadamard gates to each qubit
        2. Apply controlled phase rotations
        3. Combine the operations into a complete QFT circuit

        The circuit consists of:
        - Hadamard gates on each qubit
        - Controlled phase rotations (CP) between pairs of qubits
        - The phase for CP gates is calculated as 2π/2^k where k is the distance between qubits

        :return: The quantum state after applying QFT
        :rtype: MultiQubit
        """
        HGate = SingleQubitGate('H')
        IGate = SingleQubitGate('I')

        current_qubit_tensor = self.__qubit_tensor
        for qubit_index in range(self.__number_of_qubits):
            gate_tensor = GateTensor()
            for index in range(self.__number_of_qubits):
                if index == qubit_index:
                    # Apply the hadamard gate:
                    gate_tensor.add_single_qubit_gate(HGate)
                else:
                    # Apply an identity matrix to all other qubits
                    gate_tensor.add_single_qubit_gate(IGate)
            current_qubit_tensor = gate_tensor.apply_operator(current_qubit_tensor)
            self.__matrix = np.matmul(self.__matrix, gate_tensor.get_matrix())
            gate_tensor = GateTensor()
            for gate_index in range(2, self.__number_of_qubits + 1 - qubit_index):
                phase = 2 * np.pi / (2**gate_index)
                PGate = SingleQubitGate('P', phase)
                gate_tensor.add_controlled_gate(gate_index - 1, qubit_index, current_qubit_tensor, PGate)
                current_qubit_tensor = gate_tensor.apply_operator(current_qubit_tensor)
                self.__matrix = np.matmul(self.__matrix, gate_tensor.get_matrix())

        return current_qubit_tensor

    def get_result(self):
        """
        Get the result of the QFT computation.

        :return: The quantum state after applying QFT
        :rtype: MultiQubit
        """
        return self.__result_qubit_tensor

    def print_operator(self):
        """
        Print the complete QFT operator matrix.

        This method prints the matrix representation of the entire QFT operation,
        which is the product of all Hadamard and controlled phase rotation gates
        applied during the transform.

        """
        print(self.__matrix)