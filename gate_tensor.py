import numpy as np
from qubit_tensor import QubitTensor

INV_DIM = "This operator is incompatible with the given qubit tensor."

class GateTensor:
    """
    A class to represent a gate operator for qubits in tensor form.

    The GateTensor class allows for adding single qubit gates and applying the 
    gate operator on qubit tensors. The operator is stored as a tensor product 
    of individual gate matrices.

    :ivar __operator_tensor: The operator tensor for the gate, represented as a NumPy array.
    :vartype __operator_tensor: np.array
    :ivar __number_of_compatible_qubits: The number of qubits this gate operator is compatible with.
    :vartype __number_of_compatible_qubits: int
    """

    def __init__(self):
        """
        Initializes a GateTensor object with an identity tensor and zero qubits.

        :param __operator_tensor: Initialized as a 1D NumPy array with a single element, `1`.
        :type __operator_tensor: np.ndarray
        :param __number_of_compatible_qubits: Initialized to 0, indicating no qubits are currently compatible.
        :type __number_of_compatible_qubits: int
        """
        self.__operator_tensor = np.ones((1,), dtype=complex)
        self.__number_of_compatible_qubits = 0

    def add_single_qubit_gate(self, gate):
        """
        Adds a single qubit gate to the tensor operator by performing a tensor product
        (Kronecker product) with the existing operator tensor.

        :param gate: A single qubit gate to be added.
        :type gate: SingleQubitGate
        :raises ValueError: If the gate matrix is incompatible with the current operator tensor.
        """
        gate_matrix = gate.get_matrix()
        self.__operator_tensor = np.kron(self.__operator_tensor, gate_matrix)
        self.__number_of_compatible_qubits += 1
    
    def print_matrix(self):
        """
        Prints the current operator tensor matrix to the console.

        This function is useful for debugging and visualizing the gate operator.
        """
        print(self.__operator_tensor)

    def apply_operator(self, qubit_tensor):
        """
        Applies the gate operator (stored in the tensor) to the given qubit tensor.

        The gate operator is applied to the input qubit tensor, and the result is returned
        as a new qubit tensor.

        :param qubit_tensor: The qubit tensor to apply the operator on.
        :type qubit_tensor: QubitTensor
        :return: A new QubitTensor that results from applying the operator to the input tensor.
        :rtype: QubitTensor
        :raises ValueError: If the number of qubits in the qubit tensor does not match 
                            the number of qubits the gate operator is compatible with.
        """
        number_of_qubits = qubit_tensor.get_number_of_qubits()
        if number_of_qubits != self.__number_of_compatible_qubits:
            raise ValueError(INV_DIM)
        qubit_tensor_vector = qubit_tensor.get_tensor_vector()
        result_vector = np.dot(self.__operator_tensor, qubit_tensor_vector)
        result_qubit_tensor = QubitTensor(result_vector, number_of_qubits)
        return result_qubit_tensor
