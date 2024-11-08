import numpy as np
from multi_qubit import MultiQubit
from gate import Gate

INV_DIM = "This operator is incompatible with the given qubit tensor."
INV_CTRL = "The given control qubit is invalid. The control qubit should be between 0 and number of qubits -1."
INV_TAR = "The given target qubit is invalid. The target qubit should be between 0 and number of qubits -1."

class Circuit:
    """
    :ivar __number_of_compatible_qubits: The number of qubits this gate operator is compatible with.
    :vartype __number_of_compatible_qubits: int
    :ivar __number_of_layers: The number of layers/iterations in this circuit.
    :vartype __number_of_layers: int
    :ivar __circuit: Initializes numpy array of dimension 1 by number of qubits and initializes all values to "I". This represents one layer without any gates.
    :vartype __circuit: numpy.ndarray

    A class to represent a quantum circuit using quantum gates.

    The Circuit class allows for building a quantum circuit, by adding and removing a quantum gate on every qubit at each vertical and horizontal axis. Every iteration from left to right is described by layers. Each layer is compromised of a tensor product of single qubit gates or controlled gates.

    Steps for circuit building:
    ---------------------------

    1. Initializes a first empty layer without any gates.
    2. Adds single qubit gates and control gates. Build a two dimensional array. Every row of the array represents a layer and every column represents a gate on that particular qubit. If no gate is chosen the identity gate will be the default gate.
    3. Move to the next layer and repeat step 2.
    4. After finishing designing the whole circuit we compute the final matrix. This is done by computing the Krorecker product in each layer and multiplying all the matrices of all layers.
    """

    def __init__(self, multi_qubit: MultiQubit) -> None:
        """
        Initializes an empty circuit object with the right amount of qubits given by the multi_qubit argument.

        :param multi_qubit: The tensor product of qubits given
        :type multi_qubit: MultiQubit 
        :param __number_of_compatible_qubits: Initialized to 0, indicating no qubits are currently compatible.
        :type __number_of_compatible_qubits: int
        :param __number_of_layers: The number of layers/iterations in this circuit.
        :type __number_of_layers: int
        :param __circuit: Initializes numpy array of dimension 1 by number of qubits and initializes all values to "I". This represents one layer without any gates.
        :type __circuit: numpy.ndarray
        """
        self.__number_of_compatible_qubits = multi_qubit.get_number_of_qubits()
        self.__number_of_layers = 1
        self.__circuit= np.full(((1,self.__number_of_compatible_qubits)), Gate())

    def add_single_qubit_gate(self, gate: str, target_qubit: int) -> None:
        """
        Adds a single qubit gate to a specified qubit in the current layer.

        :param gate: A single qubit gate to be added. Represented as a string: ('I', 'X', 'Y', 'Z', 'H', or 'P')
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

        The gate operator is applied to the input qubit tensor, and the result is returned as a new qubit tensor.

        :param qubit_tensor: The qubit tensor to apply the operator on.
        :type qubit_tensor: MultiQubit
        :return: A new MultiQubit that results from applying the operator to the input tensor.
        :rtype: MultiQubit
        :raises ValueError: If the number of qubits in the qubit tensor does not match the number of qubits the gate operator is compatible with.
        """
        number_of_qubits = qubit_tensor.get_number_of_qubits()
        if number_of_qubits != self.__number_of_compatible_qubits:
            raise ValueError(INV_DIM)
        qubit_tensor_vector = qubit_tensor.get_tensor_vector()
        result_vector = np.dot(self.__operator_tensor, qubit_tensor_vector)
        result_qubit_tensor = MultiQubit(result_vector, number_of_qubits)
        return result_qubit_tensor

    def add_controlled_gate(self, control_qubit, target_qubit, qubit_tensor,gate):
        """
        Adds a controlled-U gate to the tensor operator based on the specified control and target qubits.

        The controlled-U gate, denoted as 'CU', is a general construction that applies the single-qubit unitary transformation U on the target qubit only if the control qubit is in the :math:`|1⟩` state. 
        Mathematically, this operation can be represented as:
        
        .. math::

            CU = |0⟩⟨0| \otimes I + |1⟩⟨1| \otimes U
        
        where 'I' is the identity matrix. This means that if the control qubit is in the :math:`|0⟩` state, the target qubit remains unchanged, while if the control qubit is in the :math:`|1⟩` state, the transformation 'U' is applied to the target qubit.

        Example matrix representation for a controlled-U gate:
        
        .. math::
            CU = \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & u_{00} & u_{01} \\\\
                    0 & 0 & u_{10} & u_{11}
            \\end{pmatrix}

        :param control_qubit: The index of the control qubit.
        :type control_qubit: int
        :param target_qubit: The index of the target qubit where the gate will be applied if the control qubit is in the :math:`|1⟩` state.
        :type target_qubit: int
        :param qubit_tensor: The qubit tensor to which the controlled gate will be added.
        :type qubit_tensor: QubitTensor
        :param gate: The single qubit gate to be conditionally applied to the target qubit.
        :type gate: SingleQubitGate
        :raises ValueError: If the control or target qubit indices are invalid.
        
        This method constructs a controlled gate by using projection matrices on the control qubit. The resulting
        operator tensor will represent the control and target qubits, allowing the gate operation to be applied 
        conditionally based on the control qubit’s state.
        """
        num_of_qubits = qubit_tensor.get_number_of_qubits()
        identity_matrix = np.identity(2,dtype=complex)
        proj_00_mat = np.array([[1,0],[0,0]])
        proj_11_mat = np.array([[0,0],[0,1]])

        # Check if valid target and control qubits where given.
        if not 0<= control_qubit < num_of_qubits:
            raise ValueError(INV_CTRL)
        if not 0<= target_qubit < num_of_qubits:
            raise ValueError(INV_TAR)
        
        control_matrix = np.array([1])
        target_matrix = np.array([1])

        gate_matrix = gate.get_matrix()

        for qubit_index in range(num_of_qubits):
            if(control_qubit == qubit_index):
                control_matrix = np.kron(control_matrix,proj_00_mat)
            elif (target_qubit == qubit_index):
                control_matrix = np.kron(control_matrix,identity_matrix)
            else:
                control_matrix = np.kron(control_matrix,identity_matrix)

        for qubit_index in range(num_of_qubits):
            if(control_qubit == qubit_index):
                target_matrix = np.kron(target_matrix,proj_11_mat)
            elif (target_qubit == qubit_index):
                target_matrix = np.kron(target_matrix,gate_matrix)
            else:
                target_matrix = np.kron(target_matrix,identity_matrix)

        self.__number_of_compatible_qubits = num_of_qubits
        self.__operator_tensor = control_matrix + target_matrix

    def get_matrix(self):
        """
        Get the matrix of all gate tensor products.

        :return: The matrix representation of the final operator.
        :rtype: numpy.ndarray
        """

        return self.__operator_tensor