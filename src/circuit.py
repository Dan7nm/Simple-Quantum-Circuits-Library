# Imports:
from cell import QuantumCircuitCell
from multi_qubit import MultiQubit
import numpy as np
from numpy.typing import NDArray
from typing import Set,Any,Tuple
import torch
import matplotlib.pyplot as plt
from c_register import ClassicalRegister

# Constants
INV_QUBIT_INDEX = "The qubit index is invalid. The qubit index should be between 0 and number of qubits - 1."
INV_LAYER_INDEX = "The layer index is invalid. The layer index should be an integer between 0 and number of layers - 1"
CELL_TAKEN = "This cell already has a gate assigned to it. Use the remove gate method to remove a gate."
INV_INPUT = "The input state has a different number of qubits to that of this quantum circuit."
COMP_CIRCUIT = "The circuit was not computed. Before applying a state you should compute the circuit and after every update to a gate."
INV_NUM_QUBITS = "The number of qubits should be atleast 1."
INV_CTRL_TARG = "Invalid target and qubit index input. The target qubit and the control qubit should be different from each other"
INV_INIT_LAYERS = "Invalid number of layers. The number should be non zero positive integer."
INV_POS_VAL = "The value is invalid. The value should be a positive non zero integer."
INV_FILENAME = "The filename is invalid. The filename should be a string."
INV_Q_STATE = "The input is invalid. The input should be a multiqubit object."
INV_C_REG = "The input is invalid. The input should be a classical register object."
INV_NUM_C_REG = "Invalid Number of bits in the classical register."

class QuantumCircuit:
    """
    A class to represent a quantum circuit using quantum gates.

    The QuantumCircuit class allows for building a quantum circuit, by adding and removing a quantum gate on every qubit at each vertical and horizontal axis. Every iteration from left to right is described by layers. Each layer is compromised of a tensor product of single qubit gates or controlled gates.

    Steps for circuit building:
    1. Initializes a first empty layer without any gates.
    2. Adds single qubit gates and control gates. Build a two dimensional array. Every row of the array represents a layer and every column represents a gate on that particular qubit. If no gate is chosen the identity gate will be the default gate.
    3. Move to the next layer and repeat step 2.
    4. After finishing designing the whole circuit we compute the final matrix. This is done by computing the Krorecker product in each layer and multiplying all the matrices of all layers.

    
    Parameters
    ----------
    input_state : MultiQubit
        The initial quantum state of the system.
    classical_register : ClassicalRegister, optional
        A classical register to store measurement results. Defaults to `None`.
    num_of_layers : int, optional
        The number of layers in the quantum circuit. Defaults to 1.
    device : optional
        The computation device to run the circuit on (e.g., 'cpu', 'gpu'). Defaults to `None`.
    error_single_gate : float, optional
        The error magnitude for single qubit gates. Defaults to 0.0.
    error_control_gate : float, optional
        The error magnitude for controlled gates. Defaults to 0.0.

    Attributes
    ----------
    quantum_state : MultiQubit
        The quantum state of the circuit represented as a MultiQubit object.
    classical_register : ClassicalRegister
        The classical register associated with the circuit, represented as a ClassicalRegister object.   
    num_of_layers : int, optional
        The number of layers in the circuit. Default is 1. 
    device : torch.device, optional
        The device to be used for computation. Defaults to 'cuda' if available, otherwise 'cpu'.  
    circuit_qubit_num : int
        The number of qubits in the quantum state.  
    number_of_layers : int
        The current number of layers in the circuit.  
    circuit : numpy.ndarray
        The circuit represented as an empty array with shape (0, number_of_qubits), which stores the gates applied to the quantum state.   
    circuit_operator : numpy.ndarray
        The matrix representing the circuit operator, initialized to the identity matrix of size 2^n, where n is the number of qubits.
    circuit_is_computed : bool
        A flag indicating whether the circuit has been computed (False by default).
    is_dynamic : bool
        A flag indicating whether the circuit is dynamic (False by default).
    error_single_gate : float
        The error magnitude for single qubit gates, used to simulate noise in the circuit.
    error_control_gate : float
        The error magnitude for controlled gates, used to simulate noise in the circuit.

    """
    def __init__(self, input_state: MultiQubit, classical_register: ClassicalRegister= None, num_of_layers: int = 1, device= None, error_single_gate: float = 0.0, error_control_gate:float = 0.0) -> None:
        self.__device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.__quantum_state = input_state
        if classical_register is not None and not isinstance(classical_register,ClassicalRegister):
            raise ValueError(INV_C_REG)
        self.__classical_register = classical_register

        self.__circuit_qubit_num = input_state.get_number_of_qubits()
        self.__number_of_layers = 0

        self.__valid_pos_val(num_of_layers)

        self.__circuit= np.empty((0, self.__circuit_qubit_num),dtype=QuantumCircuitCell)

        for i in range(0,num_of_layers):
            self.add_layer()

        self.__circuit_operator = np.identity(2 ** self.__circuit_qubit_num,dtype=np.complex128)

        self.__circuit_is_computed = False

        self.__error_single_gate = error_single_gate

        self.__error_control_gate = error_control_gate

        self.__is_dynamic = False
    
    def add_single_qubit_gate(self, target_qubit: int, layer_index: int, gate_type: str, phi: float = 0.0) -> None:
        """
        Add a single-qubit gate to the circuit.

        Parameters
        ----------
        target_qubit : int
            Index of the qubit to which the gate is applied.
        layer_index : int
            Index of the layer where the gate should be added.
        gate_type : str
            Type of quantum gate to apply. Supported types include 'X', 'Y', 'Z', 'H', and 'P'.
        phi : float, optional
            Phase angle for phase gates. Default is 0.0.

        Raises
        ------
        ValueError
            If the qubit index or layer index is invalid, or if the target cell is already occupied.
        """
        gate_type = gate_type.upper()

        # Check if layer and qubit indexes are valid and the cell is empty
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit,layer_index)
        gate = QuantumCircuitCell()
        gate.set_single_qubit_gate(gate_type, phi, self.__error_single_gate)
        self.__circuit[layer_index][target_qubit] = gate

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def add_controlled_qubit_gate(self, target_qubit: int, layer_index: int,  control_qubit: int, gate_type: str, phi: float = 0.0) -> None:
        """
        Add a controlled quantum gate to the circuit.

        :param target_qubit: Index of the target qubit
        :type target_qubit: int
        :param layer_index: Index of the layer where the gate should be added
        :type layer_index: int
        :param control_qubit: Index of the control qubit
        :type control_qubit: int
        :param gate_type: Type of quantum gate ('X', 'Y', 'Z', 'H', 'P')
        :type gate_type: str
        :param phi: Phase angle for phase gates (default: 0.0)
        :type phi: float, optional
        :raises ValueError: If qubit indices or layer index is invalid, or if cells are occupied
        """
        gate_type = gate_type.upper()
        
        # Check that the control qubit and target qubits are different:
        if target_qubit == control_qubit:
            raise ValueError(INV_CTRL_TARG)
        # Check if layer and qubit indexes are valid and the cell is empty
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit, layer_index)
        self.__valid_qubit_index(control_qubit, layer_index)
        gate = QuantumCircuitCell()
        gate.set_controlled_qubit_gate(control_qubit, target_qubit, gate_type, phi, self.__error_control_gate)
        self.__circuit[layer_index][target_qubit] = gate
        self.__circuit[layer_index][control_qubit] = gate

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False


    def add_swap_gate(self,first_qubit: int, second_qubit: int ,layer_index: int) -> None:
        """
        Add a SWAP gate between two qubits in the circuit.

        :param first_qubit: Index of the first qubit to swap
        :type first_qubit: int
        :param second_qubit: Index of the second qubit to swap
        :type second_qubit: int
        :param layer_index: Index of the layer where the gate should be added
        :type layer_index: int
        :raises ValueError: If qubit indices or layer index is invalid, or if cells are occupied
        """
        # Check that the first qubit and second qubits are different:
        if first_qubit == second_qubit:
            raise ValueError(INV_CTRL_TARG)
        # Check if layer and qubit indexes are valid and the cell is empty.
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(first_qubit, layer_index)
        self.__valid_qubit_index(second_qubit, layer_index)
        gate = QuantumCircuitCell()
        gate.set_swap_gate(first_qubit,second_qubit)
        self.__circuit[layer_index][first_qubit] = gate
        self.__circuit[layer_index][second_qubit] = gate

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def remove_single_qubit_gate(self, qubit_index: int, layer_index: int) -> None:
        """
        Remove a single qubit gate from the circuit.

        :param qubit_index: Index of the qubit to remove the gate from
        :type qubit_index: int
        :param layer_index: Index of the layer where the gate should be removed
        :type layer_index: int
        :raises ValueError: If qubit index or layer index is invalid
        """
        # Check if layer and qubit indexes are valid and the cell is empty.
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(qubit_index,layer_index, adding_gate = False)
        self.__circuit[layer_index][qubit_index] = None

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def remove_two_qubit_gate(self, target_qubit: int, control_qubit: int, layer_index: int) -> None:
        """
        Remove a two-qubit gate (controlled or SWAP) from the circuit.

        :param target_qubit: Index of the target qubit
        :type target_qubit: int
        :param control_qubit: Index of the control qubit
        :type control_qubit: int
        :param layer_index: Index of the layer where the gate should be removed
        :type layer_index: int
        :raises ValueError: If qubit indices or layer index is invalid
        """
        # Check that the control qubit and target qubits are different:
        if target_qubit == control_qubit:
            raise ValueError(INV_CTRL_TARG)
        # Check if layer and qubit indexes are valid and the cell is empty. 
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit,layer_index, adding_gate = False)
        self.__valid_qubit_index(control_qubit,layer_index, adding_gate = False)

        self.__circuit[layer_index][target_qubit] = None
        self.__circuit[layer_index][control_qubit] = None

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def add_layer(self, layer_index: int = None) -> None:
        """
        Adds a new empty layer to the circuit.

        Adds to the end by default if no index is given.
        The new layer is initialized with None values (identity gates).

        Parameters
        ----------
        layer_index : int, optional
            The index to which a layer will be added.
        """

        self.__number_of_layers += 1
        new_row = np.full((1, self.__circuit_qubit_num), None)

        if layer_index is None:
            self.__circuit = np.vstack((self.__circuit, new_row))
            insertion_index = self.__number_of_layers - 2  # Index of the previous layer
        else:
            self.__valid_layer_index(layer_index)  # Validate index *before* insertion
            self.__circuit = np.insert(self.__circuit, layer_index, new_row, axis=0)
            insertion_index = layer_index - 1

        # Add classical bits if there is a measurement gate or a classical bit in the previous layer.
        if self.__number_of_layers > 1:
            for qubit_index in range(self.__circuit_qubit_num):
                cell = self.__circuit[insertion_index][qubit_index]
                if cell is not None and (cell.is_measure_gate() or cell.is_classical_bit()):
                    c_bit_cell = QuantumCircuitCell()  # Avoid recreating if already a classical bit
                    c_bit_cell.set_classical_bit()
                    if layer_index is None:
                        self.__circuit[self.__number_of_layers - 1][qubit_index] = c_bit_cell
                    else:
                        self.__circuit[layer_index][qubit_index] = c_bit_cell

        self.__circuit_is_computed = False

    def remove_layer(self,layer_index: int = None) -> None:
        """
        Remove the a specified layer from the circuit. If no layer was specified than the last layer is removed by default.
        
        Parameters
        ----------
        layer_index : int
            The index to which a layer will be removed.
        """
        # Check if the layer index is valid
        if layer_index is not None:
            self.__valid_layer_index(layer_index=layer_index)
        
        if layer_index is None:
            self.__circuit = self.__circuit[:-1]
        else:
            self.__circuit = np.delete(self.__circuit,layer_index,axis=0)

        self.__number_of_layers -= 1

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

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

    def __valid_qubit_index(self,qubit_index: int,layer_index: int, adding_gate: bool = True) -> None:
        """
        Remove the last layer from the circuit.
        
        Reduces the number of layers by one and removes all gates in the last layer.
        """
        # Check for valid indexes
        if not isinstance(qubit_index,int) and not isinstance(layer_index) and not 0 <= qubit_index <= self.__circuit_qubit_num - 1:
            raise ValueError(INV_QUBIT_INDEX)
        # Check this step only if we are adding a gate. In removal this step is not used.
        # Check if the cell is an empty to add a gate
        if adding_gate and self.__circuit[layer_index][qubit_index] is not None:
            raise ValueError(CELL_TAKEN)
        
    def __valid_layer_index(self, layer_index) -> None:
        """
        Validate a layer index. Check if the given index is non negative and an integer.

        :param layer_index: Index of the layer to validate
        :type layer_index: int
        :raises ValueError: If layer index is invalid
        """
        if not isinstance(layer_index,int) and not 0 <= layer_index <= self.__number_of_layers - 1:
            raise ValueError(INV_LAYER_INDEX)
        
    def __fill_identity_gates(self) -> None:
        """
        Fill all empty cells in the circuit with identity gates.

        This method replaces all None values in the circuit with identity gates,
        which are necessary for computing the tensor products correctly.
        """
        for layer_index, layer in enumerate(self.__circuit):
            for qubit_index, cell in enumerate(layer):
                # Only replace cells that are None
                if cell is None:  
                    identity_gate = QuantumCircuitCell()
                    identity_gate.set_single_qubit_gate("I")
                    self.__circuit[layer_index][qubit_index] = identity_gate

    def __compute_single_qubit_gates(self, layer_gates: NDArray) -> torch.Tensor:
        """
        This method  computes all single qubit gates in one layer.

        :param layer_gates: An array of all gates in the layer we want to compute.
        :type layer_gates: NDArray
        :return: The method returns the unitary matrix of that layer
        :rtype: Tensor
        """
        result_matrix = torch.tensor([1],dtype=torch.complex128,device=self.__device)
        identity_matrix = torch.eye(2,dtype=torch.complex128,device=self.__device)

        for gate in layer_gates:
            # Check if current gate is a single qubit gate.
            if gate.is_single_qubit_gate():
                # If conditional gate we want to update the gate matrix depending on the classical register bit.
                if gate.is_conditional_gate():
                    gate.apply_conditional_gate()
                gate_matrix = torch.tensor(gate.get_matrix(),dtype=torch.complex128, device=self.__device)

                # Compute the kronecker product of all single qubit gates
                result_matrix = torch.kron(result_matrix,gate_matrix)
            else:
                # If the gate is not a single qubit gate we compute the kronecker product with the identity matrix.
                result_matrix = torch.kron(result_matrix,identity_matrix)
        
        return result_matrix

    def __swap_gate_matrix(self,first_index: int, second_index: int) -> torch.Tensor:
        """
        Generate the matrix representation of a SWAP gate between first_index and second qubit index in an n-qubit system.
        
        :param first_index: Index of the first qubit (0-based indexing)
        :type first_index: int
        :param second_index: Index of the second qubit (0-based indexing)
        :type second_index: int

        :return: 2^n x 2^n matrix representing the SWAP operation
        :rtype: Tensor
        """
        # Calculate matrix dimension
        dim = 2 ** self.__circuit_qubit_num
        
        # Initialize the matrix with zeros
        swap_matrix = torch.zeros((dim,dim),dtype=torch.complex128,device=self.__device)
        
        # Generate all possible basis states
        for state in range(dim):
            # Convert state to binary representation
            binary = format(state, f'0{self.__circuit_qubit_num}b')
            bits = list(binary)
            
            # Swap the bits at positions i and j
            bits[first_index], bits[second_index] = bits[second_index], bits[first_index]
            
            # Convert back to decimal
            new_state = int(''.join(bits), 2)
            
            # Set the matrix element
            swap_matrix[new_state, state] = 1
            
        return swap_matrix

    def __compute_swap_gates(self,layer_gates: NDArray) -> NDArray:
        """
        This method  computes all swap gates in one layer.

        :param layer_gates: An array of all gates in the layer we want to compute.
        :type layer_gates: NDArray
        :return: The method returns the unitary matrix of that layer
        :rtype: NDArray
        """
        result_matrix = torch.eye(2 ** self.__circuit_qubit_num,dtype=torch.complex128,device=self.__device)

        processed_qubits = set()

        for qubit_index,gate in enumerate(layer_gates):
            # Check if we already processed this qubits:
            if qubit_index in processed_qubits:
                continue
            # IF the gate is a control gate we compute the swap matrix and multiplt with result matrix
            if gate.is_swap_gate():
                first_index = gate.get_control_index()
                second_index = gate.get_target_index()
                processed_qubits.update({first_index,second_index})

                swap_matrix = self.__swap_gate_matrix(first_index=first_index,second_index=second_index)

                # Multiply all the matrices of all swap gates
                result_matrix = torch.matmul(result_matrix,swap_matrix)
        
        return result_matrix

    def __create_controlled_operation(
        self,
        gate: QuantumCircuitCell,
        control_index: int,
        target_index: int
    ) -> torch.Tensor:
        """
        Create a controlled operation matrix for the given gate.
        
        Args:
            gate: The controlled gate to process
            is_control_first: Whether the control qubit comes before the target
            
        Returns:
            The resulting controlled operation matrix
        """
        index_diff = abs(control_index - target_index) - 1
        identity_mid = torch.eye(2 ** index_diff,dtype=torch.complex128,device=self.__device)
        proj_00_matrix = torch.tensor([[1,0],[0,0]], dtype=torch.complex128,device=self.__device)
        proj_11_matrix = torch.tensor([[0,0],[0,1]], dtype=torch.complex128,device=self.__device)
        gate_matrix = torch.tensor(gate.get_matrix(),dtype=torch.complex128, device=self.__device)
        identity_matrix = torch.eye(2,dtype=torch.complex128,device=self.__device)
        
        # Check if control index is before target index
        if control_index < target_index:
            first_term = torch.kron(torch.kron(proj_00_matrix, identity_mid), identity_matrix)
            second_term = torch.kron(torch.kron(proj_11_matrix, identity_mid), gate_matrix)
        else:
            first_term = torch.kron(torch.kron(identity_matrix, identity_mid), proj_00_matrix)
            second_term = torch.kron(torch.kron(gate_matrix, identity_mid),proj_11_matrix)
            
        return first_term + second_term

    def __embed_in_circuit(self, gate: QuantumCircuitCell, operation: NDArray, control_index: int, target_index: int) -> torch.Tensor:
        """
        Embed the controlled operation in the full circuit space.
        
        Args:
            gate: The controlled gate being processed
            operation: The controlled operation matrix to embed
            
        Returns:
            The embedded operation matrix
        """
        lower_idx = min(control_index, target_index)
        higher_idx = max(control_index, target_index)
        
        # Create identity matrices for unused qubits
        identity_prev = torch.eye(2 ** lower_idx,dtype=torch.complex128,device=self.__device)
        identity_after = torch.eye(2 ** (self.__circuit_qubit_num - higher_idx - 1),dtype=torch.complex128,device=self.__device)
        
        # Embed the operation in the full circuit space
        return torch.kron(torch.kron(identity_prev, operation), identity_after)

    def __compute_controlled_gates(self, layer_gates: NDArray) -> torch.Tensor:
        """
        Compute all controlled gates in one layer.
        
        Parameters
        ----------
        layer_gates: NDArray
            List of controlled gates in the layer.
            
        Returns
        -------
        Tensor    
            The unitary matrix representing the complete layer
        """
        processed_qubits: Set[int] = set()
        result_matrix = torch.eye(2 ** self.__circuit_qubit_num,dtype=torch.complex128,device=self.__device)
        
        for qubit_index,gate in enumerate(layer_gates):
            if qubit_index in processed_qubits:
                continue
            if gate.is_control_gate():
                control_index = gate.get_control_index()
                target_index = gate.get_target_index()
                # Mark qubits as processed
                processed_qubits.update({control_index, target_index})
                
                # Create and embed the controlled operation
                controlled_op = self.__create_controlled_operation(gate, control_index,target_index)
                embedded_op = self.__embed_in_circuit(gate, controlled_op,control_index,target_index)
                
                # Update the result matrix
                result_matrix = torch.matmul(result_matrix,embedded_op)
            
        return result_matrix

    def __compute_layer(self,input_state:MultiQubit,layer_index: int) -> Tuple[torch.Tensor,MultiQubit]:
        """
        Compute the combined operation of all gates in a single layer.

        Parameters
        ----------
        input_state : MultiQubit
            The input state on which the layer acts upon.
        layer_index : int
            Index of the layer to compute.
        
        Returns
        -------
        Tuple[torch.Tensor,MultiQubit]
            A tuple of a Matrix representing the combined operation of all gates in the layer and the resulting MultiQubit.
        """
        # Get all gates in the current layer
        layer_gates = self.__circuit[layer_index]

        # Check for measurement gates and peform a measurement if needed:
        input_state = self.__compute_measurement_gates(input_state,layer_gates)

        result_matrix = torch.eye(2 ** self.__circuit_qubit_num, dtype= torch.complex128,device=self.__device)
        
        # Compute all single qubit matrices:
        single_qubit_gates_matrix = self.__compute_single_qubit_gates(layer_gates)

        # Compute all controlled gates:
        controlled_gates_matrix = self.__compute_controlled_gates(layer_gates)

        # Compute all swap gates in this layer:
        swap_gates_matrices = self.__compute_swap_gates(layer_gates)

        # Multiply all the computed matrices
        result_matrix = result_matrix @ single_qubit_gates_matrix @ controlled_gates_matrix @ swap_gates_matrices

        # Multiply the input state on the resulting matrix (ensure same device/dtype):
        input_state_tensor = (
            torch.from_numpy(input_state.get_tensor_vector())
            .to(dtype=torch.complex128, device=self.__device)
        )
        result_tensor = result_matrix @ input_state_tensor
        result_state = MultiQubit(result_tensor.cpu().numpy())
             
        return result_matrix,result_state
        
    def __compute_circuit(self,layer_index: int = None) -> MultiQubit:
        """
        The method computes the circuit. If the circuit is dynamic we apply the operator of each layer on the previous layer quantum state to the desired layer index. By default we run the whole circuit. If the circuit is non dynamic we compute the whole circuit unitary matrix and save for later use if we want to rerun the circuit on a different state.

        This method:
        1. Fills empty cells with identity gates
        2. Computes the each layer of the circuit.
        3. Multiplies all layer matrices to get the final circuit operator or applies each layer on the input state if the circuit is dynamic.

        Parameters
        ----------
        layer_index: int
            The layer to index indicating to which layer we want to compute. None by default indicating to compute the whole circuit.

        Returns
        -------
        MultiQubit
            THe computed quantum state.

        Raises
        ------
        ValueError
            If the layer index is not valid
        """
        layers_to_compute = (self.__number_of_layers if layer_index is None else layer_index + 1)
        # Check if the layer index is valid:
        self.__valid_layer_index(layers_to_compute - 1)
        # First fill all empty cells with identity gates
        self.__fill_identity_gates()

        computed_layers = torch.eye(2 ** self.__circuit_qubit_num,dtype=torch.complex128,device=self.__device)
        # Multiply the matrices of each layer to compute the final matrix of the whole circuit.
        current_state = self.__quantum_state
        for layer_index in range(layers_to_compute):
            layer_matrix,current_state = self.__compute_layer(current_state,layer_index)
            computed_layers = torch.matmul(computed_layers,layer_matrix)

        self.__circuit_operator = computed_layers.cpu().numpy()

        # Update that the circuit was computed only if we computed all layers:
        if layer_index is None or layers_to_compute == self.__number_of_layers:
            self.__circuit_is_computed = True

        # Return the computed state:
        return current_state

    def reset_circuit(self) -> None:
        """
        Reset the circuit to its initial empty state.

        Clears all gates and returns to a single empty layer with identity gates.
        """
        self.__number_of_layers = 1
        # Initialize a circuit with one layer with no gates (identity gate is counted as no gate)
        self.__circuit = np.full(
            (1, self.__circuit_qubit_num),
            None,
            dtype=QuantumCircuitCell
        )
        self.__circuit_operator = np.identity(
            2 ** self.__circuit_qubit_num,
            dtype=np.complex128
        )

        self.__circuit_is_computed = False
        # After reset, circuit is non-dynamic
        self.__is_dynamic = False

    def print_operator_matrix(self) -> None:
        """
        Print the final matrix that is applid on the state.
        """
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if self.__is_dynamic:
            raise ValueError("This action is invalid for dynamic circuits.")
        if not self.__circuit_is_computed:
            self.__compute_non_dynamic_circuit()
            self.__circuit_is_computed = True

        print(self.__circuit_operator)

    def get_circuit_operator_matrix(self) -> NDArray[np.complex128]:
        """
        This method returns the final computed matrix of the entire circuit:

        Returns
        -------
        NDarray
            Matrix representing the entire circuit.
        """
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if not self.__circuit_is_computed:
            self.__compute_circuit()
            self.__circuit_is_computed = True

        return self.__circuit_operator

    def device_in_use(self) -> None:
        """
        This method prints the device in use by this quantum circuit.
        """
        print(f"The device computing device used in this circuit: {self.__device}")

    def get_number_of_layers(self) -> int:
        """
        This method returns the number of layers in this circuit.

        Returns
        -------
        int
            The number of layers in the circuit.
        """ 
        return self.__number_of_layers

    def get_number_of_compatible_qubits(self) -> int:
        """
        This method returns the number of qubits this circuit is compatible with.

        Returns
        -------
        int
            The number of qubit this circuit is compatible with.
        """
        return self.__circuit_qubit_num
    
    def get_array(self) -> NDArray:
        """
        This method returns the matrix with all of the gates.

        Returns
        -------
        NDArray
            Matrix with all of the gates.
        """
        return self.__circuit
    
    def load_qft_preset(self,include_measurement:bool = False,m:int = None,measurement_error:float = 0.0) -> None:
        """
        This method loads a prebuild Quantum Fourier Transform circuit using the number of qubits given. This QFT circuit is the regular circuit using a traditional design opposed to the dynamical one. If m is specified an Approximate Quantum Fourier Transform circuit will be used.

        Parameters
        ----------
        m : int, optional
            The cutoff number that will be used for Approximate Quantum Fourier Transform. If None is given the regular QFT circuit will be used.
        measurement_error : float, optional
            The error rate for measurement gates. Default is 0.0 (no error).
        include_measurement : bool, optional
            If True, adds measurement gates to the end of the circuit. Default is False.
        
        Returns
        -------
        None
        """
        # start from a clean circuit
        self.reset_circuit()
        curr_layer_index = 0
        for qubit_index in range(self.__circuit_qubit_num):
            # Add a hadamard gate at the start of each qubit axis
            self.add_single_qubit_gate(qubit_index,curr_layer_index,'H')
            self.add_layer()
            curr_layer_index += 1
            # Add controlled phase shift gates
            if m is None:
                max_dist = self.__circuit_qubit_num + 1 - qubit_index
            else:
                max_dist = min(m+2,self.__circuit_qubit_num + 1 - qubit_index)
            for qubit_dist in range(2, max_dist):
                phase = (2 * np.pi)/ (2**qubit_dist)
                self.add_controlled_qubit_gate(qubit_index,curr_layer_index,qubit_index + qubit_dist - 1,'P',phase)
                self.add_layer()
                curr_layer_index += 1

        if include_measurement:
            # Add measurement gates at the end of each qubit axis
            for qubit_index in range(self.__circuit_qubit_num):
                self.add_measure_gate(qubit_index, curr_layer_index,qubit_index,measurement_error)
        else:
            self.remove_layer(curr_layer_index)  # Remove the last empty layer

        # Add Swap gates:
        # for qubit_index in range(self.__circuit_qubit_num):
        #     if qubit_index < self.__circuit_qubit_num - 1 - qubit_index:
        #         self.add_swap_gate(qubit_index,self.__circuit_qubit_num - 1 - qubit_index,curr_layer_index)        

    def draw_circuit(self,filename:str=None) -> None:
        """
        Print a visualization of the quantum circuit. You can specify to visualize in matplotlib or CLI. The method will visualize using matplotlib by default.

        The visualization includes:
        - Horizontal lines (─) representing qubit wires
        - Single qubit gates with their type (H, X, Y, Z, P)
        - Control points (●) for controlled gates
        - Target points (⊕ for X gates, ◯ for others)
        - Swap gates (⨉)

        Parameters
        ----------
        filename : str
            If a filename is provided, the circuit diagram will be saved as an image file with the given name. Supported formats include PNG, JPG, and SVG. If None, the diagram will be displayed on screen.

        Raises
        ------
        ValueError
            If the input is not a string and not 'mpl' or 'cli'.
        """
        if self.__number_of_layers == 0 or self.__circuit_qubit_num == 0:
            print("Empty circuit")
            return

        fig, ax = plt.subplots(figsize=(self.__number_of_layers * 1.5,self.__circuit_qubit_num * 0.8))

        # Determine measurement layer (if any) for each qubit
        meas_layer_by_qubit = {q: None for q in range(self.__circuit_qubit_num)}
        for layer in range(self.__number_of_layers):
            for q in range(self.__circuit_qubit_num):
                cell = self.__circuit[layer][q]
                if cell is not None and cell.is_measure_gate() and meas_layer_by_qubit[q] is None:
                    meas_layer_by_qubit[q] = layer

        # Add horizontal lines for qubits (ascending order from top to bottom)
        # Draw single line up to the measurement, then double-line (classical) after the measurement.
        dy = 0.04  # vertical separation for classical double line
        for qubit_index in range(self.__circuit_qubit_num):
            y = self.__circuit_qubit_num - 1 - qubit_index
            m_layer = meas_layer_by_qubit[qubit_index]
            if m_layer is None:
                # No measurement on this wire: quantum line across
                ax.plot([0, self.__number_of_layers], [y, y], 'k-', lw=1)
            else:
                # Quantum segment before measurement
                if m_layer > 0:
                    ax.plot([0, m_layer], [y, y], 'k-', lw=1)
                # Classical double line after (and including) the measurement column
                ax.plot([m_layer, self.__number_of_layers], [y - dy, y - dy], color='green', lw=2)
                ax.plot([m_layer, self.__number_of_layers], [y + dy, y + dy], color='green', lw=2)

        for layer in range(self.__number_of_layers):
            for qubit in range(self.__circuit_qubit_num):
                gate = self.__circuit[layer][qubit]

                if gate is None or gate.get_gate_type() == "I" or gate.is_classical_bit():
                    # Skip cells with no gates
                    continue
                elif gate.is_swap_gate():
                    # Handle SWAP gate
                    idx1 = self.__circuit_qubit_num - 1 - gate.get_control_index()
                    idx2 = self.__circuit_qubit_num - 1 - gate.get_target_index()
                    ax.plot([layer, layer], [idx1, idx2], 'k--')
                    ax.text(layer, idx1, '\u2716', ha='center', va='center', fontsize=12)
                    ax.text(layer, idx2, '\u2716', ha='center', va='center', fontsize=12)
                elif gate.is_control_gate():
                    # Handle controlled gate
                    control_idx = self.__circuit_qubit_num - 1 - gate.get_control_index()
                    target_idx = self.__circuit_qubit_num - 1 - gate.get_target_index()
                    ax.plot([layer, layer], [control_idx, target_idx], 'k-')
                    ax.text(layer, control_idx, '\u25CF', ha='center', va='center', fontsize=12)
                    ax.text(layer, target_idx, gate.get_gate_type(), ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='square,pad=0.7', facecolor='white', edgecolor='black'))
                else:
                    # Single-qubit gate or measure gate
                    qubit_idx = self.__circuit_qubit_num - 1 - qubit
                    gate_color = 'lightgreen' if gate.is_measure_gate() else 'white'
                    ax.text(layer, qubit_idx, gate.get_gate_type(), 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='square,pad=0.7', facecolor=gate_color, edgecolor='black'))

            # Second pass: draw classical double-line connections in this layer
            # Connect any measurement gates in this layer to conditional gates that reference their bit
            measured_qubits = set()
            for q in range(self.__circuit_qubit_num):
                cell = self.__circuit[layer][q]
                if cell is not None and cell.is_measure_gate():
                    measured_qubits.add(q)

            # Draw connections from the layer's measurement to all conditional gates in this layer
            # In dynamic QFT, each such layer has exactly one measurement and multiple conditional gates using that bit.
            if measured_qubits:
                # Choose the first (and typically only) measured qubit in this layer
                src_q = sorted(measured_qubits)[0]
                y_src = self.__circuit_qubit_num - 1 - src_q
                x = layer
                for tgt_q in range(self.__circuit_qubit_num):
                    cell = self.__circuit[layer][tgt_q]
                    if cell is not None and cell.is_conditional_gate():
                        y_tgt = self.__circuit_qubit_num - 1 - tgt_q
                        # Draw a double line from measurement to conditional gate at this layer x
                        ax.plot([x, x], [y_src, y_tgt], color='green', linewidth=2, alpha=1)
                        ax.plot([x + 0.04, x + 0.04], [y_src, y_tgt], color='green', linewidth=2, alpha=1)

        # Set axis limits and labels
        ax.set_xlim(-0.5, self.__number_of_layers - 0.5)
        ax.set_ylim(-0.5, self.__circuit_qubit_num - 0.5)

        # Set qubits ticks:
        ax.set_yticks(range(self.__circuit_qubit_num))
        ax.tick_params(axis='y', length=0)
        ax.set_yticklabels([f'q{i}:' for i in range(self.__circuit_qubit_num - 1, -1, -1)])

        # Add layer labels:
        ax.set_xticks([layer for layer in range(self.__number_of_layers)])
        ax.tick_params(axis='x', length=0)

        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Add labels to the axes
        ax.set_xlabel("Layers", fontsize=12)
        ax.set_ylabel("Qubits", fontsize=12)

        padding = 10 if self.__circuit_qubit_num > 6 else 1.08
        plt.tight_layout(pad=padding)
        if filename is not None:
            if not isinstance(filename,str):
                raise ValueError(INV_FILENAME)
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def add_measure_gate(self,qubit_index:int, layer_index: int,c_reg_index: int, measurement_error:float = 0.0) -> None:
        """
        Add measure gate method adds a measure gate to the circuit in a specified layer,on a specified qubit. After performing a measurement the collapsed state of a qubit is saved as a classical bit using the proivided classical register.

        Parameters
        ----------
        qubit_index : int 
            The index of the qubit to which the measure gate will be applied to.
        layer_index : int
            The layer index to which the measure gate will be applied to.
        c_reg_int : int
            The index in the classical register to save the collapsed state.
        measurement_error : float
            The probability of getting a bit flip for the true state.

        Raises
        ------
        ValueError
            If qubit or layer index are invalid a ValueError will be raised.
        ValueError
            If the provided bit index is invalid. Should be between 0 and number of classical bit in the register
        """

        self.__valid_qubit_index(qubit_index,layer_index,adding_gate=True)
        self.__valid_layer_index(layer_index)

        self.__is_dynamic = True
        self.__circuit_is_computed = False
        gate = QuantumCircuitCell()
        gate.set_measure_gate(qubit_index,self.__classical_register,c_reg_index,measurement_error)
        self.__circuit[layer_index][qubit_index]=gate

    def measure_all(self) -> MultiQubit:
        """
        This method applies the circuit on the input state and measures the resulting state. The measured state will be the collapsed state of on of the possible states.

        Returns
        -------
        MultiQubit
            The collapsed state due to measurement.
        """
        result = self.run_circuit()
        return result.measure(return_as_str=False)
    
    def run_circuit(self) -> MultiQubit:
        """
        Run the the whole circuit on the input quantum state. The method runs both on dynamical and non dynamical circuits and returns the final state. 

        Returns
        -------
        MultiQubit
             Resulting quantum state after applying the circuit.

        Raises
        ------
        ValueError
            If the input state has a different number of qubits.
        ValueError
            If the layer index is not valid.
        """
        # # If the circuit is computed and is not dynamic we apply the input state in the computed operator.
        # if self.__circuit_is_computed and not self.__is_dynamic:
        #     qubit_tensor_vector = self.__quantum_state.get_tensor_vector()
        #     result_vector = np.dot(self.__circuit_operator, qubit_tensor_vector)
        #     result_qubit_tensor = MultiQubit(result_vector)
        #     return result_qubit_tensor
        #  # If the circuit is not computed we compute the circuit and return the resulting state.
        # else:
        #     return self.__compute_circuit()

        return self.__compute_circuit()

    def add_conditional_gate(self,target_qubit:int,layer_index:int,c_reg_index:int,gate_type:str,phi:float=0) -> None:
        """
        This method adds a conditional gate and applies the specified unitary of the specified classical bit is one.
        Parameters
        ----------
        qubit_index : int 
            The index of the qubit to which the measure gate will be applied to.
        layer_index : int
            The layer index to which the measure gate will be applied to.
        c_reg_index : int
            The index in the classical register that will be used as conditional bit.

        Raises
        ------
        ValueError
            If qubit or layer index are invalid a ValueError will be raised.
        ValueError
            If the provided bit index is invalid. Should be between 0 and number of classical bit in the register
        """
        gate_type = gate_type.upper()

        self.__valid_qubit_index(target_qubit,layer_index,adding_gate=True)
        self.__valid_layer_index(layer_index)

        self.__is_dynamic = True
        self.__circuit_is_computed = False
        gate = QuantumCircuitCell()
        gate.set_conditional_gate(gate_type, phi, self.__classical_register, c_reg_index, self.__error_single_gate)
        self.__circuit[layer_index][target_qubit]=gate
        
        self.__circuit_is_computed = False

    def change_input_state(self,input_state:MultiQubit) -> None:
        """
        This methods allows to change the input state to another quantum state.

        Raises
        ------
        ValueError
            If the input state isn't the same dimension as this circuit.
        """
        # Check that the number of qubits is correct.
        self.__check_valid_quantum_state(input_state)
        if input_state.get_number_of_qubits() != self.__circuit_qubit_num:
            raise ValueError(INV_INPUT)
        self.__quantum_state = input_state    

    def __check_valid_quantum_state(self,input_state:MultiQubit) -> None:
        """
        Checks if the input is a multiqubit class.

        Raises
        ------
        ValueError 
            If the input is not a multiqubit object
        """
        if not isinstance(input_state,MultiQubit):
            raise ValueError(INV_Q_STATE)
        
    def change_classical_register(self,classical_register:ClassicalRegister) ->None:
        """
        This methods allows to change the classical register to another classical register object.

        Raises
        ------
        ValueError
            If the input is not a classical register object.
        """
        if not isinstance(classical_register,ClassicalRegister):
            raise ValueError(INV_C_REG)
        self.__classical_register = classical_register

    def get_quantum_state(self) -> MultiQubit:
        """
        This method returns the input quantum state of this circuit.

        Returns
        -------
        MultiQubit
            The input quantum state.
        """ 
        return self.__quantum_state
    
    def get_classical_register(self) -> ClassicalRegister:
        """
        This method returns the classical register of this circuit.

        Returns
        -------
        ClassicalRegister
            The classical register of the circuit.
        """ 
        return self.__classical_register

    def __compute_measurement_gates(self,input_state:MultiQubit,layer_gates: NDArray) -> MultiQubit:
        """
        If there are measurement gates in the layer we perform a measurement and return the resulting MultiQubit state.

        Parameters
        ----------
        input_state : MultiQubit
            The input state on which we perform the measurement.

        Returns
        -------
        MultiQubit
            The resulting MultiQubit state.
        """
        curr_state = input_state
        for gate in layer_gates:
            if gate.is_measure_gate():
                curr_state = gate.measure(curr_state)
        return curr_state
    
    def load_dynamic_qft_preset(self,m:int = None,measurement_error:float= 0.0) -> None:
        """
        This method loads a prebuild dynamic Quantum Fourier Transform circuit. If the initialized classical register doesn't have the correct amount of bits we raise an error. 
        If m is specified an Approximate Quantum Fourier Transform circuit will be used.

        Parameters
        ----------
        m : int, optional
            The cutoff number that will be used for Approximate Quantum Fourier Transform. If None is given the regular QFT circuit will be used.
        measurement_error : float
            The probability of getting a bit flip for the true state. Default is 0.0

        Raises
        ------
        ValueError
            If the classical register has different number of bits than the number of qubits in the circuit.

        """
        self.reset_circuit()

        # Check number of bits in the classical register
        if self.__classical_register.get_bits_num() != self.__circuit_qubit_num:
            raise ValueError(INV_NUM_C_REG)
        
        # Reset the classical register to avoid bugs.
        self.__classical_register.reset()        

        curr_layer_index = 0
        for qubit_index in range(self.__circuit_qubit_num):
            # Add Hadamrd gate and measure gate.
            self.add_single_qubit_gate(qubit_index,curr_layer_index,"H")
            self.add_layer()
            curr_layer_index += 1
            self.add_measure_gate(qubit_index,curr_layer_index,qubit_index,measurement_error)
            first_phase_qubit_idx = qubit_index + 1

            if m is None:
                max_dist = self.__circuit_qubit_num + 1 - qubit_index
            else:
                max_dist = min(m + 2, self.__circuit_qubit_num + 1 - qubit_index)

            for qubit_dist in range(2, max_dist):
                phase = (2 * np.pi)/ (2**qubit_dist)
                self.add_conditional_gate(first_phase_qubit_idx,curr_layer_index,qubit_index,"P",phase)
                first_phase_qubit_idx += 1
            
            if (qubit_index + 1) < self.__circuit_qubit_num:
                self.add_layer()
                curr_layer_index += 1

    def run_many(self, num_of_runs:int = 1000) -> MultiQubit:
        """
        This method runs the circuit multiple times and returns quantum state which probabilities are defined by the number of states we got at every run of the circuit.
        This is useful in the case of dynamic circuit where we get only a one state and we want to return the original state before the collapse.

        Parameters
        ----------
        num_of_runs : int
            The number of times to run the circuit. Is 10000 by default.

        Returns
        -------
        MultiQubit
            The quantum state with probabilities we mentioned above.
        """

        # vector that counts how many times each state we collapsed to each state.
        count_vector = np.zeros(2**self.__circuit_qubit_num)

        for run in range(num_of_runs):
            if self.__is_dynamic:
                collapsed_state = self.run_circuit()
            else:
                collapsed_state = self.measure_all()
                
            state_vector = collapsed_state.get_tensor_vector()

            # Return the index of corresponding to the collapsed state
            state_index = np.where(state_vector != 0)[0][0]
            count_vector[state_index] += 1

        count_vector /= num_of_runs
        count_vector = np.sqrt(count_vector)

        return MultiQubit(count_vector)