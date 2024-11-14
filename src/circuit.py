from gate import Gate
from multi_qubit import MultiQubit
import numpy as np
from numpy.typing import NDArray

INV_QUBIT_INDEX = "The qubit index is invalid. The qubit index should be between 0 and number of qubits - 1."
INV_LAYER_INDEX = "The layer index is invalid. The layer index should be between 0 and number of layers - 1"
CELL_TAKEN = "This cell already has a gate assigned to it. Use the remove gate method to remove a gate."
INV_INPUT = "The input state has a different number of qubits to that of this quantum circuit."
COMP_CIRCUIT = "The circuit was not computed. Before applying a state you should compute the circuit and after every update to a gate."

class Circuit:
    """
    :ivar number_of_compatible_qubits: Number of qubits in this circuit.
    :vartype number_of_compatible_qubits: int

    A class to represent a quantum circuit using quantum gates.

    The Circuit class allows for building a quantum circuit, by adding and removing a quantum gate on every qubit at each vertical and horizontal axis. Every iteration from left to right is described by layers. Each layer is compromised of a tensor product of single qubit gates or controlled gates.

    Steps for circuit building:
    ---------------------------

    1. Initializes a first empty layer without any gates.
    2. Adds single qubit gates and control gates. Build a two dimensional array. Every row of the array represents a layer and every column represents a gate on that particular qubit. If no gate is chosen the identity gate will be the default gate.
    3. Move to the next layer and repeat step 2.
    4. After finishing designing the whole circuit we compute the final matrix. This is done by computing the Krorecker product in each layer and multiplying all the matrices of all layers.


    Example Usage:
    --------------
    >>> q0 = Qubit(1,0)
    >>> q1 = Qubit(0,1)
    >>> mt = MultiQubit()
    >>> mt.add_qubit(q1)
    >>> mt.add_qubit(q1)
    >>> mt.add_qubit(q1)
    >>> mt.add_qubit(q0)
    >>> mt.add_qubit(q0)
    >>> mt.print_tensor_form()
    >>> circuit = Circuit(mt)
    >>> circuit.add_controlled_qubit_gate(0,0,1,"X")
    >>> circuit.add_layer()
    >>> circuit.add_single_qubit_gate(3,1,"X")
    >>> circuit.add_single_qubit_gate(0,1,"X")
    >>> circuit.add_swap_gate(2,4,0)
    >>> circuit.compute_circuit()
    >>> circuit.print_circuit()
    >>> result = circuit.apply_circuit()
    >>> result.print_tensor_form()
    Tensor product in basis state form: |11100⟩
    Circuit Diagram:
    q0: ─[CX]──[X]──
    q1: ──●0────────
    q2: ──⨉─────────
    q3: ───────[X]──
    q4: ──⨉─────────
    Tensor product in basis state form: |11011⟩
    """
    def __init__(self,number_of_qubits: int) -> None:
        self.__number_of_compatible_qubits = number_of_qubits
        self.__number_of_layers = 1
        # Initialize a circuit with one layer with no gates (identity gate is counted as no gate)
        self.__circuit= np.full(((1,self.__number_of_compatible_qubits)),None)
        self.__circuit_operator = np.identity(2 ** self.__number_of_compatible_qubits)

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False
    
    def add_single_qubit_gate(self, target_qubit: int, layer_index: int, gate_type: str, phi: float = 0.0) -> None:
        """
        Add a single qubit gate to the circuit.

        :param target_qubit: Index of the qubit to apply the gate to
        :type target_qubit: int
        :param layer_index: Index of the layer where the gate should be added
        :type layer_index: int
        :param gate_type: Type of quantum gate ('X', 'Y', 'Z', 'H', 'P')
        :type gate_type: str
        :param phi: Phase angle for phase gates (default: 0.0)
        :type phi: float, optional
        :raises ValueError: If qubit index or layer index is invalid, or if cell is occupied
        """
        # Check if layer and qubit indexes are valid and the cell is empty
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit,layer_index)
        gate = Gate()
        gate.set_single_qubit_gate(gate_type,phi)
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
        # Check if layer and qubit indexes are valid and the cell is empty
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit, layer_index)
        self.__valid_qubit_index(control_qubit, layer_index)
        gate = Gate()
        gate.set_controlled_qubit_gate(control_qubit,target_qubit,gate_type,phi)
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
        # Check if layer and qubit indexes are valid and the cell is empty.
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(first_qubit, layer_index)
        self.__valid_qubit_index(second_qubit, layer_index)
        gate = Gate()
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
        # Check if layer and qubit indexes are valid and the cell is empty. 
        self.__valid_layer_index(layer_index)
        self.__valid_qubit_index(target_qubit,layer_index, adding_gate = False)
        self.__valid_qubit_index(control_qubit,layer_index, adding_gate = False)

        self.__circuit[layer_index][target_qubit] = None
        self.__circuit[layer_index][control_qubit] = None

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def add_layer(self) -> None:
        """
        Add a new empty layer to the circuit.
        
        The new layer is initialized with None values, representing identity gates.
        """
        self.__number_of_layers+=1
        # Add another row of nones:
        new_row = np.full((1, self.__number_of_compatible_qubits), None)
        self.__circuit = np.vstack((self.__circuit, new_row))

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def remove_last_layer(self) -> None:
        """
        Remove the last layer from the circuit.
        
        Reduces the number of layers by one and removes all gates in the last layer.
        """
        self.__circuit = self.__circuit[:-1]
        self.__number_of_compatible_qubits -= 1

        # The circuit was updated show it should be computed to avoid getting a wrong state.
        self.__circuit_is_computed = False

    def __valid_qubit_index(self,qubit_index: int,layer_index: int, adding_gate: bool = True) -> None:
        """
        Remove the last layer from the circuit.
        
        Reduces the number of layers by one and removes all gates in the last layer.
        """
        # Check for valid indexes
        if not 0 <= qubit_index <= self.__number_of_compatible_qubits - 1:
            raise ValueError(INV_QUBIT_INDEX)
        # Check this step only if we are adding a gate. In removal this step is not used.
        # Check if the cell is an empty to add a gate
        if adding_gate and self.__circuit[layer_index][qubit_index] is not None:
            raise ValueError(CELL_TAKEN)
        
    def __valid_layer_index(self, layer_index) -> None:
        """
        Validate a layer index.

        :param layer_index: Index of the layer to validate
        :type layer_index: int
        :raises ValueError: If layer index is invalid
        """
        if not 0 <= layer_index <= self.__number_of_layers:
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
                    identity_gate = Gate()
                    identity_gate.set_single_qubit_gate("I")
                    self.__circuit[layer_index][qubit_index] = identity_gate


    def __compute_layer(self, layer_index: int) -> NDArray[np.complex128]:
        """
        Compute the combined operation of all gates in a single layer.

        :param layer_index: Index of the layer to compute
        :type layer_index: int
        :return: Matrix representing the combined operation of all gates in the layer
        :rtype: NDArray[np.complex128]
        """
        # Get all gates in the current layer
        layer = self.__circuit[layer_index]
        processed_qubits = set()
        matrices_to_multiply = []
        
        # Process each gate in the layer
        for qubit_index, gate in enumerate(layer):
            # Skip if this qubit has already been processed (happens with controlled gates)
            if qubit_index in processed_qubits:
                continue
                
            if gate.is_control_gate():
                # For controlled gates, create a controlled operation matrix
                control_idx = gate.get_control_index()
                target_idx = gate.get_target_index()
                # Mark both qubits as processed
                processed_qubits.add(control_idx)
                processed_qubits.add(target_idx)
                
                # Create controlled gate matrix
                n = 2 ** self.__number_of_compatible_qubits
                controlled_matrix = np.identity(n, dtype=complex)
                gate_matrix = gate.get_matrix()
                
                # Calculate positions where control qubit is 1
                for i in range(n):
                    # Convert to binary and check if control qubit is 1
                    binary = format(i, f'0{self.__number_of_compatible_qubits}b')
                    if binary[control_idx] == '1':
                        # Apply gate to target qubit
                        new_state = list(binary)
                        if binary[target_idx] == '0':
                            # Apply first column of gate matrix
                            controlled_matrix[i, i] = gate_matrix[0, 0]
                            new_state[target_idx] = '1'
                            j = int(''.join(new_state), 2)
                            controlled_matrix[i, j] = gate_matrix[0, 1]
                        else:
                            # Apply second column of gate matrix
                            new_state[target_idx] = '0'
                            j = int(''.join(new_state), 2)
                            controlled_matrix[i, j] = gate_matrix[1, 0]
                            controlled_matrix[i, i] = gate_matrix[1, 1]
                
                matrices_to_multiply.append(controlled_matrix)
                
            elif gate.is_swap_gate():
                # For swap gates, create a swap operation matrix
                first_idx = gate.get_control_index()
                second_idx = gate.get_target_index()
                # Mark both qubits as processed
                processed_qubits.add(first_idx)
                processed_qubits.add(second_idx)
                
                # Create swap matrix
                n = 2 ** self.__number_of_compatible_qubits
                swap_matrix = np.zeros((n, n), dtype=complex)
                
                # Fill swap matrix
                for i in range(n):
                    binary = list(format(i, f'0{self.__number_of_compatible_qubits}b'))
                    # Swap the bits at the specified positions
                    binary[first_idx], binary[second_idx] = binary[second_idx], binary[first_idx]
                    j = int(''.join(binary), 2)
                    swap_matrix[i, j] = 1
                    
                matrices_to_multiply.append(swap_matrix)
                
            else:
                # For single qubit gates, compute the matrix using tensor product
                processed_qubits.add(qubit_index)
                
                # Initialize identity matrices for qubits before the current one
                if qubit_index > 0:
                    before_matrix = np.identity(2 ** qubit_index, dtype=complex)
                else:
                    before_matrix = np.array([[1]], dtype=complex)
                    
                # Initialize identity matrices for qubits after the current one
                remaining_qubits = self.__number_of_compatible_qubits - qubit_index - 1
                if remaining_qubits > 0:
                    after_matrix = np.identity(2 ** remaining_qubits, dtype=complex)
                else:
                    after_matrix = np.array([[1]], dtype=complex)
                    
                # Compute the final matrix for this gate using tensor products
                gate_matrix = gate.get_matrix()
                final_matrix = np.kron(np.kron(before_matrix, gate_matrix), after_matrix)
                matrices_to_multiply.append(final_matrix)
        
        # Multiply all matrices together
        result_matrix = np.eye(2 ** self.__number_of_compatible_qubits, dtype=complex)
        for matrix in matrices_to_multiply:
            result_matrix = np.matmul(matrix, result_matrix)
            
        return result_matrix
        
    def compute_circuit(self) -> None:
        """
        Compute the final unitary matrix representing the entire circuit.

        This method:
        1. Fills empty cells with identity gates
        2. Computes the matrix for each layer
        3. Multiplies all layer matrices to get the final circuit operator
        """
        # First fill all empty cells with identity gates
        self.__fill_identity_gates()
        # Multiply the matrices of each layer to compute the final matrix of the whole circuit.
        for layer_index in range(self.__number_of_layers):
            layer_matrix = self.__compute_layer(layer_index)
            self.__circuit_operator = np.matmul(self.__circuit_operator,layer_matrix)

        # Update that the circuit was computed
        self.__circuit_is_computed = True

    def reset_circuit(self) -> None:
        """
        Reset the circuit to its initial empty state.

        Clears all gates and returns to a single empty layer with identity gates.
        """
        self.__number_of_layers = 1
        # Initialize a circuit with one layer with no gates (identity gate is counted as no gate)
        self.__circuit= np.full(((1,self.__number_of_compatible_qubits)),None)
        self.__circuit_operator = np.identity(self.__number_of_compatible_qubits)

    def apply_circuit(self, input_state: MultiQubit) -> MultiQubit:
        """
        Apply the circuit operator to the input quantum state.

        :param input_state: New quantum state to use as input.
        :type input_state: MultiQubit
        :return: Resulting quantum state after applying the circuit.
        :rtype: MultiQubit
        :raises ValueError: If the input state has a different number of qubits
        """
        # Check that the number of qubits is correct.
        if input_state.get_number_of_qubits() != self.__number_of_compatible_qubits:
            raise ValueError(INV_INPUT)
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if not self.__circuit_is_computed:
            self.compute_circuit()
            
        qubit_tensor_vector = input_state.get_tensor_vector()
        result_vector = np.dot(self.__circuit_operator, qubit_tensor_vector)
        result_qubit_tensor = MultiQubit(result_vector, self.__number_of_compatible_qubits)
        return result_qubit_tensor
    
    def print_circuit(self) -> None:
        """
        Print a text-based visualization of the quantum circuit.

        The visualization includes:
        - Horizontal lines (─) representing qubit wires
        - Single qubit gates with their type (H, X, Y, Z, P)
        - Control points (●) for controlled gates
        - Target points (⊕ for X gates, ◯ for others)
        - Swap gates (⨉)
        """
        if self.__number_of_layers == 0 or self.__number_of_compatible_qubits == 0:
            print("Empty circuit")
            return
        
        # Dictionary to map gate types to symbols
        gate_symbols = {
            'H': 'H',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'P': 'P',
            'I': 'I'
        }
        
        # Build the circuit layer by layer
        circuit_lines = [[] for _ in range(self.__number_of_compatible_qubits)]
        
        # Process each layer
        for layer in range(self.__number_of_layers):
            # Track which qubits have been processed in this layer
            processed_qubits = set()
            
            # First pass: Add gates and controls
            for qubit in range(self.__number_of_compatible_qubits):
                if qubit in processed_qubits:
                    continue
                    
                gate = self.__circuit[layer][qubit]
                
                # Empty space (identity gate)
                if gate is None or gate.get_gate_type() == "I":
                    circuit_lines[qubit].append("──────")
                
                elif gate.is_swap_gate():
                    # Handle swap gate
                    first_idx = gate.get_control_index()
                    second_idx = gate.get_target_index()
                    processed_qubits.add(first_idx)
                    processed_qubits.add(second_idx)
                    # Add swap symbols
                    for i in range(self.__number_of_compatible_qubits):
                        if i == first_idx:
                            circuit_lines[i].append(f'──⨉{second_idx}──')
                        if i == second_idx:
                            circuit_lines[i].append(f'──⨉{first_idx}──')
                                
                elif gate.is_control_gate():
                    # Handle controlled gates
                    control_idx = gate.get_control_index()
                    target_idx = gate.get_target_index()
                    processed_qubits.add(control_idx)
                    processed_qubits.add(target_idx)
                    # Determine gate symbol for target
                    target_symbol = 'C' + gate.get_gate_type()   
                    # Add gate elements
                    for i in range(self.__number_of_compatible_qubits):
                        if i == control_idx:
                            circuit_lines[i].append(f'──●{qubit}──')
                        elif i == target_idx:
                            circuit_lines[i].append(f'─[{target_symbol}]─')
                else:
                    # Handle single qubit gates
                    processed_qubits.add(qubit)
                    gate_symbol = gate.get_gate_type()
                    # Add the gate
                    for i in range(self.__number_of_compatible_qubits):
                        if i == qubit:
                            circuit_lines[i].append(f'─[{gate_symbol}]──')
        
        # Print the circuit
        print("\nCircuit Diagram:")
        for i, line in enumerate(circuit_lines):
            print(f'q{i}: {"".join(line)}')

    def print_operator_matrix(self) -> None:
        """
        Print the final matrix that is applid on the state.
        """
        print(self.__circuit_operator)

    def get_circuit_operator_matrix(self) -> NDArray[np.complex128]:
        """
        This function returns the final computed matrix of the entire circuit:
        :return: Matrix representing the entire circuit.
        :rtype: NDArray 
        """
        return self.__circuit_operator

    def print_array(self) -> None:
        print(self.__circuit)

    def load_qft_preset(self) -> None:
        """
        This method loads a prebuild Quantum Fourier Transform circuit using the number of qubits given.
        """
        curr_layer_index = 0
        for qubit_index in range(self.__number_of_compatible_qubits):
            # Add a hadamard gate at the start of each qubit axis
            self.add_single_qubit_gate(qubit_index,curr_layer_index,'H')
            self.add_layer()
            curr_layer_index += 1
            # Add controlled phase shift gates
            for phase_gate_index in range(2, self.__number_of_compatible_qubits + 1 - qubit_index):
                phase = (2 * np.pi)/ (2**phase_gate_index)
                self.add_controlled_qubit_gate(qubit_index,curr_layer_index,qubit_index + phase_gate_index - 1,'P',phase)
                self.add_layer()
                curr_layer_index += 1

        # Add Swap gates:
        for qubit_index in range(self.__number_of_compatible_qubits):
            if qubit_index < self.__number_of_compatible_qubits - 1 - qubit_index:
                self.add_swap_gate(qubit_index,self.__number_of_compatible_qubits - 1 - qubit_index,curr_layer_index)




