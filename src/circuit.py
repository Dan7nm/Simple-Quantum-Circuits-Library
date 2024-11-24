from gate import Gate
from multi_qubit import MultiQubit
import numpy as np
from numpy.typing import NDArray
from typing import List,Tuple,Set
import matplotlib.pyplot as plt
import random

INV_QUBIT_INDEX = "The qubit index is invalid. The qubit index should be between 0 and number of qubits - 1."
INV_LAYER_INDEX = "The layer index is invalid. The layer index should be an integer between 0 and number of layers - 1"
CELL_TAKEN = "This cell already has a gate assigned to it. Use the remove gate method to remove a gate."
INV_INPUT = "The input state has a different number of qubits to that of this quantum circuit."
COMP_CIRCUIT = "The circuit was not computed. Before applying a state you should compute the circuit and after every update to a gate."
INV_NUM_QUBITS = "The number of qubits should be atleast 1."
INV_CTRL_TARG = "Invalid target and qubit index input. The target qubit and the control qubit should be different from each other"
INV_INIT_LAYERS = "Invalid number of layers. The number should be non zero positive integer."
INV_POS_VAL = "The value is invalid. The value should be a positive non zero integer."

class QuantumCircuit:
    """
    :ivar number_of_compatible_qubits: Number of qubits in this circuit.
    :vartype number_of_compatible_qubits: int

    A class to represent a quantum circuit using quantum gates.

    The QuantumCircuit class allows for building a quantum circuit, by adding and removing a quantum gate on every qubit at each vertical and horizontal axis. Every iteration from left to right is described by layers. Each layer is compromised of a tensor product of single qubit gates or controlled gates.

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
    def __init__(self,number_of_qubits: int, num_of_layers: int = 1) -> None:
        # Check if valid inputs:
        self.__valid_pos_val(number_of_qubits)
        self.__valid_pos_val(num_of_layers)

        self.__number_of_compatible_qubits = number_of_qubits
        self.__number_of_layers = num_of_layers

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
        # Check that the control qubit and target qubits are different:
        if target_qubit == control_qubit:
            raise ValueError(INV_CTRL_TARG)
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
        # Check that the first qubit and second qubits are different:
        if first_qubit == second_qubit:
            raise ValueError(INV_CTRL_TARG)
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

    def add_layer(self, layer_index: int= None) -> None:
        """
        Adds a new empty layer to aspecified layer index in the circuit. If now index is given then adds the layer to the end of the circuit by default.
        
        The new layer is initialized with None values, representing identity gates.

        Parameters
        ----------
        layer_index : int
            The index to which a layer will be added.

        """
        # Check if the layer index is valid
        if layer_index is not None:
            self.__valid_layer_index(layer_index=layer_index)
    
        self.__number_of_layers+=1
        if layer_index is None:
            # Add another row of nones:
            new_row = np.full((1, self.__number_of_compatible_qubits), None)
            self.__circuit = np.vstack((self.__circuit, new_row))
        else:
            new_row = np.full((1, self.__number_of_compatible_qubits), None)
            self.__circuit = np.insert(self.__circuit,layer_index,new_row,axis=0)
        # The circuit was updated show it should be computed to avoid getting a wrong state.
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

        


    def __valid_qubit_index(self,qubit_index: int,layer_index: int, adding_gate: bool = True) -> None:
        """
        Remove the last layer from the circuit.
        
        Reduces the number of layers by one and removes all gates in the last layer.
        """
        # Check for valid indexes
        if not isinstance(qubit_index,int) and not isinstance(layer_index) and not 0 <= qubit_index <= self.__number_of_compatible_qubits - 1:
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
        if not isinstance(layer_index,int) and not 0 <= layer_index <= self.__number_of_layers:
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

    def __compute_single_qubit_gates(self, layer_gates: NDArray) -> NDArray:
        """
        This function  computes all single qubit gates in one layer.

        :param layer_gates: An array of all gates in the layer we want to compute.
        :type layer_gates: NDArray
        :return: The function returns the unitary matrix of that layer
        :rtype: NDArray
        """
        result_matrix = np.ones((1,),dtype=complex)
        identity_matrix = np.identity(2)

        for gate in layer_gates:
            # Check if current gate is a single qubit gate.
            if gate.is_single_qubit_gate():
                gate_matrix = gate.get_matrix()
                # Compute the kronecker product of all single qubit gates
                result_matrix = np.kron(result_matrix,gate_matrix)
            else:
                # If the gate is not a single qubit gate we compute the kronecker product with the identity matrix.
                result_matrix = np.kron(result_matrix,identity_matrix)
        
        return result_matrix

    def __swap_gate_matrix(self,first_index: int, second_index: int) -> NDArray:
        """
        Generate the matrix representation of a SWAP gate between first_index and second qubit index in an n-qubit system.
        
        :param first_index: Index of the first qubit (0-based indexing)
        :type first_index: int
        :param second_index: Index of the second qubit (0-based indexing)
        :type second_index: int

        :return: 2^n x 2^n matrix representing the SWAP operation
        :rtype: NDArray
        """
        # Calculate matrix dimension
        dim = 2 ** self.__number_of_compatible_qubits
        
        # Initialize the matrix with zeros
        swap_matrix = np.zeros((dim, dim))
        
        # Generate all possible basis states
        for state in range(dim):
            # Convert state to binary representation
            binary = format(state, f'0{self.__number_of_compatible_qubits}b')
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
        :return: The function returns the unitary matrix of that layer
        :rtype: NDArray
        """
        result_matrix = np.identity(2 ** self.__number_of_compatible_qubits)

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

                result_matrix = result_matrix @ swap_matrix
        
        return result_matrix

    def __create_controlled_operation(
        self,
        gate: Gate,
        control_index: int,
        target_index: int
    ) -> NDArray:
        """
        Create a controlled operation matrix for the given gate.
        
        Args:
            gate: The controlled gate to process
            is_control_first: Whether the control qubit comes before the target
            
        Returns:
            The resulting controlled operation matrix
        """
        index_diff = abs(control_index - target_index) - 1
        identity_mid = np.identity(2 ** index_diff)
        proj_00_matrix = np.array([[1,0],[0,0]], dtype=complex)
        proj_11_matrix = np.array([[0,0],[0,1]], dtype=complex)
        gate_matrix = gate.get_matrix()
        
        # Check if control index is before target index
        if control_index < target_index:
            first_term = np.kron(np.kron(proj_00_matrix, identity_mid), np.identity(2))
            second_term = np.kron(np.kron(proj_11_matrix, identity_mid), gate_matrix)
        else:
            first_term = np.kron(np.kron(np.identity(2), identity_mid), proj_00_matrix)
            second_term = np.kron(np.kron(gate_matrix, identity_mid),proj_11_matrix)
            
        return first_term + second_term

    def __embed_in_circuit(self, gate: Gate, operation: NDArray, control_index: int, target_index: int) -> NDArray:
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
        identity_prev = np.identity(2 ** lower_idx)
        identity_after = np.identity(2 ** (self.__number_of_compatible_qubits - higher_idx - 1))
        
        # Embed the operation in the full circuit space
        return np.kron(np.kron(identity_prev, operation), identity_after)

    def __compute_controlled_gates(self, layer_gates: NDArray) -> NDArray:
        """
        Compute all controlled gates in one layer.
        
        Parameters
        ----------
        layer_gates: NDArray
            List of controlled gates in the layer.
            
        Returns
        -------
        out - NDArray    
            The unitary matrix representing the complete layer
        """
        processed_qubits: Set[int] = set()
        result_matrix = np.identity(2 ** self.__number_of_compatible_qubits)
        
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
                result_matrix = result_matrix @ embedded_op
            
        return result_matrix

    def __compute_layer(self, layer_index: int) -> NDArray[np.complex128]:
        """
        Compute the combined operation of all gates in a single layer.

        Parameters
        ----------
        layer_index : int
            Index of the layer to compute.
        
        Returns
        -------
        matrix : NDArray[np.complex128]
            Matrix representing the combined operation of all gates in the layer.
        """
        # Get all gates in the current layer
        layer_gates = self.__circuit[layer_index]

        result_matrix = np.identity(2 ** self.__number_of_compatible_qubits, dtype= complex)
        
        # Compute all single qubit matrices:
        single_qubit_gates_matrix = self.__compute_single_qubit_gates(layer_gates)

        # Compute all controlled gates:
        controlled_gates_matrix = self.__compute_controlled_gates(layer_gates)

        # Compute all swap gates in this layer:
        swap_gates_matrices = self.__compute_swap_gates(layer_gates=layer_gates)

        # Multtiply all computed matrices
        result_matrix = result_matrix @ single_qubit_gates_matrix @ controlled_gates_matrix @ swap_gates_matrices
             
        return result_matrix
        
    def __compute_circuit(self) -> None:
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
        self.__circuit_operator = np.identity(2 ** self.__number_of_compatible_qubits)

        self.__circuit_is_computed = False

    def apply_circuit(self, input_state: MultiQubit) -> MultiQubit:
        """
        Apply the circuit operator to the input quantum state.

        Parameters
        ----------
        input_state : MultiQubit
            New quantum state to use as input.

        Returns
        -------
        mult_qubit : MultiQubit
             Resulting quantum state after applying the circuit.

        Raises
        ------
        ValueError
            If the input state has a different number of qubits
        """
        # Check that the number of qubits is correct.
        if input_state.get_number_of_qubits() != self.__number_of_compatible_qubits:
            raise ValueError(INV_INPUT)
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if not self.__circuit_is_computed:
            self.__compute_circuit()
            self.__circuit_is_computed = True

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
                            circuit_lines[i].append(f'──●{target_idx}──')
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
            print(f'    q{i}: {"".join(line)}')
        
        layers_indexes = [f"──{i}───" for i in range(self.__number_of_layers)]

        print(f"layers: {''.join(layers_indexes)}")

    def print_operator_matrix(self) -> None:
        """
        Print the final matrix that is applid on the state.
        """
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if not self.__circuit_is_computed:
            self.__compute_circuit()
            self.__circuit_is_computed = True

        print(self.__circuit_operator)

    def get_circuit_operator_matrix(self) -> NDArray[np.complex128]:
        """
        This function returns the final computed matrix of the entire circuit:

        Returns
        -------
        circuit_operator : NDarray
            Matrix representing the entire circuit.
        """
        # Check that the circuit was computed before applying a state if not than we compute the circuit:
        if not self.__circuit_is_computed:
            self.__compute_circuit()
            self.__circuit_is_computed = True

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



    def __collapse_to_state(self,amplitude:complex) -> float:
        """
        This function returns with specified amplitude a bolean if the state collapsed to this state or not.

        Parameters
        ----------
        Amplitude : complex
            The amplitude of the given state.

        Returns
        -------
        out : bool
            True if we collapsed to the current desired state, otherwise return False.
        
        """
        prob = abs(amplitude) ** 2
        return random.random() < prob

    def measure_all(self,input_state: MultiQubit,number_of_shots: int = 10000) -> None:
        """
        This function measures the entire circuit and collapes using born rule to one of the superposition states with probability of the amplitudes squared.

        The function can measure the circuit a specified number of times. The method plots in a graph the probablity of each state.

        Parameters
        ----------
        number_of_shots : int
            The number of times to measure the entire circuit.
        input_state : MultiQubit
            The input state to measure.
        """
        self.__valid_pos_val(value= number_of_shots)
        result_state = self.apply_circuit(input_state= input_state)
        result_vector = result_state.get_tensor_vector()

        # Dictionary to stores the states and number of times we collapsed to that state.
        states_dict = {}

        for shot in range(number_of_shots):
            for state in range(2 ** self.__number_of_compatible_qubits):
                if self.__collapse_to_state(result_vector[state]):
                    state_binary = format(state, f'0{self.__number_of_compatible_qubits}b')
                    states_dict[state_binary] = states_dict.get(state_binary,0) + 1
        
        states_list = list(states_dict.keys())
        probs_list = np.array(list(states_dict.values())) / number_of_shots

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(states_list, probs_list, color='blue', alpha=0.7)

        # Add labels and title
        plt.xlabel('Quantum States', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Probability Distribution of Measured Quantum States\n(Number of Measurments: {number_of_shots})', fontsize=14)

        # Show grid and the plot
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()






