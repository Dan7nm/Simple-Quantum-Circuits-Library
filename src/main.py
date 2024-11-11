from circuit import Circuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np

def main():
    # q0 = Qubit(1,0)
    # q1 = Qubit(0,1)
    # mt = MultiQubit()
    # mt.add_qubit(q1)
    # mt.add_qubit(q1)
    # mt.add_qubit(q1)
    # mt.add_qubit(q0)
    # mt.add_qubit(q0)
    # mt.print_tensor_form()
    # circuit = Circuit(mt)
    # circuit.add_controlled_qubit_gate(0,0,1,"X")
    # circuit.add_layer()
    # circuit.add_single_qubit_gate(3,1,"X")
    # circuit.add_single_qubit_gate(0,1,"X")
    # circuit.add_swap_gate(2,4,0)
    # circuit.compute_circuit()
    # circuit.print_circuit()
    # result = circuit.apply_circuit()
    # result.print_tensor_form()
    
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    mt = MultiQubit()
    mt.add_qubit(q0)    
    mt.add_qubit(q0)    
    # mt.add_qubit(q0)    
    mt.print_tensor_form()
    circuit = Circuit(mt)
    circuit.load_qft_preset()
    circuit.add_layer()
    circuit.compute_circuit()
    circuit.print_circuit()
    circuit.print_operator_matrix()
    result = circuit.apply_circuit()
    result.print_tensor_form()
    


if __name__ == "__main__":
    main()