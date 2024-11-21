from circuit import QuantumCircuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np

def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    mt = MultiQubit()
    mt.add_qubit(q1)
    mt.add_qubit(q0)
    mt.add_qubit(q0)
    # mt.print_tensor_form()

    circuit = QuantumCircuit(3)
    circuit.add_controlled_qubit_gate(1,0,0,"X")
    circuit.add_layer(layer_index=0)
    circuit.add_layer(layer_index=2)
    circuit.print_circuit()
    circuit.remove_layer(1)
    circuit.print_circuit()
    # result = circuit.apply_circuit(mt)

    # result.print_tensor_form()

    


if __name__ == "__main__":
    main()