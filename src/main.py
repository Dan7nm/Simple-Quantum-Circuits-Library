from circuit import QuantumCircuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np

def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    mt = MultiQubit()
    mt.add_qubit(q0)
    mt.add_qubit(q0)
    mt.add_qubit(q1)
    mt.print_tensor_form()

    circuit = QuantumCircuit(3)
    circuit.load_qft_preset()
    circuit.print_circuit()

    result = circuit.apply_circuit(input_state= mt)
    result.print_tensor_form()
    # result.print_vector_form()

if __name__ == "__main__":
    main()