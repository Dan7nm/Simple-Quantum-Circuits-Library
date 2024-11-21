from circuit import QuantumCircuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np

def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    q2 = Qubit(np.sqrt(3)/2,1/2)
    mt = MultiQubit()
    mt.add_qubit(q1)
    mt.add_qubit(q1)
    mt.add_qubit(q0)
    mt.add_qubit(q1)
    mt.print_tensor_form()

    

    circuit = QuantumCircuit(4)
    circuit.load_qft_preset()

    result = circuit.apply_circuit(input_state= mt)
    result.print_tensor_form()


    circuit.measure_all(input_state= mt, number_of_shots= 10000)

    # circuit.print_circuit()

    


if __name__ == "__main__":
    main()