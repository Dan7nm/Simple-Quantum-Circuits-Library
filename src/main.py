from circuit import Circuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np

def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    mt = MultiQubit()
    mt.add_qubit(q0)
    mt.add_qubit(q1)

    circuit = Circuit(3)
    circuit.add_single_qubit_gate(0,0,"X")
    circuit.add_single_qubit_gate(1,0,"X")

    circuit.compute_circuit()
    circuit.print_circuit()
    circuit.print_operator_matrix()
    


if __name__ == "__main__":
    main()