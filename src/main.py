from circuit import QuantumCircuit
from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np
from gui import CircuitDrawer
def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)
    mt = MultiQubit()

    circuit = QuantumCircuit(3)
    circuit.load_qft_preset()
    gui = CircuitDrawer(circuit)
    gui.start()
    print("Finished running")

if __name__ == "__main__":
    main()