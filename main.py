from qubit import Qubit
from qubit_tensor import QubitTensor
from gate import SingleQubitGate
from gate_tensor import GateTensor
from qft import QFT
import numpy as np

def main():
    q0 = Qubit(1j,0)
    q1 = Qubit(0,1)

    q0.print_qubit()
    q0.print_vector_form()

if __name__ == "__main__":
    main()