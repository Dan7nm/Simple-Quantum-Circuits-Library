from qubit import Qubit
from multi_qubit import MultiQubit
from gate import SingleQubitGate
from gate_tensor import GateTensor
from qft import QFT
import numpy as np

def main():
    q0 = Qubit(1,0)
    q1 = Qubit(0,1)

    mt = MultiQubit()
    mt.add_qubit(q0)
    mt.add_qubit(q1)

    mt.print_tensor_form()
    mt.print_vector_form()

if __name__ == "__main__":
    main()