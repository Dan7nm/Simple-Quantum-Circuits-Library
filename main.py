from qubit import Qubit
from qubit_tensor import QubitTensor
from gate import SingleQubitGate
import numpy as np

def main():
    q1 = Qubit(0,1)
    q1.print_vector_form()
    g1 = SingleQubitGate("R",np.pi)
    q2 = g1.apply_matrix(q1)
    q2.print_vector_form()
    q2.print_qubit()

if __name__ == "__main__":
    main()