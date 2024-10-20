from qubit import Qubit
from qubit_tensor import QubitTensor
from gate import SingleQubitGate
import numpy as np

def main():
    # q1 = Qubit(1,0)
    # q2 = Qubit(0,1)
    # alpha = 1/np.sqrt(2)
    # beta = 1/np.sqrt(2)
    # q3 = Qubit(alpha,beta)
    # qt = QubitTensor()
    # qt.add_qubit(q1)
    # qt.add_qubit(q2)
    # qt.add_qubit(q3)
    # qt.print_vector_form()
    # qt.print_tensor_form()

    g1 = SingleQubitGate("Y")
    g1.print_matrix()
    print(g1.get_matrix())

if __name__ == "__main__":
    main()