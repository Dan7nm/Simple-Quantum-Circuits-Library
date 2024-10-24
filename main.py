from qubit import Qubit
from qubit_tensor import QubitTensor
from gate import SingleQubitGate
from gate_tensor import GateTensor
import numpy as np

def main():
    q1 = Qubit(1,0)
    q2 = Qubit(0,1)
    t = QubitTensor()
    t.add_qubit(q1)
    t.add_qubit(q2)
    t.print_tensor_form()
    t.print_vector_form()
    x = SingleQubitGate('X')
    identity = SingleQubitGate()
    oper = GateTensor()
    oper.add_single_qubit_gate(x)
    oper.add_single_qubit_gate(identity)
    oper.print_matrix()
    t_new = oper.apply_operator(t)
    t_new.print_vector_form()
    t_new.print_tensor_form()


if __name__ == "__main__":
    main()