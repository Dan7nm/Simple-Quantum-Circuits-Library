from qubit import Qubit
from qubit_tensor import QubitTensor
from gate import SingleQubitGate
from gate_tensor import GateTensor
from qft import QFT
import numpy as np

def main():
    q1 = Qubit(1,0)
    q2 = Qubit(0,1)

    t = QubitTensor()
    t.add_qubit(q2)
    t.add_qubit(q1)
    t.add_qubit(q2)

    t.print_tensor_form()

    qft = QFT(t)

    # qft.print_operator()
    result_tensor = qft.get_result()
    result_tensor.print_tensor_form()

if __name__ == "__main__":
    main()