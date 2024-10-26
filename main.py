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
    t.add_qubit(q1)
    gate = SingleQubitGate("P",np.pi)
    gt = GateTensor()
    gt.add_controlled_gate(0,1,t,gate)
    gt.print_matrix()

if __name__ == "__main__":
    main()