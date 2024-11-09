from gate import Gate
import numpy as np

def main():
    gate = Gate()
    gate.set_controlled_qubit_gate(1,2,"P",np.pi/2)
    gate.print_matrix()
    print(gate.get_control_index())
    print(gate.get_target_index())



if __name__ == "__main__":
    main()