from gate import Gate
import numpy as np

def main():
    gate = Gate()
    gate.set_single_qubit_gate("P",np.pi/2)
    gate.print_matrix()
    matrix = gate.get_matrix()
    print(matrix)

if __name__ == "__main__":
    main()