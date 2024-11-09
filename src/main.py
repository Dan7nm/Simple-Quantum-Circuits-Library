from gate import Gate
import numpy as np

def main():
    gate = Gate()
    gate.set_single_qubit_gate("H")
    gate.print_matrix()
if __name__ == "__main__":
    main()