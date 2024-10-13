from qubit import Qubit
import numpy as np

def main():
    alpha = 1/np.sqrt(2)
    beta = 1/np.sqrt(2)
    q = Qubit(alpha,beta)
    q.print_qubit()
    q.set_rel_ph_one(1)
    q.print_qubit()


if __name__ == "__main__":
    main()