from multi_qubit import MultiQubit
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   circuit = QuantumCircuit(number_of_qubits=5)
   circuit.load_qft_preset()
   circuit.draw_circuit()


   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()