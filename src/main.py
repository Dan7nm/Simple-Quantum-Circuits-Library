from multi_qubit import MultiQubit
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   circuit = QuantumCircuit(number_of_qubits=4,num_of_layers=2)
   circuit.add_controlled_qubit_gate(0,0,1,"X")
   circuit.add_controlled_qubit_gate(2,0,3,"Y")
   circuit.add_swap_gate(0,3,1)
   circuit.add_single_qubit_gate(1,1,"Z")
   circuit.draw_circuit()


   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()