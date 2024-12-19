from multi_qubit import MultiQubit
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   circuit = QuantumCircuit(3,num_of_layers=3)
   circuit.add_controlled_qubit_gate(0,0,1,"X")
   circuit.add_measure_gate(0,1)
   circuit.add_measure_gate(2,2)
   circuit.add_single_qubit_gate(2,0,"Z")
   circuit.draw_circuit()


   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()