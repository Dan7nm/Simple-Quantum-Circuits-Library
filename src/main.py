from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   mt = MultiQubit(qubits_num=3)
   circuit = QuantumCircuit(mt)
   circuit.load_qft_preset()

   result = circuit.run_circuit()

   result2 = circuit.run_many(num_of_runs=10000)

   result.plot_probabilities()
   result2.plot_probabilities()

   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()