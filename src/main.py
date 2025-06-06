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
   clas_register = ClassicalRegister(3)
   dyn_circuit = QuantumCircuit(mt, clas_register)
   dyn_circuit.load_dynamic_qft_preset()
   circuit.load_qft_preset()
   
   result = circuit.run_circuit()
   result_dyn = dyn_circuit.run_many(100)

   result.plot_probabilities()
   result_dyn.plot_probabilities()

   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()