from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   qubit_num = 5
   vector = np.zeros(2**qubit_num)
   vector[0]=1
   mt = MultiQubit(vector)
   c_reg = ClassicalRegister(qubit_num)
   mt.print_tensor_form()
   circuit = QuantumCircuit(mt,c_reg)
   circuit.load_dynamic_qft_preset()
   circuit.draw_circuit()
   result = circuit.run_circuit()
   result.print_tensor_form()

   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()