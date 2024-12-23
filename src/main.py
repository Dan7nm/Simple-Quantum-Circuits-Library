from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   mt = MultiQubit(np.full(4,1/2))
   creg = ClassicalRegister(2)
   mt.print_tensor_form()
   circuit = QuantumCircuit(mt,creg)
   circuit.add_measure_gate(0,0,0)
   circuit.add_measure_gate(1,0,1)
   circuit.draw_circuit()
   result = circuit.run_circuit()
   result.print_tensor_form()
   creg.print_c_reg()

   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()