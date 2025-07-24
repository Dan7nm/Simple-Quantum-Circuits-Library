from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   mt = MultiQubit(vector=np.array([1,0,0,0,0,0,0,0], dtype=complex))
   circuit = QuantumCircuit(mt,error_magnitude=0.0)
   mt.print_tensor_form()
   circuit.load_qft_preset()

   # circuit.draw_circuit()

   result = circuit.run_circuit()

   result.plot_amplitudes()

   # mt = MultiQubit(vector=np.array([1,0,0,0], dtype=complex))
   # circuit = QuantumCircuit(mt,error_magnitude=0.0)
   # mt.print_tensor_form()

   # circuit.add_single_qubit_gate(0,0,'X',0.0)
   # # circuit.draw_circuit()

   # result = circuit.run_circuit()

   # result.print_tensor_form()


   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()