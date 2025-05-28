from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   mt = MultiQubit(qubits_num=3)
   # mt.print_tensor_form()
   classical_reg = ClassicalRegister(3)
   dynamic_circuit = QuantumCircuit(mt,classical_register=classical_reg)
   reg_circuit = QuantumCircuit(mt)
   reg_circuit.load_qft_preset()
   dynamic_circuit.load_dynamic_qft_preset()
   reg_res = reg_circuit.run_circuit()
   dyn_res = dynamic_circuit.run_many(10000)
   reg_res.plot_probabilities()
   dyn_res.plot_probabilities()


   

   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()