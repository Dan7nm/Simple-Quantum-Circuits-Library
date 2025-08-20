from multi_qubit import MultiQubit
from c_register import ClassicalRegister
import numpy as np
import time
from circuit import QuantumCircuit
def main():
   start_time = time.perf_counter()
   
   # Write code here:
   mt = MultiQubit(qubits_num=5)
   c_reg = ClassicalRegister(num_bits=5)
   circuit = QuantumCircuit(input_state=mt,classical_register=c_reg)
   circuit.load_qft_preset(include_measurement=True)
   circuit.draw_circuit()


   # Calculate the runtime:
   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()