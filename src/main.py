# from circuit import QuantumCircuit,QFTCircuit
# from qubit import Qubit
from multi_qubit import MultiQubit
import numpy as np
import time

def main():
   start_time = time.perf_counter()
   vector = np.full(16,1/4)
   mt = MultiQubit(vector)
   mt.print_tensor_form()
   for i in range(4):
       mt = mt.measure_qubit(i)
       mt.print_tensor_form()

   end_time = time.perf_counter()
   run_time = end_time - start_time
   print(f"===== Finished running in {run_time:.2f} seconds. =====")

if __name__ == "__main__":
    main()