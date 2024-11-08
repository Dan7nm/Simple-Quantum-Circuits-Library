import tkinter as tk
from tkinter import ttk
from gate import SingleQubitGate
from multi_qubit import MultiQubit
from qubit import Qubit
from circuit import Circuit

class CircuitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Circuit Simulator")
        self.setup_ui()
        
        # Initialize quantum circuit components
        self.multi_qubit = MultiQubit()
        self.gate_tensor = Circuit()
        self.circuit_diagram = []  # Stores qubit and gate actions for display

    def setup_ui(self):
        # Frame for circuit controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        # Add Qubit section
        ttk.Label(control_frame, text="Add Qubit (α,β):").grid(row=0, column=0, padx=5)
        self.alpha_entry = ttk.Entry(control_frame, width=5)
        self.alpha_entry.grid(row=0, column=1)
        self.beta_entry = ttk.Entry(control_frame, width=5)
        self.beta_entry.grid(row=0, column=2)
        add_qubit_button = ttk.Button(control_frame, text="Add Qubit", command=self.add_qubit)
        add_qubit_button.grid(row=0, column=3, padx=5)

        # Add Gate section
        ttk.Label(control_frame, text="Add Gate:").grid(row=1, column=0, padx=5)
        self.gate_type = ttk.Combobox(control_frame, values=["I", "X", "Y", "Z", "H", "P", "R"])
        self.gate_type.set("I")
        self.gate_type.grid(row=1, column=1)
        self.angle_entry = ttk.Entry(control_frame, width=5)
        self.angle_entry.grid(row=1, column=2)
        add_gate_button = ttk.Button(control_frame, text="Add Gate", command=self.add_gate)
        add_gate_button.grid(row=1, column=3, padx=5)

        # Circuit diagram display
        ttk.Label(self.root, text="Circuit Diagram:").pack()
        self.circuit_display = tk.Text(self.root, height=10, width=50)
        self.circuit_display.pack(pady=5)

        # Output section
        ttk.Label(self.root, text="Output:").pack()
        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.pack(pady=5)
        
        # Apply and Reset buttons
        apply_button = ttk.Button(self.root, text="Apply Circuit", command=self.apply_circuit)
        apply_button.pack(side=tk.LEFT, padx=10)
        reset_button = ttk.Button(self.root, text="Reset Circuit", command=self.reset_circuit)
        reset_button.pack(side=tk.RIGHT, padx=10)

    def add_qubit(self):
        try:
            alpha = complex(self.alpha_entry.get())
            beta = complex(self.beta_entry.get())
            qubit = Qubit(alpha, beta)
            self.multi_qubit.add_qubit(qubit)
            qubit_repr = f"{alpha}|0⟩+{beta}|1⟩:"  # Each new qubit starts with this symbol
            self.circuit_diagram.append([qubit_repr])  # Each qubit line is a list of gate representations
            self.update_circuit_display()
            self.output_text.insert(tk.END, f"Qubit added: |{alpha}, {beta}>\n")
        except ValueError as e:
            self.output_text.insert(tk.END, f"Error: {e}\n")

    def add_gate(self):
        gate_type = self.gate_type.get()
        phi = float(self.angle_entry.get()) if gate_type == "P" else 0.0
        gate = SingleQubitGate(gate_type, phi)
        self.gate_tensor.add_single_qubit_gate(gate)
        
        # Represent the gate in the circuit
        # if gate_type == "H":
        #     gate_repr = "[H]"
        # elif gate_type == "R":
        #     gate_repr = "[R]"
        # else:
        #     gate_repr = f"[{gate_type}]"
        gate_repr = f"[{gate_type}]"

        # Add gate to each qubit's line representation
        for qubit_line in self.circuit_diagram:
            qubit_line.append(f"--{gate_repr}--")

        self.update_circuit_display()
        self.output_text.insert(tk.END, f"Gate applied: {gate_type}\n")

    def apply_circuit(self):
        try:
            result_multi_qubit = self.gate_tensor.apply_operator(self.multi_qubit)  # Assuming this method applies all gates
            result = result_multi_qubit.get_tensor_vector()
            self.output_text.insert(tk.END, f"Final State Tensor:\n{result}\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error applying circuit: {e}\n")

    def update_circuit_display(self):
        # Refresh the circuit diagram display
        self.circuit_display.delete(1.0, tk.END)
        for qubit_line in self.circuit_diagram:
            self.circuit_display.insert(tk.END, "".join(qubit_line) + "\n")

    def reset_circuit(self):
        # Reset multi-qubit and gate tensor
        self.multi_qubit = MultiQubit()
        self.gate_tensor = GateTensor()
        self.circuit_diagram.clear()
        self.output_text.delete(1.0, tk.END)
        self.circuit_display.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Circuit reset.\n")

# Running the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitApp(root)
    root.mainloop()
