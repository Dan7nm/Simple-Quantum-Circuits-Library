import tkinter as tk
from tkinter import ttk, messagebox
from qubit import Qubit
from multi_qubit import MultiQubit
from circuit import Circuit
from typing import List, Optional
import re
import io
import sys

class QuantumCircuitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Circuit Builder")
        self.root.geometry("800x700")
        
        # State variables
        self.multi_qubit: Optional[MultiQubit] = None
        self.circuit: Optional[Circuit] = None
        self.current_layer = 0
        self.qubit_states: List[tuple] = []
        
        self.create_gui_elements()
        
    def format_quantum_state(self, state_type="Initial"):
        """Captures and formats the quantum state display."""
        output = io.StringIO()
        sys.stdout = output
        
        if state_type == "Initial" and self.multi_qubit:
            self.multi_qubit.print_tensor_form()
        elif state_type == "Final" and self.circuit:
            result = self.circuit.apply_circuit()
            result.print_tensor_form()
            
        sys.stdout = sys.__stdout__
        state_str = output.getvalue()
        
        # Format the display with a clear header and border
        formatted_display = f"\n{'='*20} {state_type} State {'='*20}\n"
        formatted_display += state_str
        formatted_display += f"{'='*50}\n"
        
        return formatted_display
        
    def create_gui_elements(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Qubit Initialization Section
        self.create_qubit_init_section(main_frame)
        
        # Circuit Building Section
        self.create_circuit_section(main_frame)
        
        # Display Section
        self.create_display_section(main_frame)
        
        # State Display Section
        self.create_state_display_section(main_frame)
        
    def create_qubit_init_section(self, parent):
        # Qubit Initialization Frame
        init_frame = ttk.LabelFrame(parent, text="Qubit Initialization", padding="5")
        init_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add qubit entry fields
        ttk.Label(init_frame, text="α:").grid(row=0, column=0, padx=5)
        self.alpha_entry = ttk.Entry(init_frame, width=10)
        self.alpha_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(init_frame, text="β:").grid(row=0, column=2, padx=5)
        self.beta_entry = ttk.Entry(init_frame, width=10)
        self.beta_entry.grid(row=0, column=3, padx=5)
        
        ttk.Button(init_frame, text="Add Qubit", command=self.add_qubit).grid(row=0, column=4, padx=5)
        ttk.Button(init_frame, text="Initialize Circuit", command=self.initialize_circuit).grid(row=0, column=5, padx=5)
        
    def create_circuit_section(self, parent):
        # Circuit Controls Frame
        circuit_frame = ttk.LabelFrame(parent, text="Circuit Controls", padding="5")
        circuit_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Gate controls
        ttk.Label(circuit_frame, text="Gate Type:").grid(row=0, column=0, padx=5)
        self.gate_type = ttk.Combobox(circuit_frame, values=['X', 'Y', 'Z', 'H', 'P'])
        self.gate_type.grid(row=0, column=1, padx=5)
        
        ttk.Label(circuit_frame, text="Target Qubit:").grid(row=0, column=2, padx=5)
        self.target_qubit = ttk.Entry(circuit_frame, width=5)
        self.target_qubit.grid(row=0, column=3, padx=5)
        
        ttk.Label(circuit_frame, text="Control Qubit:").grid(row=0, column=4, padx=5)
        self.control_qubit = ttk.Entry(circuit_frame, width=5)
        self.control_qubit.grid(row=0, column=5, padx=5)
        
        ttk.Label(circuit_frame, text="Phase (φ):").grid(row=0, column=6, padx=5)
        self.phase_entry = ttk.Entry(circuit_frame, width=5)
        self.phase_entry.insert(0, "0.0")
        self.phase_entry.grid(row=0, column=7, padx=5)
        
        # Buttons
        ttk.Button(circuit_frame, text="Add Single Gate", command=self.add_single_gate).grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(circuit_frame, text="Add Controlled Gate", command=self.add_controlled_gate).grid(row=1, column=2, columnspan=2, pady=5)
        ttk.Button(circuit_frame, text="Add Layer", command=self.add_layer).grid(row=1, column=4, columnspan=2, pady=5)
        ttk.Button(circuit_frame, text="Compute Circuit", command=self.compute_circuit).grid(row=1, column=6, columnspan=2, pady=5)
        
    def create_display_section(self, parent):
        # Display Frame
        display_frame = ttk.LabelFrame(parent, text="Circuit Display", padding="5")
        display_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Text widget for displaying circuit and states
        self.display_text = tk.Text(display_frame, height=20, width=80)
        self.display_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.display_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.display_text.configure(yscrollcommand=scrollbar.set)
        
    def create_state_display_section(self, parent):
        """Creates a dedicated section for quantum state display."""
        state_frame = ttk.LabelFrame(parent, text="Quantum States", padding="5")
        state_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Text widget specifically for states
        self.state_text = tk.Text(state_frame, height=10, width=80)
        self.state_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for state display
        state_scrollbar = ttk.Scrollbar(state_frame, orient=tk.VERTICAL, command=self.state_text.yview)
        state_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.state_text.configure(yscrollcommand=state_scrollbar.set)
        
    def add_qubit(self):
        try:
            alpha = complex(self.alpha_entry.get())
            beta = complex(self.beta_entry.get())
            
            # Validate quantum state
            if abs(abs(alpha)**2 + abs(beta)**2 - 1.0) > 1e-10:
                raise ValueError("Invalid quantum state: |α|² + |β|² must equal 1")
            
            self.qubit_states.append((alpha, beta))
            self.update_display(f"\nAdded qubit with α={alpha}, β={beta}")
            
            # Clear entries
            self.alpha_entry.delete(0, tk.END)
            self.beta_entry.delete(0, tk.END)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            
    def initialize_circuit(self):
        if not self.qubit_states:
            messagebox.showerror("Error", "Add at least one qubit first!")
            return
            
        self.multi_qubit = MultiQubit()
        for alpha, beta in self.qubit_states:
            self.multi_qubit.add_qubit(Qubit(alpha, beta))
            
        self.circuit = Circuit(self.multi_qubit)
        
        # Display initial state
        self.state_text.delete(1.0, tk.END)
        initial_state = self.format_quantum_state("Initial")
        self.state_text.insert(tk.END, initial_state)
        
        # Display circuit
        self.update_display("\n=== Circuit Initialized ===")
        self.display_circuit()
        
    def add_single_gate(self):
        if not self.circuit:
            messagebox.showerror("Error", "Initialize the circuit first!")
            return
            
        try:
            target = int(self.target_qubit.get())
            gate = self.gate_type.get()
            phi = float(self.phase_entry.get())
            
            self.circuit.add_single_qubit_gate(target, self.current_layer, gate, phi)
            self.update_display(f"\nAdded {gate} gate to qubit {target}")
            self.display_circuit()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            
    def add_controlled_gate(self):
        if not self.circuit:
            messagebox.showerror("Error", "Initialize the circuit first!")
            return
            
        try:
            target = int(self.target_qubit.get())
            control = int(self.control_qubit.get())
            gate = self.gate_type.get()
            phi = float(self.phase_entry.get())
            
            self.circuit.add_controlled_qubit_gate(target, self.current_layer, control, gate, phi)
            self.update_display(f"\nAdded controlled-{gate} gate between control {control} and target {target}")
            self.display_circuit()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            
    def add_layer(self):
        if not self.circuit:
            messagebox.showerror("Error", "Initialize the circuit first!")
            return
            
        self.circuit.add_layer()
        self.current_layer += 1
        self.update_display(f"\nAdded new layer {self.current_layer}")
        self.display_circuit()
        
    def compute_circuit(self):
        if not self.circuit:
            messagebox.showerror("Error", "Initialize the circuit first!")
            return
            
        try:
            self.circuit.compute_circuit()
            
            # Display final state
            final_state = self.format_quantum_state("Final")
            self.state_text.insert(tk.END, final_state)
            self.state_text.see(tk.END)
            
            # Update main display
            self.update_display("\n=== Circuit Computed Successfully ===")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def update_display(self, text):
        self.display_text.insert(tk.END, f"{text}\n")
        self.display_text.see(tk.END)
        
    def display_circuit(self):
        # Redirect print output to capture circuit diagram
        import io
        import sys
        
        # Create a string buffer and redirect stdout
        output = io.StringIO()
        sys.stdout = output
        
        # Print the circuit
        self.circuit.print_circuit()
        
        # Reset stdout and get the output
        sys.stdout = sys.__stdout__
        circuit_str = output.getvalue()
        
        # Update display
        self.update_display(circuit_str)

def main():
    root = tk.Tk()
    app = QuantumCircuitGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()