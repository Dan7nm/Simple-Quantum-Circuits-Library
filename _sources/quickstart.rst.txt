Quickstart Guide
================

This guide will help you get started with running simple QFT simulations using this library.

Introduction
------------

The **Quantum Fourier Transform (QFT)** is a fundamental operation in quantum computing that serves as the quantum analogue of the classical discrete Fourier transform (DFT). It plays a crucial role in many quantum algorithms, including Shor's algorithm for factoring large numbers, which has profound implications for cryptography. In this simulation, we explore the implementation and behavior of the Quantum Fourier Transform, demonstrating how quantum states are transformed in a superposition of basis states. By leveraging the power of quantum parallelism, this simulation provides insight into how QFT can efficiently process information in ways that are exponentially faster than classical methods for certain tasks. Through step-by-step visualization and quantum circuit manipulation, this simulation aims to deepen the understanding of QFT's mechanics and its potential applications in real-world quantum algorithms.

Installation
------------

First, ensure you have NumPy installed::

    pip install numpy

Second, clone the repository using git::

    git clone https://github.com/Dan7nm/Simple-Quantum-Circuits-Library.git 

Or you can go to the project link in on the homepage.

Basic Usage
-----------

Here's a complete example demonstrating how to create qubits, combine them into a tensor product, and apply the QFT:

.. code-block:: python

    from qubit import Qubit
    from qubit_tensor import QubitTensor
    from qft import QFT
    import numpy as np

    # Create individual qubits
    # Let's create a two-qubit system in state |00⟩
    q1 = Qubit(1.0, 0.0)  # |0⟩ state
    q2 = Qubit(1.0, 0.0)  # |0⟩ state

    # Create a tensor product of the qubits
    tensor = QubitTensor()
    tensor.add_qubit(q1)
    tensor.add_qubit(q2)

    # Print initial state
    print("Initial state:")
    tensor.print_tensor_form()

    # Apply QFT
    qft = QFT(tensor)
    result = qft.get_result()

    # Print final state
    print("\nAfter QFT:")
    result.print_tensor_form()

Step-by-Step Explanation
------------------------

1. Creating Qubits
~~~~~~~~~~~~~~~~~~

Each qubit is initialized with amplitudes for the :math:`|0⟩` and :math:`|1⟩` states:

.. code-block:: python

    # Create a qubit in state |0⟩
    qubit = Qubit(alpha=1.0, beta=0.0)

    # Create a qubit in state |1⟩
    qubit = Qubit(alpha=0.0, beta=1.0)

    # Create a qubit in superposition (|0⟩ + |1⟩)/√2
    qubit = Qubit(alpha=1/np.sqrt(2), beta=1/np.sqrt(2))

2. Creating Qubit Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple qubits using the QubitTensor class:

.. code-block:: python

    tensor = QubitTensor()
    
    # Add qubits one at a time
    tensor.add_qubit(q1)
    tensor.add_qubit(q2)
    
    # View the state
    tensor.print_tensor_form()

3. Applying QFT
~~~~~~~~~~~~~~~

The QFT is applied by creating a QFT object with your tensor:

.. code-block:: python

    # Apply QFT
    qft = QFT(tensor)
    
    # Get the result
    result = qft.get_result()
    
    # View the transformation matrix
    qft.print_operator()

Example: Creating a Bell State and Applying QFT
-----------------------------------------------

Here's a more complex example creating a Bell state and applying QFT:

.. code-block:: python

    # Create a Bell state (|00⟩ + |11⟩)/√2
    q1 = Qubit(1/np.sqrt(2), 1/np.sqrt(2))  # (|0⟩ + |1⟩)/√2
    q2 = Qubit(1.0, 0.0)                    # |0⟩

    tensor = QubitTensor()
    tensor.add_qubit(q1)
    tensor.add_qubit(q2)

    # Apply Hadamard gate to first qubit and CNOT
    h_gate = SingleQubitGate('H')
    gate_tensor = GateTensor()
    gate_tensor.add_single_qubit_gate(h_gate)
    gate_tensor.add_single_qubit_gate(SingleQubitGate('I'))
    
    # Apply gates
    tensor = gate_tensor.apply_operator(tensor)
    
    # Apply QFT
    qft = QFT(tensor)
    result = qft.get_result()

    print("Final state after QFT:")
    result.print_tensor_form()

Tips and Notes
--------------

1. Always ensure your qubits are properly normalized (α² + β² = 1)
2. The QFT implementation automatically handles the phase rotations and Hadamard gates
3. You can view intermediate states using print_tensor_form() or print_vector_form()
4. The number of qubits determines the size of the transformation matrix (2^n × 2^n)

Common Issues
-------------

1. **Invalid Amplitudes**: Ensure qubit amplitudes satisfy normalization
2. **Dimension Mismatch**: Check that gate operations match tensor dimensions
3. **Phase Errors**: Be careful with relative phases when creating qubits

For complete API details, refer to the individual class documentation in the source files.