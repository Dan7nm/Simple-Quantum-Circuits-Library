Quantum Gates
=============

In quantum computing, gates manipulate qubits through various transformations. Here we describe some of the most commonly used gates in quantum circuits, including the X, Y, Z, Hadamard, Phase Shift, and Swap gates.

Single Qubit Gates
------------------

I Gate (The identity gate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../img/Qcircuit_I.svg
   :alt: I Gate
   :width: 250px
   :align: center

The I gate or the identity gate is the identity matrix and is defined as follows:

.. math::
   I =  \begin{pmatrix}
         1 & 0 \\
         0 & 1
      \end{pmatrix}

The identity gate doesn't change the the quantum state or the qubit state. This property is useful in some cases.

X Gate (Pauli-X)
~~~~~~~~~~~~~~~~

.. image:: ../../img/Qcircuit_NOT.svg
   :alt: NOT Gate
   :width: 250px
   :align: center

The X gate, also known as the Pauli-X gate or the quantum NOT gate, flips the state of a qubit from |0⟩ to |1⟩ or from |1⟩ to |0⟩.

.. math::

   X = \begin{pmatrix}
   0 & 1 \\
   1 & 0
   \end{pmatrix}

This gate acts similarly to the classical NOT gate, flipping the value of the qubit. For example, if the input state is |0⟩, applying an X gate transforms it to |1⟩, and vice versa.

Y Gate (Pauli-Y)
~~~~~~~~~~~~~~~~

.. image:: ../../img/Qcircuit_Y.svg
   :alt: Y Gate
   :width: 250px
   :align: center


The Y gate, also known as the Pauli-Y gate, combines the flipping of the qubit’s state with a phase shift of π radians.

.. math::

   Y = \begin{pmatrix}
   0 & -i \\
   i & 0
   \end{pmatrix}

The Y gate flips the qubit like the X gate but introduces an imaginary phase factor. It transforms the |0⟩ state into i|1⟩ and the |1⟩ state into -i|0⟩.

Z Gate (Pauli-Z)
~~~~~~~~~~~~~~~~

.. image:: ../../img/Qcircuit_Z.svg
   :alt: NOT Gate
   :width: 250px
   :align: center

The Z gate, also known as the Pauli-Z gate, applies a phase shift of π radians to the |1⟩ state while leaving the |0⟩ state unchanged.

.. math::

   Z = \begin{pmatrix}
   1 & 0 \\
   0 & -1
   \end{pmatrix}

In effect, the Z gate flips the phase of the |1⟩ state but doesn’t affect the computational basis of the qubit. This gate can be used for phase inversion operations.

H Gate (Hadamard Gate)
~~~~~~~~~~~~~~~~

.. image:: ../../img/Hadamard_gate.svg
   :alt: NOT Gate
   :width: 250px
   :align: center

The Hadamard gate creates a superposition of the |0⟩ and |1⟩ states. When applied to a qubit in the |0⟩ state, it generates an equal superposition of |0⟩ and |1⟩, sometimes denoted as the plus state:

.. math::

    |+⟩ = \frac{1}{\sqrt{2}}(|0⟩+|1⟩)

And when applied on the |1⟩ state, it generates the minus state:

.. math::

    |-⟩ = \frac{1}{\sqrt{2}}(|0⟩-|1⟩)

The Hadamard gate matrix:

.. math::

   H = \frac{1}{\sqrt{2}} \begin{pmatrix}
   1 & 1 \\
   1 & -1
   \end{pmatrix}

The Hadamard gate plays a crucial role in quantum algorithms such as QFT, creating superpositions and allowing interference patterns to emerge in quantum computations.

R Gate (Phase Shift Gate)
~~~~~~~~~~~~~~~~
The Phase Shift gate introduces a phase shift of φ to the |1⟩ state while leaving the |0⟩ state unchanged. It generalizes the Z gate by allowing for arbitrary phase shifts.

.. math::

   R_\phi = \begin{pmatrix}
   1 & 0 \\
   0 & e^{i\phi}
   \end{pmatrix}

The Phase Shift gate is important in many quantum algorithms where phase manipulation is needed, such as in the Quantum Fourier Transform.

Two Qubit Gates
---------------

CNOT Gate
~~~~~~~~~

The CNOT (Controlled-NOT) gate is a fundamental two-qubit gate used in quantum computing. It operates on two qubits: a control qubit and a target qubit. The CNOT gate flips the state of the target qubit (applies an X gate) if the control qubit is in the |1⟩ state. If the control qubit is in the |0⟩ state, the target qubit remains unchanged.

The matrix representation of the CNOT gate is:

.. math::

    \text{CNOT} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 0
    \end{pmatrix}

The truth table for the CNOT gate is as follows:

.. list-table::
   :header-rows: 1

   * - Control Qubit
     - Target Qubit (Input)
     - Target Qubit (Output)
   * - 0
     - 0
     - 0
   * - 0
     - 1
     - 1
   * - 1
     - 0
     - 1
   * - 1
     - 1
     - 0

Controlled Z Gate
~~~~~~~~~~~~~~~~~

The CZ (Controlled-Z) gate is another two-qubit gate where the second qubit (target qubit) experiences a phase flip (Z gate) if the control qubit is in the |1⟩ state. Unlike the CNOT gate, the CZ gate does not flip the target qubit's value, but applies a phase change (sign flip) to the target qubit when the control qubit is in the |1⟩ state.

The matrix representation of the CZ gate is:

.. math::

    \text{CZ} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & -1
    \end{pmatrix}

The truth table for the CZ gate is as follows:

.. list-table::
   :header-rows: 1

   * - Control Qubit
     - Target Qubit (Input)
     - Target Qubit (Output)
   * - 0
     - 0
     - 0
   * - 0
     - 1
     - 1
   * - 1
     - 0
     - 0
   * - 1
     - 1
     - -1 (Phase Flip)


Swap Gate
~~~~~~~~~
The Swap gate exchanges the states of two qubits.

.. math::

   SWAP = \begin{pmatrix}
   1 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 1
   \end{pmatrix}

It swaps the qubit states |01⟩ and |10⟩, leaving |00⟩ and |11⟩ unchanged. The Swap gate is useful for rearranging qubits within quantum circuits.

Gates Summary
------------
Here’s a summary of how each gate operates on the standard computational basis:

- **X Gate**: Flips |0⟩ to |1⟩ and |1⟩ to |0⟩.
- **Y Gate**: Flips the state and adds a phase of π to the |1⟩ state.
- **Z Gate**: Adds a phase of π to the |1⟩ state.
- **Hadamard Gate**: Creates a superposition of |0⟩ and |1⟩.
- **Phase Shift Gate**: Adds a phase of φ to the |1⟩ state.
- **Swap Gate**: Exchanges the states of two qubits.

