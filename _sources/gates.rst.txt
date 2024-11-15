Quantum Gates
=============

In quantum computing, gates manipulate qubits through various transformations. Here we describe some of the most commonly used gates in quantum circuits, including the X, Y, Z, Hadamard, Phase Shift, and Swap gates.

Single Qubit Gates
------------------

I Gate (The identity gate)
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

The X gate, also known as the Pauli-X gate or the quantum NOT gate, flips the state of a qubit from :math:`|0⟩` to :math:`|1⟩` or from :math:`|1⟩` to :math:`|0⟩`.

.. math::

   X = \begin{pmatrix}
   0 & 1 \\
   1 & 0
   \end{pmatrix}

This gate acts similarly to the classical NOT gate, flipping the value of the qubit. For example, if the input state is :math:`|0⟩`, applying an X gate transforms it to :math:`|1⟩`, and vice versa.

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

The Y gate flips the qubit like the X gate but introduces an imaginary phase factor. It transforms the :math:`|0⟩` state into i:math:`|1⟩` and the :math:`|1⟩` state into -i:math:`|0⟩`.

Z Gate (Pauli-Z)
~~~~~~~~~~~~~~~~

.. image:: ../../img/Qcircuit_Z.svg
   :alt: NOT Gate
   :width: 250px
   :align: center

The Z gate, also known as the Pauli-Z gate, applies a phase shift of π radians to the :math:`|1⟩` state while leaving the :math:`|0⟩` state unchanged.

.. math::

   Z = \begin{pmatrix}
   1 & 0 \\
   0 & -1
   \end{pmatrix}

In effect, the Z gate flips the phase of the :math:`|1⟩` state but doesn’t affect the computational basis of the qubit. This gate can be used for phase inversion operations.

H Gate (Hadamard Gate)
~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../img/Hadamard_gate.svg
   :alt: NOT Gate
   :width: 250px
   :align: center

The Hadamard gate creates a superposition of the :math:`|0⟩` and :math:`|1⟩` states. When applied to a qubit in the :math:`|0⟩` state, it generates an equal superposition of :math:`|0⟩` and :math:`|1⟩`, sometimes denoted as the plus state:

.. math::

    |+⟩ = \frac{1}{\sqrt{2}}(|0⟩+|1⟩)

And when applied on the :math:`|1⟩` state, it generates the minus state:

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
~~~~~~~~~~~~~~~~~~~~~~~~~
The Phase Shift gate introduces a phase shift of φ to the :math:`|1⟩` state while leaving the :math:`|0⟩` state unchanged. It generalizes the Z gate by allowing for arbitrary phase shifts.

.. math::

   R_\phi = \begin{pmatrix}
   1 & 0 \\
   0 & e^{i\phi}
   \end{pmatrix}

The Phase Shift gate is important in many quantum algorithms where phase manipulation is needed, such as in the Quantum Fourier Transform.

Two Qubit Gates
---------------

The controlled-U gate, denoted as 'CU', is a general construction that applies the single-qubit unitary transformation U on the target qubit only if the control qubit is in the :math:`|1⟩` state. 

In a two qubit system where the first qubit is the control qubit and the second qubit is the target qubit, the dirac notation will be as follows:

.. math::

   CU = |0⟩⟨0| \otimes I + |1⟩⟨1| \otimes U

Where 'I' is the identity matrix. This means that if the control qubit is in the :math:`|0⟩` state, the target qubit remains unchanged, while if the control qubit is in the :math:`|1⟩` state, the transformation 'U' is applied to the target qubit.

In a general case where the control qubit is the i-th qubit and the target is the j qubit, the dirac notation will be as follows:

.. math::

   CU = I ^{\otimes i-1}\otimes (|0⟩⟨0| \otimes I ^{\otimes l} \otimes I + |1⟩⟨1| \otimes I ^{\otimes l} \otimes U )\otimes I^{\otimes m} 

Where r is the number of qubits between i and j and m is the number of qubits after the j-th qubit. This library computes controlled gates based on this notation.

The projection matrix :math:`|0⟩⟨0|` is applied on the i-th qubit in this tensor product and the identity matrix is applied on the j-th qubit. If the control qubit is in :math:`|0⟩` state than j-th qubit is unchanged. However, if the control qubit is in :math:`|1⟩` state then the projection matrix is applied :math:`|1⟩⟨1|` thus unitary Matrix U is applied on the j-th qubit.

Matrix representation for a controlled-U gate:

.. math::

   CU = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & u_{00} & u_{01} \\
            0 & 0 & u_{10} & u_{11}
   \end{pmatrix}

CNOT Gate (Controlled X Gate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CNOT (Controlled-NOT) gate is a fundamental two-qubit gate used in quantum computing. It operates on two qubits: a control qubit and a target qubit. The CNOT gate flips the state of the target qubit (applies an X gate) if the control qubit is in the :math:`|1⟩` state. If the control qubit is in the :math:`|0⟩` state, the target qubit remains unchanged.

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

The CZ (Controlled-Z) gate is another two-qubit gate where the second qubit (target qubit) experiences a phase flip (Z gate) if the control qubit is in the :math:`|1⟩` state. Unlike the CNOT gate, the CZ gate does not flip the target qubit's value, but applies a phase change (sign flip) to the target qubit when the control qubit is in the :math:`|1⟩` state.

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

The matrix representation for a two qubit system is as follows:

.. math::

   SWAP = \begin{pmatrix}
   1 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 1
   \end{pmatrix}

It swaps the qubit states :math:`|01⟩` and :math:`|10⟩`, leaving :math:`|00⟩` and :math:`|11⟩` unchanged. The Swap gate is useful for rearranging qubits within quantum circuits.

In a general case for n qubit system, the dirac notation for a swap operation is as follows:

.. math::

   SWAP_{i,j} = \sum_{x_1, x_2, \dots, x_n \in \{0,1\}} | x_1 \dots x_j \dots x_i \dots x_n \rangle \langle x_1 \dots x_i \dots x_j \dots x_n |
   

Gates Summary
-------------
Here’s a summary of how each gate operates on the standard computational basis:

- **I Gate**: Doesn't change the state
- **X Gate**: Flips :math:`|0⟩` to :math:`|1⟩` and :math:`|1⟩` to :math:`|0⟩`.
- **Y Gate**: Flips the state and adds a phase of π to the :math:`|1⟩` state.
- **Z Gate**: Adds a phase of π to the :math:`|1⟩` state.
- **H Gate**: Creates a superposition of :math:`|0⟩` and :math:`|1⟩`.
- **R Gate**: Adds a phase of φ to the :math:`|1⟩` state.
- **Controlled U Gate**: Applies the unitary matrix U on the target qubit state if the control qubit is :math:`|1⟩`.
- **CNOT Gate**: Flips the target qubit state if the control qubit is :math:`|1⟩`.
- **CNOT Gate**: Adds phase change target qubit state if the control qubit is :math:`|1⟩`.
- **Swap Gate**: Exchanges the states of two qubits.

