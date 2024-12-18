import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import random

EPSILON = 1e-10


INV_AMP = "Amplitude is invalid. The given amplitude needs to be between 0 and 1 and satisfy the normalization condition."

class Qubit:
    """
    :ivar alpha: The amplitude of the :math:`|0⟩` state, a value between 0 and 1.
    :vartype alpha: complex
    :ivar beta: The amplitude of the :math:`|1⟩` state, a value between 0 and 1.
    :vartype beta: complex
    
    A class to represent a quantum bit (Qubit) with two possible states: :math:`|0⟩` and :math:`|1⟩`,
    described by complex amplitudes alpha and beta.
    """

    def __init__(self, alpha: complex, beta: complex) -> None:
        """
        Constructs all the necessary attributes for the qubit object.

        :param alpha: The amplitude of the :math:`|0⟩` state, should be a complex number with amplitude between 0 and 1.
        :type alpha: complex
        :param beta: The amplitude of the :math:`|1⟩` state, should be a complex number with amplitude between 0 and 1.
        :type beta: complex
        :raises ValueError: If the given amplitudes do not satisfy the normalization condition.
        """
        if self._is_valid_amplitudes(alpha, beta):
            self.__alpha = alpha
            self.__beta = beta
            self.__qubit_vector = np.array([alpha,beta])
        else:
            raise ValueError(INV_AMP)

    def __complex_to_euler(self, z: complex) -> Tuple[float]:
        """
        Convert a complex number to its Euler form, r * e^(iθ).
        
        :param z: Complex number to convert
        :type z: complex
        :return: Magnitude and phase (r, theta)
        :rtype: tuple
        """
        r = abs(z)
        theta = np.angle(z)

        # If the number is really small, round the value
        if r < EPSILON:
            r = 0
        if theta < EPSILON:
            theta = 0
        return r, theta


    def _is_valid_amplitudes(self, alpha: complex, beta: complex) -> bool:
        """
        Checks if the given amplitudes are valid and satisfy the normalization condition.

        :param alpha: The amplitude of the :math:`|0⟩` state.
        :type alpha: complex
        :param beta: The amplitude of the :math:`|1⟩` state.
        :type beta: complex
        :return: True if the amplitudes are valid and their squares sum to 1, False otherwise.
        :rtype: bool
        """
        r_0, theta_0 = self.__complex_to_euler(alpha)
        r_1, theta_1 = self.__complex_to_euler(beta)
        return (0<= r_0 <= 1 and 0<= r_1 <= 1 and abs((r_0**2 + r_1**2) - 1) <= EPSILON)
        

    def get_alpha(self) -> complex:
        """
        Returns the amplitude of the :math:`|0⟩` state.

        :return: The amplitude of the :math:`|0⟩` state (alpha).
        :rtype: complex
        """
        return self.__alpha

    def get_beta(self) -> complex:
        """
        Returns the amplitude of the :math:`|1⟩` state.

        :return: The amplitude of the :math:`|1⟩` state (beta).
        :rtype: complex
        """
        return self.__beta
    
    def get_vector(self) -> NDArray[np.complex128]:
        """
        Returns the qubit in vector form.

        :return: The vector form of the qubit (__qubit_vector).
        :rtype: np.ndarray
        """
        return self.__qubit_vector

    def set_amplitudes(self, alpha: complex, beta: complex) -> None:
        """
        Sets the amplitude of the :math:`|0⟩` state and :math:`|1⟩` state.

        :param alpha: The amplitude of the :math:`|0⟩` state, should be a complex number with amplitude between 0 and 1.
        :type alpha: complex
        :param beta: The amplitude of the :math:`|1⟩` state, should be a complex number with amplitude between 0 and 1.
        :type beta: complex
        :raises ValueError: If the given amplitudes do not satisfy the normalization condition.
        """
        if self._is_valid_amplitudes(alpha, beta):
            self.__alpha = alpha
            self.__beta = beta
            self.__qubit_vector = np.array([alpha,beta])
        else:
            raise ValueError(INV_AMP)

    def print_qubit(self) -> None:
        """
        Prints the qubit state in the form:
        Qubit state is :math:`α\exp{i\phi_{0}}|0⟩` + :math:`β\exp{i\phi_{1}}|1⟩`.

        The phase terms are included only if their corresponding relative phases are non-zero.
        """
        r_0, theta_0 = self.__complex_to_euler(self.__alpha)
        r_1, theta_1 = self.__complex_to_euler(self.__beta)
        alpha_term = f"{r_0:.2f}*exp(i{theta_0:0.2f})" if theta_0!= 0.0 else f"{r_0:.2f}"
        beta_term = f"{r_1:.2f}*exp(i{theta_1:0.2f})" if theta_1 != 0.0 else f"{r_1:.2f}"
        if r_0 == 0:
            print(f"Qubit state is {beta_term}|1⟩")
        elif r_1 == 0:
            print(f"Qubit state is {alpha_term}|0⟩")
        else:
            print(f"Qubit state is {alpha_term}|0⟩ + {beta_term}|1⟩")

    def print_vector_form(self) -> None:
        """
        Prints the qubit state in the vector form:
        Qubit state is [alpha,beta].
        """
        print(self.__qubit_vector)

    def measure(self) -> int:
        """
        Measure the qubit and return the collapsed state.

        Simulates a measurement in the computational basis (0 state and 1 state) by calculating the probabilities of each state and using a random number in a uniform distribution on [0,1] interval to determine the outcome.

        Returns
        -------
        int
            The measurement result: 0 or 1.

        Notes
        -----
        The probabilities of each state are determined by the amplitudes squared of each qubit.

        Example
        -------
        >>> qubit = Qubit(0.6, 0.8)
        >>> result = qubit.measure()
        >>> print(result)  # Output will be '0' with probability 0.36, or '1' with probability 0.64
        """

        # Calculate the probability of measuring the state |0⟩
        prob0 = np.abs(self.__alpha)**2  # |α|^2 gives the probability of |0⟩
        
        # Calculate the probability of measuring the state |1⟩
        prob1 = np.abs(self.__beta)**2   # |β|^2 gives the probability of |1⟩

        # Generate a random number between 0 and 1 to simulate the measurement process
        rand_num = random.uniform(0, 1)

        # If the random number is within the probability of measuring |0⟩, return '0'
        if 0 <= rand_num <= prob0:
            return 0
        
        # If the random number is within the probability of measuring |1⟩, return '1'
        if prob0 < rand_num <= 1:
            return 1
