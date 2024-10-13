import numpy as np

INV_AMP = "Amplitude is invalid. The given amplitude needs to be between 0 and 1 and satisfy the normalization condition."

class Qubit:
    """
    A class to represent a quantum bit (Qubit) with two possible states: |0⟩ and |1⟩,
    described by complex amplitudes alpha and beta.

    :ivar alpha: The amplitude of the |0⟩ state, a value between 0 and 1.
    :vartype alpha: float
    :ivar beta: The amplitude of the |1⟩ state, a value between 0 and 1.
    :vartype beta: float
    :ivar rel_ph_zero: The relative phase of the |0⟩ state is exp(rel_ph_zero*π*i). Default value is 0.0.
    :vartype rel_ph_zero: float
    :ivar rel_ph_one: The relative phase of the |1⟩ state is exp(rel_ph_one*π*i). Default value is 0.0.
    :vartype rel_ph_one: float
    """

    def __init__(self, alpha, beta, rel_ph_zero=0.0, rel_ph_one=0.0):
        """
        Constructs all the necessary attributes for the qubit object.

        :param alpha: The amplitude of the |0⟩ state, should be a value between 0 and 1.
        :type alpha: float
        :param beta: The amplitude of the |1⟩ state, should be a value between 0 and 1.
        :type beta: float
        :param rel_ph_zero: The relative phase of the |0⟩ state, defaults to 0.0
        :type rel_ph_zero: float, optional
        :param rel_ph_one: The relative phase of the |1⟩ state, defaults to 0.0
        :type rel_ph_one: float, optional
        :raises ValueError: If the given amplitudes do not satisfy the normalization condition.
        """
        if self._is_valid_amplitudes(alpha, beta):
            self.__alpha = alpha
            self.__beta = beta
            self.__rel_ph_zero = rel_ph_zero
            self.__rel_ph_one = rel_ph_one
        else:
            raise ValueError(INV_AMP)

    def _is_valid_amplitudes(self, alpha, beta):
        """
        Checks if the given amplitudes are valid and satisfy the normalization condition.

        :param alpha: The amplitude of the |0⟩ state.
        :type alpha: float
        :param beta: The amplitude of the |1⟩ state.
        :type beta: float
        :return: True if the amplitudes are valid and their squares sum to 1, False otherwise.
        :rtype: bool
        """
        return (0 <= alpha <= 1 and 
                0 <= beta <= 1 and 
                np.isclose((alpha)**2 + (beta)**2, 1, rtol=1e-9, atol=1e-9))

    def get_alpha(self):
        """
        Returns the amplitude of the |0⟩ state.

        :return: The amplitude of the |0⟩ state (alpha).
        :rtype: float
        """
        return self.__alpha

    def get_beta(self):
        """
        Returns the amplitude of the |1⟩ state.

        :return: The amplitude of the |1⟩ state (beta).
        :rtype: float
        """
        return self.__beta
    
    def get_rel_ph_zero(self):
        """
        Returns the relative phase for the |0⟩ state.

        :return: The relative phase for the |0⟩ state (rel_ph_zero).
        :rtype: float
        """
        return self.__rel_ph_zero
    
    def get_rel_ph_one(self):
        """
        Returns the relative phase for the |1⟩ state.

        :return: The relative phase for the |1⟩ state (rel_ph_one).
        :rtype: float
        """
        return self.__rel_ph_one

    def set_alpha(self, alpha):
        """
        Sets the amplitude of the |0⟩ state.

        :param alpha: The amplitude of the |0⟩ state, must be between 0 and 1.
        :type alpha: float
        :raises ValueError: If the amplitude is not valid.
        """
        if self._is_valid_amplitudes(alpha, self.__beta):
            self.__alpha = alpha
        else:
            raise ValueError(INV_AMP)

    def set_beta(self, beta):
        """
        Sets the amplitude of the |1⟩ state.

        :param beta: The amplitude of the |1⟩ state, must be between 0 and 1.
        :type beta: float
        :raises ValueError: If the amplitude is not valid.
        """
        if self._is_valid_amplitudes(self.__alpha, beta):
            self.__beta = beta
        else:
            raise ValueError(INV_AMP)

    def set_rel_ph_zero(self, new_phase):
        """
        Sets the relative phase for the |0⟩ state.

        :param new_phase: The new phase value for the |0⟩ state.
        :type new_phase: float
        """
        self.__rel_ph_zero = new_phase

    def set_rel_ph_one(self, new_phase):
        """
        Sets the relative phase for the |1⟩ state.

        :param new_phase: The new phase value for the |1⟩ state.
        :type new_phase: float
        """
        self.__rel_ph_one = new_phase

    def print_qubit(self):
        """
        Prints the qubit state in the form:
        Qubit state is [phase term]α|0⟩ + [phase term]β|1⟩.

        The phase terms are included only if their corresponding relative phases are non-zero.
        """
        alpha_term = f"exp(i{self.__rel_ph_zero}π){self.__alpha:.4f}" if self.__rel_ph_zero != 0.0 else f"{self.__alpha:.4f}"
        beta_term = f"exp(i{self.__rel_ph_one}π){self.__beta:.4f}" if self.__rel_ph_one != 0.0 else f"{self.__beta:.4f}"
        print(f"Qubit state is {alpha_term}|0⟩ + {beta_term}|1⟩")