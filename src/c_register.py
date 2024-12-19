import numpy as np

INDEX_ERROR = "Index out of range. The index should be between 0 and number of classical bit - 1 in the classical register."

class ClassicalRegister:
    """
    A class representing a classical register for quantum computation.
    
    """
    
    def __init__(self, num_bits: int = 1) -> None:
        """
        Initializes the classical register with a given number of bits.
        
        Parameters
        ----------
        num_bits : int, optional
            The number of bits in the register (default is 1).
        """
        self.__num_bits = num_bits
        # Initializes the classical register with zeros.
        self.__c_reg = np.zeros(num_bits)

    def change_number_of_bits(self, num_bits: int) -> None:
        """
        Changes the number of bits in the classical register and resets its values to zero.
        
        Parameters
        ----------
        num_bits : int
            The new number of bits for the classical register.
        """
        self.__num_bits = num_bits
        self.__c_reg = np.zeros(num_bits)

    def print_c_reg(self) -> None:
        """
        Prints the current state of the classical register.
        
        This method displays the register's array of bits to the console.
        """
        print(self.__c_reg)

    def __getitem__(self, index: int) -> int:
        """
        Overloads the indexing operator to access specific elements of the register.
        
        Parameters
        ----------
        index : int
            The index of the element to retrieve from the register.
        
        Returns
        -------
        int
            The value at the specified index of the register.
        
        Raises
        ------
        IndexError
            If the index is out of range for the register.
        """
        if 0 <= index < self.__num_bits:
            return self.__c_reg[index]
        else:
            raise IndexError(INDEX_ERROR)

    def __setitem__(self, index: int, value: int) -> None:
        """
        Overloads the indexing operator to set specific elements of the register.
        
        Parameters
        ----------
        index : int
            The index of the element to set in the register.
        
        value : int
            The value to assign to the register element (must be either 0 or 1).
        
        Raises
        ------
        IndexError
            If the index is out of range for the register.
        
        ValueError
            If the value is not 0 or 1.
        """
        if 0 <= index < self.__num_bits:
            if value not in (0, 1):  # Ensure value is either 0 or 1
                raise ValueError("Classical register can only hold values 0 or 1.")
            self.__c_reg[index] = value
        else:
            raise IndexError(INDEX_ERROR)
        
    def get_bits_num(self)->int:
        """
        Returns number of classical bit in the classical register.

        Returns
        -------
        int
            The number of bits in the classical register
        """
        return self.__num_bits
