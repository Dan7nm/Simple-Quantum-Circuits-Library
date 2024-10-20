import numpy as np

INV_GATE_TYP = "Invalid gate type."

class SingleQubitGate:
    def __set_I(self):
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = 1

    def __set_X(self):
        self.__gate_matrix[0][1] = 1
        self.__gate_matrix[1][0] = 1

    def __set_Y(self):
        self.__gate_matrix[0][1] = -1j
        self.__gate_matrix[1][0] = 1j

    def __set_Z(self):
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = -1

    def __set_H(self):
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[0][1] = 1
        self.__gate_matrix[1][0] = 1
        self.__gate_matrix[1][1] = -1
        self.__gate_matrix *= (1/np.sqrt(2))

    def __set_R(self,phi):
        self.__gate_matrix[0][0] = 1
        self.__gate_matrix[1][1] = np.exp(1j*phi)


    def __init__(self,gate_type='I',phi=0.0):
        # Initalize a two dimensional matrix to represent each gate
        self.__gate_matrix = np.zeros((2,2), dtype=complex)
        # Dictionaty which corresponds for every letter to a matching matrix
        matrix_type = {
            "I": self.__set_I,
            "X": self.__set_X,
            "Y": self.__set_Y,
            "Z": self.__set_Z,
            "H": self.__set_H,
            "R": self.__set_R
        }
        if gate_type in matrix_type:
            if(gate_type == "R"):
                matrix_type[gate_type](phi)
            else:
                matrix_type[gate_type]()
        else:
            raise ValueError(INV_GATE_TYP)

    def get_matrix(self):
        """Returns the gate matrix."""
        return self.__gate_matrix

    def print_matrix(self):
        """Prints the gate matrix with proper formatting and without apostrophes."""
        for row in self.__gate_matrix:
            formatted_row = ['[']
            for elem in row:
                # Print the element directly and format as needed
                if elem.imag == 0:
                    # If it's purely real, display just the real part
                    formatted_row.append(f"{elem.real:.0f}")
                elif elem.real == 0:
                    # If the real part is 0 and it's complex, show the imaginary part
                    formatted_row.append(f"{elem.imag}j")
                else:
                    # If it has both real and imaginary parts, display them
                    formatted_row.append(f"{elem.real}{elem.imag:+}j")
            formatted_row.append(']')
            
            # Join the formatted row and print it without the apostrophes
            print(" ".join(formatted_row))

# class TwoQubitGate:
