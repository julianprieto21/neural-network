import numpy as np

class Activation:
    def __init__(self, name: str) -> None:
        """
        Constructor de una función de activación

        :param name: nombre de la función de activación
        """
        self.name = name

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación

        :param input: matriz de entrada
        :return: matriz de salida de la función de activación
        """
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación ReLU
        """
        super().__init__('relu')

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación ReLU

        :param input: matriz de entrada
        :return: matriz de salida de la función de activación ReLU
        """
        return np.maximum(input, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación ReLU

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        return grad_output * (input > 0)

class Sigmoid(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Sigmoid
        """
        super().__init__('sigmoid')

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Sigmoid

        :param input: matriz de entrada
        :return: matriz de salida de la función de activación Sigmoid
        """
        return 1 / (1 + np.exp(-input))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Sigmoid

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """        
        #TODO: Revisar
        return grad_output * (1 - input ** 2)
    
class Softmax(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Softmax
        """
        super().__init__('softmax')

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Softmax

        :param input: matriz de entrada
        :return: matriz de salida de la función de activación Softmax
        """
        e_x = np.exp(input - np.max(input, axis=0))
        return e_x / np.sum(e_x, axis=0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Softmax

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        #TODO: Revisar
        return grad_output * (input - np.max(input, axis=1, keepdims=True)) / (np.sum(input, axis=1, keepdims=True))
    
class Tanh(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Tanh
        """
        super().__init__('tanh')

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Tanh

        :param input: matriz de entrada
        :return: matriz de salida de la función de activación Tanh
        """
        return np.tanh(input)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Tanh

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        #TODO: Revisar
        return grad_output * (1 - input ** 2)