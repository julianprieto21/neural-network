import numpy as np

class Activation:
    def __init__(self, name: str) -> None:
        """
        Constructor de una función de activación

        :param name: nombre de la función de activación
        """
        self.name = name
        self.forward_data = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación

        :param data: matriz de entrada
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

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación ReLU

        :param data: matriz de entrada
        :return: matriz de salida de la función de activación ReLU
        """
        self.forward_data = data
        return np.maximum(data, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación ReLU

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        grad_output = self.forward_data * (grad_output.reshape((self.forward_data.shape)) > 0)
        return grad_output

class Sigmoid(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Sigmoid
        """
        super().__init__('sigmoid')

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Sigmoid

        :param data: matriz de entrada
        :return: matriz de salida de la función de activación Sigmoid
        """
        self.forward_data = data
        return 1 / (1 + np.exp(-data))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Sigmoid

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """        
        #TODO: Revisar
        return grad_output * (1 - self.forward_data ** 2)
    
class Softmax(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Softmax
        """
        super().__init__('softmax')

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Softmax

        :param data: matriz de entrada
        :return: matriz de salida de la función de activación Softmax
        """
        self.forward_data = data
        e_x = np.exp(data - np.max(data, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Softmax

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        #TODO: Revisar
        softmax_output = self.forward(self.forward_data)  # data es la entrada original al softmax
        grad_output = softmax_output * (grad_output - np.sum(grad_output * softmax_output, axis=0, keepdims=True))
        return grad_output

class Tanh(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Tanh
        """
        super().__init__('tanh')

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Tanh

        :param data: matriz de entrada
        :return: matriz de salida de la función de activación Tanh
        """
        self.forward_data = data
        return np.tanh(data)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Tanh

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        #TODO: Revisar
        return grad_output * (1 - input ** 2)