import numpy as np

class Activation:
    def __init__(self, name: str) -> None:
        """
        Constructor de una función de activación.

        :param name: nombre de la función de activación
        """
        self.name = name
        self.forward_data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación.

        :param x: matriz de entrada
        :return: matriz de salida de la función de activación
        """
        raise NotImplementedError
    
    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación.

        :param x_grad: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación ReLU.
        """

        super().__init__('relu')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación ReLU.

        :param x: matriz de entrada
        :return: matriz de salida de la función de activación ReLU
        """

        self.forward_data = x
        return np.maximum(x, np.array(0.0, x.dtype))

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación ReLU.

        :param x_grad: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """

        return x_grad * (self.forward_data > 0).astype(float)
    
class Softmax(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Softmax.
        """
        super().__init__('softmax')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Softmax.

        :param x: matriz de entrada
        :return: matriz de salida de la función de activación Softmax
        """
        self.forward_data = x
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Softmax.

        :param x_grad: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        return x_grad

class Sigmoid(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Sigmoid.
        """
        super().__init__('sigmoid')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Sigmoid.

        :param x: matriz de entrada
        :return: matriz de salida de la función de activación Sigmoid
        """

        self.forward_data = 1 / (1 + np.exp(-x))
        return self.forward_data

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Sigmoid.

        :param x_grad: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """

        return x_grad * (self.forward_data * (1 - self.forward_data))
    
class Tanh(Activation):
    def __init__(self) -> None:
        """
        Constructor de una función de activación Tanh.
        """
        super().__init__('tanh')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la función de activación Tanh.

        :param x: matriz de entrada
        :return: matriz de salida de la función de activación Tanh
        """

        self.forward_data = np.tanh(x)
        return self.forward_data

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la función de activación Tanh.

        :param x_grad: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """

        return x_grad * (1 - self.forward_data ** 2)