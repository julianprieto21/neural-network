import numpy as np
from .activations import Activation, ReLU
from utils.helpers import initialize_parameters

class Layer:
    def __init__(self, input_shape: tuple[int, ...], activation: Activation=ReLU(), name: str='', neurons: int=None, kernel_size: int=3, stride: int=1, padding: int=1) -> None:
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (width, height, channels)
        :param activation: función de activación. Por defecto es ReLU
        :param name: nombre de la capa
        :param neurons: cantidad de neuronas en la capa
        :param kernel_size: tamaño del kernel
        :param stride: paso de la convolución
        :param padding: padding de la convolución
        """
        self.input_shape = input_shape
        self.output_shape = None # Calcular
        self.activation = activation
        self.name = name
        self.neurons = neurons
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la capa

        :param input: matriz de entrada
        :return: matriz de salida de la capa
        """
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la capa

        :param grad_output: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        raise NotImplementedError

class Flatten(Layer):
    def __init__(self, input_shape: tuple[int, ...], name: str='flatten') -> None:
        """
        Constructor de una capa de red neuronal flatten

        :param input_shape: tupla con las dimensiones de entrada (width, height, channels)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)
        self.output_shape = (np.prod(input_shape),) # Multiplicacion de dimensiones. H x W x C

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        return x.reshape(self.output_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        return grad_output.reshape(self.input_shape)
    
class Dense(Layer):
    def __init__(self, input_shape: tuple[int, ...], activation: any, neurons: int, name: str="dense"):
        """
        Constructor de una capa de red neuronal densa

        :param input_shape: tupla con las dimensiones de entrada (width, height, channels)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, activation=activation, neurons=neurons, name=name)
        #TODO: Revisar
        _ = input_shape[1] if len(input_shape) > 1 else None
        if _ is None:
            self.output_shape = (neurons, )
        else:
            self.output_shape = (neurons, _)

        self.weights = initialize_parameters(shape=self.output_shape, name=f'W_{self.name}')
        self.bias = initialize_parameters(shape=self.output_shape, name=f'b_{self.name}')

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        z = x.dot(self.weights) + self.bias
        y = self.activation.forward(z)
        return y
    
    def backward(self, grads: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza una propagación hacia atrás

        :param grads: gradientes de la propagación hacia adelante
        :param x: tensor de entrada
        :return grads: gradientes de la propagación hacia atrás (entrada, pesos, bias)
        """

        #TODO: Revisar
        z = self.activation.backward(grads)
        grads = z.dot(self.weights.T)
        grads = grads.reshape(self.input_shape)
        w_grads = grads.dot(x.T)
        b_grads = grads.sum(axis=0)
        return grads, w_grads, b_grads

class Dropout(Layer):
    def __init__():
        pass

class MaxPooling(Layer):
    def __init__():
        pass

class Conv2D(Layer):
    def __init__():
        pass