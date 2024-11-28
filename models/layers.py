import numpy as np
from .activations import Activation, ReLU
from utils.helpers import initialize_parameters

class Layer:
    def __init__(self, input_shape: tuple[int, ...], name: str='') -> None:
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param activation: función de activación. Por defecto es ReLU
        :param name: nombre de la capa
        :param neurons: cantidad de neuronas en la capa
        :param kernel_size: tamaño del kernel
        :param stride: paso de la convolución
        :param padding: padding de la convolución
        """
        self.input_shape = input_shape
        self.name = name

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

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)
        self.output_shape = (input_shape[0] * input_shape[1] * input_shape[2],) # Multiplicacion de dimensiones. H x W x C

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
    def __init__(self, input_shape: tuple[int, ...], activation: Activation, neurons: int, name: str="dense"):
        """
        Constructor de una capa de red neuronal densa

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)
        self.activation = activation
        self.neurons = neurons
        _ = input_shape[1] if len(input_shape) > 1 else None
        if _ is None:
            self.output_shape = (neurons, )
        else:
            self.output_shape = (neurons, _)
        
        self.weights = initialize_parameters(shape=(self.input_shape[0], self.neurons))
        self.bias = initialize_parameters(shape=(self.neurons, ))

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

        #TODO: Gradientes de la propagación hacia atras de la capa densa
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
    def __init__(self, input_shape: tuple[int, ...], pool_size: int, stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución
        :param padding: padding de la convolución
        """
        super().__init__(input_shape=input_shape, name='max_pooling')
        self.pool_size = pool_size
        self.stride = self.pool_size if stride is None else stride
        self.padding = 0 if padding == 'valid' else 1
        if self.stride:
            _ = ((self.input_shape[1] - 1) // self.stride) + 1
        else:
            _ = ((self.input_shape[1] - self.pool_size) // self.pool_size) + 1
        self.output_shape = (self.input_shape[0], _, _)

class Conv2D(Layer):
    def __init__(self, input_shape: tuple[int, ...], activation: Activation, filters: int, filter_size: int, stride: int=1, padding: str='valid'):
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param activation: función de activación. Por defecto es ReLU
        :param filters: cantidad de filtros/neuronas en la capa
        :param filter_size: tamaño del filtro
        :param stride: paso de la convolución
        :param padding: padding de la convolución
        """
        super().__init__(input_shape=input_shape, name='conv2d')
        self.activation = activation
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = 0 if padding == 'valid' else 1
        _ = (input_shape[1] - self.filter_size + (2 * self.padding)) // self.stride + 1
        self.output_shape = (self.filters, _, _)
        self.weights = initialize_parameters(shape=(self.filter_size, self.filter_size, self.input_shape[0], self.filters))
        self.bias = initialize_parameters(shape=(self.filters, ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        channels, width, height = self.weights.shape
        input_input, input_channels, input_width, height = x.shape

        #TODO: Revisar que los cálculos y dimensiones sean correctos
        z = np.zeros(self.output_shape)
        for x in range(self.output_shape[1]):
            for y in range(self.output_shape[2]):
                for c in range(channels):
                    z[c, x, y] = np.sum(x[c, x:x+self.filter_size, y:y+self.filter_size] * self.weights[c, x, y, :] + self.bias[c])
        y = self.activation.forward(z)
        return y
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        input_width, input_height, channels = self.weights.shape

        #TODO:
        
        
