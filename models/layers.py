import numpy as np
from .activations import Activation, ReLU
from utils.helpers import initialize_parameters
import math

class Layer:
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='') -> None:
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
        self.output_shape = None
        self.name = name
        self.weights = None
        self.bias = None

    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        raise NotImplementedError

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la capa

        :param data: matriz de entrada
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
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='flatten') -> None:
        """
        Constructor de una capa de red neuronal flatten

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)

    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2],) # Multiplicacion de dimensiones. H x W x C

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param data: tensor de entrada
        :return y: tensor de salida
        """
        return data.reshape(self.output_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        return grad_output.reshape(self.input_shape)
    
class Dense(Layer):
    def __init__(self, neurons: int, input_shape: tuple[int, ...]=None, activation: Activation=None, name: str="dense"):
        """
        Constructor de una capa de red neuronal densa

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)
        self.activation = activation
        self.neurons = neurons
    
    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        _ = self.input_shape[1] if len(self.input_shape) > 1 else None
        if _ is None:
            self.output_shape = (self.neurons, )
        else:
            self.output_shape = (self.neurons, _)  
        self.weights = initialize_parameters(shape=(self.input_shape[0], self.neurons))
        self.bias = initialize_parameters(shape=(self.neurons, ))

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        z = data.dot(self.weights) + self.bias
        y = self.activation.forward(z) if self.activation is not None else z
        return y
    
    def backward(self, grads: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza una propagación hacia atrás

        :param grads: gradientes de la propagación hacia adelante
        :param x: tensor de entrada
        :return grads: gradientes de la propagación hacia atrás (entrada, pesos, bias)
        """

        #TODO: Gradientes de la propagación hacia atras de la capa densa
        z = self.activation.backward(z) if self.activation is not None else z
        grads = z.dot(self.weights.T)
        grads = grads.reshape(self.input_shape)
        w_grads = grads.dot(data.T)
        b_grads = grads.sum(axis=0)
        return grads, w_grads, b_grads

class Dropout(Layer):
    def __init__(self, rate: float=0.5):
        """
        Constructor de una capa de dropout

        :param rate: tasa de dropout
        """
        self.rate = rate

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param data: tensor de entrada
        :return y: tensor de salida
        """
        mask = np.random.rand(*data.shape) < self.rate
        y = data * mask
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        #TODO: Hacer gradiente de la propagación hacia atras
        return grad_output

class MaxPool2D(Layer):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name='max_pooling')
        self.pool_size = pool_size
        self.stride = self.pool_size if stride is None else stride
        self.padding = 0 if padding == 'valid' else 1
        
    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        if self.stride:
            _ = ((self.input_shape[1] - 1) // self.stride) + 1
        else:
            _ = ((self.input_shape[1] - self.pool_size) // self.pool_size) + 1
        self.output_shape = (self.input_shape[0], _, _)

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        grad = np.zeros(self.output_shape)
        for x in range(0, self.input_shape[1] - 1, self.pool_size):
            for y in range(0, self.input_shape[2] - 1, self.pool_size):
                grad[:, x // self.pool_size, y // self.pool_size] = np.amax(data[:, x:x+self.pool_size, y:y+self.pool_size], axis=(1, 2))
        return grad

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        #TODO: Hacer gradiente de la propagación hacia atras
        return grad_output

class Conv2D(Layer):
    def __init__(self, activation: Activation, filters: int, filter_size: int, input_shape: tuple[int, ...]=None, stride: int=1, padding: str='valid'):
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param activation: función de activación. Por defecto es ReLU
        :param filters: cantidad de filtros/neuronas en la capa
        :param filter_size: tamaño del filtro
        :param stride: paso de la convolución. Por defecto 1
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name='conv2d')
        self.activation = activation
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = 0 if padding == 'valid' else math.ceil((self.filter_size - 1) / 2)

    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        _ = (self.input_shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1
        self.output_shape = (self.filters, _, _)
        self.weights = initialize_parameters(shape=(self.filters, self.input_shape[0], self.filter_size, self.filter_size))
        self.bias = initialize_parameters(shape=(self.filters, ))

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        channels, _, width, height = self.weights.shape
        input_channels, input_width, height = data.shape

        #TODO: Revisar que los cálculos y dimensiones sean correctos
        z = np.zeros(self.output_shape)
        for x in range(self.output_shape[1]):
            for y in range(self.output_shape[2]):
                for c in range(channels):
                    z[c, x, y] = np.sum(data[:, x:x+self.filter_size, y:y+self.filter_size] * self.weights[c, :, :] + self.bias[c])
        y = self.activation.forward(z) if self.activation is not None else z
        return y
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """
        channels, _, _ = self.weights.shape
        _, input_height, input_width = self.input_shape

        weigth_grads = np.zeros(self.weights.shape)
        bias_grads = np.zeros(self.bias.shape)
        data_grads = np.zeros(self.input_shape)

        #TODO: Hacer gradiente de la propagación hacia atras
        # Definir de donde saco el data.
        # el bias_grads es += o = ?
        for x in range(input_width):
            for y in range(input_height):
                for c in range(channels):
                    data_grads[c, x, y] = grad_output[c, x, y] * self.weights[c, :, x, y]
                    weigth_grads[c, :, x, y] = grad_output[c, x, y] * data[:, x, y]
                    bias_grads[c] = grad_output[c, x, y]
        return grad_output
        
        
