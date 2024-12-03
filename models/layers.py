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
        self.forward_data = None
        self.weights_grads = None
        self.bias_grads = None

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
        self.forward_data = data
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
        self.weights = initialize_parameters(shape=(self.input_shape[0], self.neurons), distribution='normal')
        self.bias = initialize_parameters(shape=(self.neurons, ), distribution='zeros')

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = data
        z = data.dot(self.weights) + self.bias
        output = self.activation.forward(z) if self.activation is not None else z
        if self.output_shape != output.shape:
            raise ValueError(f'La salida de la capa {self.name} no coincide con la esperada. Esperada: {self.output_shape}, obtenida: {output.shape}')
        return output
    
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grads: gradientes de la propagación hacia atrás (entrada, pesos, bias)
        """

        #TODO: Gradientes de la propagación hacia atras de la capa densa
        grad_output = self.activation.backward(grad_output) if self.activation is not None else grad_output
        grad_output = grad_output.reshape((self.neurons, 1))
        w_grads = self.forward_data.reshape((self.input_shape[0], 1)) @ grad_output.T
        b_grads = grad_output.sum(axis=1)
        data_grads = self.weights @ grad_output
        self.weights_grads = w_grads
        self.bias_grads = b_grads
        return data_grads

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
        self.forward_data = data
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
        self.padding = padding

    def compile(self) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        if self.padding == 'same':
            _ = math.floor((self.input_shape[1] - 1) / self.stride) + 1
            self.output_shape = (self.input_shape[0], _, _)
        else:
            _ = math.floor((self.input_shape[1] - self.pool_size) / self.stride) + 1
            self.output_shape = (self.input_shape[0], _, _)

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = data
        _, input_height, input_width = self.input_shape

        output = np.zeros(self.output_shape)
        for h in range(0, input_height - self.pool_size, self.pool_size):
            for w in range(0, input_width - self.pool_size, self.pool_size):
                x = data[:, h:h+self.pool_size, w:w+self.pool_size]
                output[:, h // self.pool_size, w // self.pool_size] = np.amax(x, axis=(1, 2))

        if self.output_shape != output.shape:
            raise ValueError(f'La salida de la capa {self.name} no coincide con la esperada. Esperada: {self.output_shape}, obtenida: {output.shape}')
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :param data: tensor de entrada
        :return grad_output: gradientes de la propagación hacia atrás
        """

        #TODO: Hacer gradiente de la propagación hacia atras
        channels, input_height, input_width = self.input_shape

        data_grads = np.zeros(self.input_shape)
        for h in range(0, input_height - self.pool_size, self.pool_size):
            for w in range(0, input_width - self.pool_size, self.pool_size):
                for c in range(channels):
                    patch = self.forward_data[c, h:h+self.pool_size, w:w+self.pool_size]
                    mask = (patch == np.max(patch))
                    data_grads[c, h:h+self.pool_size, w:w+self.pool_size] = grad_output[c, h//self.pool_size, w//self.pool_size] * mask
        return data_grads

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
        self.weights = initialize_parameters(shape=(self.filters, self.input_shape[0], self.filter_size, self.filter_size), distribution='normal')
        self.bias = initialize_parameters(shape=(self.filters, ), distribution='zeros')

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = data
        channels = self.weights.shape[0]
        _, output_height, output_width = self.output_shape

        z = np.zeros(self.output_shape)
        for h in range(output_height):
            for w in range(output_width):
                for c in range(channels):
                    x = data[:, h:h+self.filter_size, w:w+self.filter_size]
                    weights = self.weights[c, :, :]
                    bias = self.bias[c]
                    z[c, h, w] = np.sum(x * weights + bias)
        output = self.activation.forward(z) if self.activation is not None else z

        if self.output_shape != output.shape:
            raise ValueError(f'La salida de la capa {self.name} no coincide con la esperada. Esperada: {self.output_shape}, obtenida: {output.shape}')
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :param data: tensor de entrada
        :return grad_output: gradientes de la propagación hacia atrás
        """

        channels, _, _, _ = self.weights.shape
        _, input_height, input_width = self.input_shape

        weigth_grads = np.zeros(self.weights.shape)
        bias_grads = np.zeros(self.bias.shape)
        data_grads = np.zeros(self.input_shape)

        #TODO: Hacer gradiente de la propagación hacia atras
        grad_output = self.activation.backward(grad_output) if self.activation is not None else grad_output
        for h in range(input_height - self.filter_size + 1):
            for w in range(input_width - self.filter_size + 1):
                for c in range(channels):
                    data_grads[:, h:h+self.filter_size, w:w+self.filter_size] = grad_output[c, h, w] * self.weights[c, :, :, :]
                    weigth_grads[c, :, :, :] = self.forward_data[:, h:h+self.filter_size, w:w+self.filter_size] * grad_output[c, h, w]
                    bias_grads[c] += grad_output[c, h, w]
        self.weights_grads = weigth_grads
        self.bias_grads = bias_grads
        return data_grads
        
        
