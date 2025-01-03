import numpy as np
from .activations import Activation
from utils.helpers import initialize_parameters
import math

class Layer:
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='', weights: np.array=None, bias: np.array=None) -> None:
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
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
        self.weights = weights
        self.bias = bias
        self.forward_data = None
        self.weights_grads = None
        self.bias_grads = None
        if input_shape is not None: self.compile(input_shape)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la capa

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        raise NotImplementedError

    def backward(self, grad_x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación de la derivada de la capa

        :param grad_x: matriz de derivada de salida
        :return: matriz de derivada de entrada
        """
        raise NotImplementedError

class Flatten(Layer):
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='flatten') -> None:
        """
        Constructor de una capa de red neuronal flatten

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.output_shape = (None, int(np.prod(input_shape[1:])))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        self.forward_data = x
        batch_size = x.shape[0]
        return x.transpose(0, 2, 3, 1).reshape(batch_size, self.output_shape[1]) # TODO: Realizar de otra manera

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, channels, height, width = self.forward_data.shape
        return x_grad.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2) # TODO: Realizar de otra manera
    
class Dense(Layer):
    def __init__(self, neurons: int, input_shape: tuple[int, ...]=None, activation: Activation=None, name: str="dense", weights: np.array=None, bias: np.array=None) -> None:
        """
        Constructor de una capa de red neuronal densa

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        :param name: nombre de la capa
        """

        self.activation = activation
        self.neurons = neurons
        super().__init__(input_shape=input_shape, name=name, weights=weights, bias=bias)
    
    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.output_shape = (None, self.neurons)
        if self.weights is None:
            self.weights = initialize_parameters(shape=(input_shape[1], self.neurons), distribution='normal')
        if self.bias is None:
            self.bias = initialize_parameters(shape=(self.neurons), distribution='zeros')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = x # Guardar datos de entrada para la propagación hacia atrás
        z = x.dot(self.weights) + self.bias # Realizar la multiplicación de los pesos y sumar el sesgo
        output = self.activation(z) if self.activation else z # Aplicar la función de activación si es que existe
        return output
    
    def backward(self, x_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás (entrada, pesos, bias)
        """
        x_grad = self.activation.backward(x_grad) if self.activation else x_grad
        weight_grads = np.dot(self.forward_data.T, x_grad) # Gradientes con respecto a los pesos
        bias_grads = x_grad.sum(axis=0) # Gradientes con respecto al sesgo
        data_grads = np.dot(x_grad, self.weights.T) # Gradientes con respecto a la entrada

        self.weights_grads = weight_grads
        self.bias_grads = bias_grads
        return data_grads

class Conv2D(Layer):
    def __init__(self, filters: int, filter_size: int, activation: Activation=None, name: str='conv', input_shape: tuple[int, ...]=None, stride: int=1, padding: str='valid', weights: np.array=None, bias: np.array=None):
        """
        Constructor de una capa de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param activation: función de activación.
        :param filters: cantidad de filtros/neuronas en la capa.
        :param filter_size: tamaño del filtro
        :param stride: paso de la convolución. Por defecto 1
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        self.activation = activation
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        super().__init__(input_shape=input_shape, name=name, weights=weights, bias=bias)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        if self.padding == 'valid': 
            self.padding = 0

            out_height = (input_shape[2] - self.filter_size) // self.stride + 1
            out_width = (input_shape[3] - self.filter_size) // self.stride + 1
        elif self.padding == 'same':
            output_size = math.ceil(input_shape[2] / self.stride)
            padd_total = max(0, (output_size - 1) * self.stride + self.filter_size - input_shape[2])
            self.padding = math.ceil(padd_total // 2)

            out_height = (input_shape[2] + 2 * self.padding - self.filter_size) // self.stride + 1
            out_width = (input_shape[3] + 2 * self.padding - self.filter_size) // self.stride + 1
        else:
            raise ValueError(f'Padding {self.padding} no soportado. Utilice "valid" o "same"')

        self.output_shape = (None, self.filters, out_height, out_width) 

        if self.weights is None:
            self.weights = initialize_parameters(shape=(self.filter_size, self.filter_size, input_shape[1], self.filters), distribution='normal')
        if self.bias is None:
            self.bias = initialize_parameters(shape=(self.filters), distribution='zeros')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """

        self.forward_data = x # (batch_size, channels, height, width)
        batch_size = x.shape[0] # (batch_size, channels, height, width)
        _, _, output_height, output_width = self.output_shape # (batch_size, filters, output_height, output_width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)) , mode='constant') # Añadir padding a la entrada
        
        z = np.zeros((batch_size, self.filters, output_height, output_width)) # (batch_size, channels, height, width)
        for h in range(output_height):
            for w in range(output_width):
                h_start, w_start = h * self.stride, w * self.stride # Ajustar las posiciones iniciales segun el stride. Por defecto stride = 1.
                h_end, w_end = h_start + self.filter_size, w_start + self.filter_size # Ajusta las posiciones finales sumandoles el tamaño del filtro.
                x_region = x[:, :, h_start:h_end, w_start:w_end] # Extraer la región de la entrada
                z[:, :, h, w] = np.tensordot(x_region, self.weights, axes=([1, 2, 3], [2, 0, 1])) + self.bias # Realizar la convolución

        output = self.activation(z) if self.activation else z # Aplicar la función de activación si es que existe
        return output # (batch_size, filters, output_height, output_width)
    
    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, _, output_height, output_width = x_grad.shape # (batch_size, filters, output_height, output_width)
        _, _, input_height, input_width = self.forward_data.shape
        x_grad = self.activation.backward(x_grad) if self.activation else x_grad
        x = self.forward_data
        
        if self.padding: 
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant') # Añadir padding a la entrada
        
        data_grads = np.zeros_like(x) # Gradientes con respecto a la entrada (batch_size, channels, height, width)
        weigth_grads = np.zeros_like(self.weights)  # Gradientes con respecto a los pesos (filter_size, filter_size, channels, filters)
        bias_grads = np.sum(x_grad, axis=(0, 2, 3))  # Gradiente del sesgo. Se calcula directamente dada su simplicidad. (filters)

        for h in range(output_height):
            for w in range(output_width):
                h_start, w_start = h * self.stride, w * self.stride # Ajustar las posiciones iniciales segun el stride. Por defecto stride = 1.
                h_end, w_end = h_start + self.filter_size, w_start + self.filter_size # Ajusta las posiciones finales sumandoles el tamaño del filtro.
                
                x_region = x[:, :, h_start:h_end, w_start:w_end] # Extraer la región de la entrada
                weigth_grads += np.tensordot(x_region, x_grad[:, :, h, w], axes=([0], [0])).transpose(1, 2, 0, 3) # Gradientes con respecto a los pesos. Hace falta transponer?

                for b in range(batch_size): # Calcular los gradientes con respecto a la entrada
                    data_grads[b, :, h_start:h_end, w_start:w_end] += np.tensordot(
                        x_grad[b, :, h, w], self.weights, axes=([0], [3])
                    ).transpose(2, 0, 1) # Hace falta transponer?
        
        if self.padding:
            data_grads = data_grads[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding] # Eliminar el padding de los gradientes con respecto a la entrada
        
        self.weights_grads = weigth_grads
        self.bias_grads = bias_grads
        return data_grads

class Pool2D(Layer):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        self.pool_size = pool_size
        self.stride = self.pool_size if stride is None else stride
        self.padding = padding
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa. Inicializando sus parámetros y generando las dimensiones de salida de la capa
        """
        if self.padding == 'valid': 
            self.padding = 0

            out_height = (input_shape[2] - self.pool_size) // self.stride + 1
            out_width = (input_shape[3] - self.pool_size) // self.stride + 1
        elif self.padding == 'same':
            output_size = math.ceil(input_shape[2] / self.stride)
            padd_total = max(0, (output_size - 1) * self.stride + self.pool_size - input_shape[2])
            self.padding = math.ceil(padd_total // 2)

            out_height = (input_shape[2] + 2 * self.padding - self.pool_size) // self.stride + 1
            out_width = (input_shape[3] + 2 * self.padding - self.pool_size) // self.stride + 1
        else:
            raise ValueError(f'Padding {self.padding} no soportado. Utilice "valid" o "same"')

        self.output_shape = (None, input_shape[1], out_height, out_width)

    def __call__():
        raise NotImplementedError
    
    def backward():
        raise NotImplementedError

class MaxPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # TODO: Entender implementación y funcionamiento de función.
        strided_x = np.lib.stride_tricks.as_strided(
            x,
            shape=(
                batch_size,
                channels,
                output_height,
                output_width,
                self.pool_size,
                self.pool_size,
            ),
            strides=(
                x.strides[0],  # batch stride
                x.strides[1],  # channel stride
                x.strides[2] * self.stride,  # spatial height stride
                x.strides[3] * self.stride,  # spatial width stride
                x.strides[2],  # pool height stride
                x.strides[3],  # pool width stride
            ),
            writeable=False,
        )

        output = np.amax(strided_x, axis=(-2, -1)) # Aplicar max pooling
        output = np.reshape(output, (batch_size, channels, output_height, output_width)) # Asegurarse que tener formato estándar (batches, channels, height, width)

        return output

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        data_grads = np.zeros(x.shape) # (batches, channels, height, width)
        pool_h, pool_w = self.pool_size, self.pool_size # Tamaño del pooling
        stride_h, stride_w = self.stride, self.stride # Paso del pooling
        _, _, output_height, output_width = x_grad.shape # (batches, channels, output_height, output_width)

        for b in range(x.shape[0]):
            for h in range(output_height):
                for w in range(output_width):
                    h_start, w_start = h * stride_h, w * stride_w # Posiciones iniciales
                    h_end, w_end = h_start + pool_h, w_start + pool_w # Posiciones finales

                    patch = x[b, :, h_start:h_end, w_start:w_end] # Extraer la región de la entrada
                    max_val = np.max(patch, axis=(1, 2), keepdims=True) # Máscara de los valores máximos
                    mask = (patch == max_val) # Máscara de los valores máximos

                    for c in range(mask.shape[0]):  # Iterar por canal
                        flat_mask = mask[c].reshape(-1)  # Aplanar para simplificar
                        true_indices = np.flatnonzero(flat_mask)  # Índices donde mask es True
                        if len(true_indices) > 1:
                            flat_mask[true_indices[1:]] = False  # Dejar solo el primer True
                        mask[c] = flat_mask.reshape(mask[c].shape)  # Restaurar forma original

                    grad = x_grad[b, :, h, w][:, np.newaxis, np.newaxis] # Gradiente de la salida
                    data_grads[b, :, h_start:h_end, w_start:w_end] += grad * mask # Gradientes con respecto a la entrada
        return data_grads

class MinPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # TODO: Entender implementación y funcionamiento de función.
        strided_x = np.lib.stride_tricks.as_strided(
            x,
            shape=(
                batch_size,
                channels,
                output_height,
                output_width,
                self.pool_size,
                self.pool_size,
            ),
            strides=(
                x.strides[0],  # batch stride
                x.strides[1],  # channel stride
                x.strides[2] * self.stride,  # spatial height stride
                x.strides[3] * self.stride,  # spatial width stride
                x.strides[2],  # pool height stride
                x.strides[3],  # pool width stride
            ),
            writeable=False,
        )

        output = np.amin(strided_x, axis=(-2, -1)) # Aplicar min pooling
        output = np.reshape(output, (batch_size, channels, output_height, output_width)) # Asegurarse que tener formato estándar (batches, channels, height, width)

        return output

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        data_grads = np.zeros(x.shape) # (batches, channels, height, width)
        pool_h, pool_w = self.pool_size, self.pool_size # Tamaño del pooling
        stride_h, stride_w = self.stride, self.stride # Paso del pooling
        _, _, output_height, output_width = x_grad.shape # (batches, channels, output_height, output_width)

        for b in range(x.shape[0]):
            for h in range(output_height):
                for w in range(output_width):
                    h_start, w_start = h * stride_h, w * stride_w # Posiciones iniciales
                    h_end, w_end = h_start + pool_h, w_start + pool_w # Posiciones finales

                    patch = x[b, :, h_start:h_end, w_start:w_end] # Extraer la región de la entrada
                    min_val = np.min(patch, axis=(1, 2), keepdims=True) # Máscara de los valores máximos
                    mask = (patch == min_val) # Máscara de los valores máximos

                    for c in range(mask.shape[0]):  # Iterar por canal
                        flat_mask = mask[c].reshape(-1)  # Aplanar para simplificar
                        true_indices = np.flatnonzero(flat_mask)  # Índices donde mask es True
                        if len(true_indices) > 1:
                            flat_mask[true_indices[1:]] = False  # Dejar solo el primer True
                        mask[c] = flat_mask.reshape(mask[c].shape)  # Restaurar forma original

                    grad = x_grad[b, :, h, w][:, np.newaxis, np.newaxis] # Gradiente de la salida
                    data_grads[b, :, h_start:h_end, w_start:w_end] += grad * mask # Gradientes con respecto a la entrada
        return data_grads

class AvgPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa de red neuronal de max pooling
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return output: tensor de salida
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # TODO: Entender implementación y funcionamiento de función.
        strided_x = np.lib.stride_tricks.as_strided(
            x,
            shape=(
                batch_size,
                channels,
                output_height,
                output_width,
                self.pool_size,
                self.pool_size,
            ),
            strides=(
                x.strides[0],  # batch stride
                x.strides[1],  # channel stride
                x.strides[2] * self.stride,  # spatial height stride
                x.strides[3] * self.stride,  # spatial width stride
                x.strides[2],  # pool height stride
                x.strides[3],  # pool width stride
            ),
            writeable=False,
        )

        output = np.mean(strided_x, axis=(-2, -1)) # Aplicar avg pooling
        output = np.reshape(output, (batch_size, channels, output_height, output_width)) # Asegurarse que tener formato estándar (batches, channels, height, width)

        return output

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        data_grads = np.zeros(x.shape) # (batches, channels, height, width)
        pool_h, pool_w = self.pool_size, self.pool_size # Tamaño del pooling
        stride_h, stride_w = self.stride, self.stride # Paso del pooling
        _, _, output_height, output_width = x_grad.shape # (batches, channels, output_height, output_width)

        for b in range(x.shape[0]):
            for h in range(output_height):
                for w in range(output_width):
                    h_start, w_start = h * stride_h, w * stride_w # Posiciones iniciales
                    h_end, w_end = h_start + pool_h, w_start + pool_w # Posiciones finales

                    grad = x_grad[b, :, h, w][:, np.newaxis, np.newaxis] # Gradiente de la salida
                    avg_grad = grad / (pool_h * pool_w) # Gradiente promedio
                    data_grads[b, :, h_start:h_end, w_start:w_end] += avg_grad # Gradientes con respecto a la entrada
        return data_grads