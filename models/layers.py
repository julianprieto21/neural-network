from __future__ import annotations
import inspect
import numpy as np
from .activations import Activation, Sigmoid, Tanh
from utils.helpers import initialize_parameters
import math

class Layer:
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='layer') -> None:
        """
        Constructor de una capa de red neuronal.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        :param name: nombre de la capa
        """
        self.name = name
        self.output_shape = None
        self.forward_data = None
        self.data_grads = None
        if input_shape is not None: self.compile(input_shape)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        raise NotImplementedError

    def backward(self, grad_x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        raise NotImplementedError

    def copy(self) -> Layer:
        """
        Devuelve una copia de la capa.

        :return: copia de la capa
        """
        signature = inspect.signature(type(self).__init__) # Obtener la firma del constructor (__init__)
        init_params = signature.parameters

        init_args = { # Extraer los valores de los parámetros obligatorios de la instancia actual
            name: getattr(self, name)
            for name, param in init_params.items()
            if name != "self" and hasattr(self, name)
        }

        new_layer = type(self)(**init_args) # Crear una nueva instancia usando los argumentos extraídos
        for attr, value in self.__dict__.items(): # Copiar los demás atributos
            setattr(new_layer, attr, value)
        return new_layer

    def get_params(self) -> list[np.ndarray]:
        """ 
        Obtiene los parámetros de la capa.
        
        :return: lista de parámetros
        """
        params = []
        if hasattr(self, 'weights'):
            params.append(self.weights)
        if hasattr(self, 'recurrent_weights'):
            params.append(self.recurrent_weights)
        if hasattr(self, 'bias'):
            params.append(self.bias)
        return params
    
    def get_grads(self) -> list[np.ndarray]:
        """ 
        Obtiene los gradientes de la capa.
        
        :return: lista de gradientes
        """
        grads = []
        if hasattr(self, 'weights_grads'):
            grads.append(self.weights_grads)
        if hasattr(self, 'recurrent_weights_grads'):
            grads.append(self.recurrent_weights_grads)
        if hasattr(self, 'bias_grads'):
            grads.append(self.bias_grads)
        return grads

class Flatten(Layer):
    def __init__(self, input_shape: tuple[int, ...]=None, name: str='flatten') -> None:
        """
        Constructor de una capa flatten.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        :param name: nombre de la capa
        """
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        self.output_shape = (None, int(np.prod(input_shape[1:])))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        # TODO: REFACTORIZAR FUNCIÓN
        self.forward_data = x
        batch_size = x.shape[0]
        return x.transpose(0, 2, 3, 1).reshape(batch_size, self.output_shape[1])

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        # TODO: REFACTORIZAR FUNCIÓN
        batch_size, channels, height, width = self.forward_data.shape
        self.data_grads = x_grad.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        return self.data_grads
    
class Dense(Layer):
    def __init__(self, neurons: int, input_shape: tuple[int, ...]=None, name: str="dense", activation: Activation=None, weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', bias_initializer: str='zeros') -> None:
        """
        Constructor de una capa densa.

        :param neurons: número de neuronas
        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        :param name: nombre de la capa
        :param activation: función de activación
        :param weights: pesos
        :param bias: bias
        :param weight_initializer: inicializador de pesos. Por defecto 'glorot_uniform'
        :param bias_initializer: inicializador de bias. Por defecto 'zeros'
        """
        self.activation = activation
        self.neurons = neurons
        self.weights = weights
        self.bias = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weights_grads = None
        self.bias_grads = None
        super().__init__(input_shape=input_shape, name=name)
    
    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        self.output_shape = (None, self.neurons)
        if self.weights is None:
            self.weights = initialize_parameters(shape=(input_shape[1], self.neurons), distribution=self.weight_initializer)
        if self.bias is None:
            self.bias = initialize_parameters(shape=(self.neurons), distribution=self.bias_initializer, is_bias=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
         Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        z = x.dot(self.weights) + self.bias
        output = self.activation(z) if self.activation else z
        return output
    
    def backward(self, x_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x_grad = self.activation.backward(x_grad) if self.activation else x_grad
        self.weights_grads = np.dot(self.forward_data.T, x_grad)
        self.bias_grads = x_grad.sum(axis=0)
        self.data_grads = np.dot(x_grad, self.weights.T)

        return self.data_grads

class Conv2D(Layer):
    def __init__(self, filters: int, filter_size: int, input_shape: tuple[int, ...]=None, name: str='conv', activation: Activation=None, stride: int=1, padding: str='valid', weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', bias_initializer: str='zeros'):
        """
        Constructor de una capa convolucional 2D.

        :param filters: cantidad de filtros/neuronas en la capa.
        :param filter_size: tamaño del filtro
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param name: nombre de la capa
        :param activation: función de activación.
        :param stride: paso de la convolución. Por defecto 1
        :param padding: padding de la convolución. Por defecto 'valid'
        :param weights: pesos
        :param bias: bias
        :param weight_initializer: inicializador de pesos. Por defecto 'glorot_uniform'
        :param bias_initializer: inicializador de bias. Por defecto 'zeros'
        """
        self.activation = activation
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.weights = weights
        self.bias = bias
        self.weights_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weights_grads = None
        self.bias_grads = None
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
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
            self.weights = initialize_parameters(shape=(self.filter_size, self.filter_size, input_shape[1], self.filters), distribution=self.weights_initializer)
        if self.bias is None:
            self.bias = initialize_parameters(shape=(self.filters), distribution=self.bias_initializer, is_bias=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
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
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x_grad = self.activation.backward(x_grad) if self.activation else x_grad
    
        x = np.pad(self.forward_data, ((0,0),(0,0),(self.padding,)*2,(self.padding,)*2)) if self.padding else self.forward_data
        batch_size, _, output_h, output_w = x_grad.shape
        _, channels, input_h, input_w = self.forward_data.shape
        
        self.bias_grads = x_grad.sum(axis=(0, 2, 3)) # Gradiente de los biases
        
        x_windows = np.lib.stride_tricks.as_strided( # Crear ventanas de entrada
            x,
            shape=(batch_size, channels, output_h, output_w, self.filter_size, self.filter_size),
            strides=(x.strides[0], x.strides[1], 
                    x.strides[2]*self.stride, x.strides[3]*self.stride,
                    x.strides[2], x.strides[3])
        )
        
        self.weights_grads = np.tensordot( # Gradientes de pesos
            x_windows,
            x_grad,
            axes=[(0, 2, 3), (0, 2, 3)]
        ).transpose(1, 2, 0, 3)  # Ajustar dimensiones
        
        pad = self.filter_size - 1
        x_grad_padded = np.pad(x_grad, ((0,0),(0,0),(pad,)*2,(pad,)*2)) # Preparar padding para convolución transpuesta
        
        rot_weights = np.rot90(self.weights, 2, axes=(0,1)) # Rotar pesos 180° para convolución transpuesta

        grad_windows = np.lib.stride_tricks.as_strided( # Ventanas del gradiente de salida
            x_grad_padded,
            shape=(batch_size, self.weights.shape[3], input_h + 2*self.padding, input_w + 2*self.padding, self.filter_size, self.filter_size),
            strides=(x_grad_padded.strides[0], x_grad_padded.strides[1],
                    x_grad_padded.strides[2], x_grad_padded.strides[3],
                    x_grad_padded.strides[2], x_grad_padded.strides[3])
        )
        
        self.data_grads = np.tensordot( # Gradientes de entrada
            grad_windows,
            rot_weights,
            axes=[(1, 4, 5), (3, 0, 1)]
        ).transpose(0, 3, 1, 2)
        
        if self.padding: # Remover padding si es necesario
            self.data_grads = self.data_grads[
                :, :, 
                self.padding:self.padding + input_h, 
                self.padding:self.padding + input_w
            ]
        
        return self.data_grads
    
    def backward_deprecated(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás sin optimización (Primer implementación). Mas lenta que backward.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, _, output_height, output_width = x_grad.shape # (batch_size, filters, output_height, output_width)
        _, _, input_height, input_width = self.forward_data.shape
        x_grad = self.activation.backward(x_grad) if self.activation else x_grad
        x = self.forward_data
        
        if self.padding: 
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant') # Añadir padding a la entrada
        
        self.data_grads = np.zeros_like(x) # Gradientes con respecto a la entrada (batch_size, channels, height, width)
        self.weights_grads = np.zeros_like(self.weights)  # Gradientes con respecto a los pesos (filter_size, filter_size, channels, filters)
        self.bias_grads = np.sum(x_grad, axis=(0, 2, 3))  # Gradiente del sesgo. Se calcula directamente dada su simplicidad. (filters)

        for h in range(output_height):
            for w in range(output_width):
                h_start, w_start = h * self.stride, w * self.stride # Ajustar las posiciones iniciales segun el stride. Por defecto stride = 1.
                h_end, w_end = h_start + self.filter_size, w_start + self.filter_size # Ajusta las posiciones finales sumandoles el tamaño del filtro.
                
                x_region = x[:, :, h_start:h_end, w_start:w_end] # Extraer la región de la entrada
                self.weights_grads += np.tensordot(x_region, x_grad[:, :, h, w], axes=([0], [0])).transpose(1, 2, 0, 3) # Gradientes con respecto a los pesos. Hace falta transponer?

                for b in range(batch_size): # Calcular los gradientes con respecto a la entrada
                    self.data_grads[b, :, h_start:h_end, w_start:w_end] += np.tensordot(
                        x_grad[b, :, h, w], self.weights, axes=([0], [3])
                    ).transpose(2, 0, 1) # Hace falta transponer?
        
        if self.padding:
            self.data_grads = self.data_grads[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding] # Eliminar el padding de los gradientes con respecto a la entrada

        return self.data_grads

class Pool2D(Layer):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa pooling 2D.
        
        :param pool_size: tamaño del pooling
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        self.pool_size = pool_size
        self.stride = self.pool_size if stride is None else stride
        self.padding = padding
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        if self.padding == 'valid': 
            self.padding = 0

        elif self.padding == 'same':
            output_size = math.ceil(input_shape[2] / self.stride)
            padd_total = max(0, (output_size - 1) * self.stride + self.pool_size - input_shape[2])
            self.padding = math.ceil(padd_total // 2)

        elif self.padding >= 0:
            self.padding = self.padding
        else:
            raise ValueError(f'Padding {self.padding} no soportado. Utilice "valid" o "same"')
        out_height = (input_shape[2] + 2 * self.padding - self.pool_size) // self.stride + 1
        out_width = (input_shape[3] + 2 * self.padding - self.pool_size) // self.stride + 1

        self.output_shape = (None, input_shape[1], out_height, out_width)

    def __call__(x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        raise NotImplementedError
    
    def backward(x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        raise NotImplementedError

class MaxPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa max pooling 2D.
        
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param pool_size: tamaño del pooling
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

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
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data  # (batches, channels, height, width)
        batches, channels, height, width = x.shape
        pool_h, pool_w = self.pool_size, self.pool_size
        stride_h, stride_w = self.stride, self.stride

        output_height = (height - pool_h) // stride_h + 1
        output_width = (width - pool_w) // stride_w + 1

        new_shape = (batches, channels, output_height, output_width, pool_h, pool_w)
        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2] * stride_h,
            x.strides[3] * stride_w,
            x.strides[2],
            x.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=strides) # Ventanas

        flattened = windows.reshape(*new_shape[:-2], -1) # Aplanar ventanas
        max_indices = flattened.argmax(axis=-1) # Obtener índices de posiciones máximas
        
        mask_flat = np.zeros(flattened.shape, dtype=bool)
        np.put_along_axis(mask_flat, max_indices[..., None], True, -1)
        mask = mask_flat.reshape(windows.shape) # Máscara con los primeros máximos

        b_idx, c_idx, oh_idx, ow_idx, ph_idx, pw_idx = np.nonzero(mask)
        h_coords = oh_idx * stride_h + ph_idx
        w_coords = ow_idx * stride_w + pw_idx

        self.data_grads = np.zeros_like(x) # Acumular gradientes usando numpy.add.at
        np.add.at(
            self.data_grads,
            (b_idx, c_idx, h_coords, w_coords),
            x_grad[b_idx, c_idx, oh_idx, ow_idx]
        )

        return self.data_grads
    
    def backward_deprecated(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        self.data_grads = np.zeros(x.shape) # (batches, channels, height, width)
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
                    self.data_grads[b, :, h_start:h_end, w_start:w_end] += grad * mask # Gradientes con respecto a la entrada
        return self.data_grads

class MinPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa min pooling 2D.
        
        :param pool_size: tamaño del pooling
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
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
                x.strides[0],
                x.strides[1],
                x.strides[2] * self.stride,
                x.strides[3] * self.stride,
                x.strides[2],
                x.strides[3],
            ),
            writeable=False,
        )

        output = np.amin(strided_x, axis=(-2, -1)) # Aplicar min pooling
        output = np.reshape(output, (batch_size, channels, output_height, output_width)) # Asegurarse que tener formato estándar (batches, channels, height, width)

        return output

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data  # (batches, channels, height, width)
        batches, channels, height, width = x.shape
        pool_h, pool_w = self.pool_size, self.pool_size
        stride_h, stride_w = self.stride, self.stride

        output_height = (height - pool_h) // stride_h + 1
        output_width = (width - pool_w) // stride_w + 1

        new_shape = (batches, channels, output_height, output_width, pool_h, pool_w)
        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2] * stride_h,
            x.strides[3] * stride_w,
            x.strides[2],
            x.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=strides) # Ventanas

        flattened = windows.reshape(*new_shape[:-2], -1) # Aplanar ventanas
        min_indices = flattened.argmin(axis=-1) # Obtener índices de posiciones mínimas
        
        mask_flat = np.zeros(flattened.shape, dtype=bool)
        np.put_along_axis(mask_flat, min_indices[..., None], True, -1)
        mask = mask_flat.reshape(windows.shape) # Máscara con los primeros mínimos

        b_idx, c_idx, oh_idx, ow_idx, ph_idx, pw_idx = np.nonzero(mask)
        h_coords = oh_idx * stride_h + ph_idx
        w_coords = ow_idx * stride_w + pw_idx

        self.data_grads = np.zeros_like(x) # Acumular gradientes usando numpy.add.at
        np.add.at(
            self.data_grads,
            (b_idx, c_idx, h_coords, w_coords),
            x_grad[b_idx, c_idx, oh_idx, ow_idx]
        )

        return self.data_grads

    def backward_deprecated(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás sin optimización (Primer implementación). Mas lenta que backward.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        self.data_grads = np.zeros(x.shape) # (batches, channels, height, width)
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
                    self.data_grads[b, :, h_start:h_end, w_start:w_end] += grad * mask # Gradientes con respecto a la entrada
        return self.data_grads

class AvgPool2D(Pool2D):
    def __init__(self, pool_size: int, input_shape: tuple[int, ...]=None, name: str='max_pooling', stride: int=None, padding: str='valid'):
        """
        Constructor de una capa average pooling 2D.
        
        :param pool_size: tamaño del pooling
        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param: name: nombre de la capa
        :param stride: paso de la convolución. Por defecto None
        :param padding: padding de la convolución. Por defecto 'valid'
        """
        super().__init__(input_shape=input_shape, name=name, pool_size=pool_size, stride=stride, padding=padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        _, _, output_height, output_width = self.output_shape # (batches, channels, output_height, output_width)
        batch_size, channels, _, _ = x.shape # (batches, channels, height, width)
        
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
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
                x.strides[0],
                x.strides[1],
                x.strides[2] * self.stride,
                x.strides[3] * self.stride,
                x.strides[2],
                x.strides[3],
            ),
            writeable=False,
        )

        output = np.mean(strided_x, axis=(-2, -1)) # Aplicar avg pooling
        output = np.reshape(output, (batch_size, channels, output_height, output_width)) # Asegurarse que tener formato estándar (batches, channels, height, width)

        return output

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data  # (batches, channels, height, width)
        batches, channels, height, width = x.shape
        pool_h, pool_w = self.pool_size, self.pool_size
        stride_h, stride_w = self.stride, self.stride
        pool_area = pool_h * pool_w

        output_height = (height - pool_h) // stride_h + 1
        output_width = (width - pool_w) // stride_w + 1

        new_shape = (batches, channels, output_height, output_width, pool_h, pool_w)

        scaled_grad = x_grad[..., np.newaxis, np.newaxis] / pool_area
        scaled_grad_expanded = np.broadcast_to(scaled_grad, new_shape) # Calcular gradientes escalados

        b_idx, c_idx, oh_idx, ow_idx, ph_idx, pw_idx = np.indices(new_shape)
        h_coords = oh_idx * stride_h + ph_idx
        w_coords = ow_idx * stride_w + pw_idx

        # Aplanar índices y gradientes
        b_flat = b_idx.ravel()
        c_flat = c_idx.ravel()
        h_flat = h_coords.ravel()
        w_flat = w_coords.ravel()
        grad_flat = scaled_grad_expanded.ravel()

        self.data_grads = np.zeros_like(x)
        np.add.at(self.data_grads, (b_flat, c_flat, h_flat, w_flat), grad_flat) # Acumular gradientes

        return self.data_grads

    def backward_deprecated(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás sin optimización (Primer implementación). Mas lenta que backward.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        x = self.forward_data # (batches, channels, height, width)
        self.data_grads = np.zeros(x.shape) # (batches, channels, height, width)
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
                    self.data_grads[b, :, h_start:h_end, w_start:w_end] += avg_grad # Gradientes con respecto a la entrada
        return self.data_grads

class RecurrentLayer(Layer):
    def __init__(self, units: int, input_shape: tuple[any,...]=None, name: str='recurrent_layer', activation: Activation=Tanh(), recurrent_activation: Activation=Sigmoid(), return_sequences: bool=False, return_state: bool=False, weights: np.array=None, recurrent_weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', recurrent_weight_initializer: str='orthogonal', bias_initializer: str='zeros', short_term_memory: np.array=None) -> None:
        """
        Constructor de una capa RNN simple

        :param units: cantidad de unidades
        :param input_shape: forma de la entrada
        :param name: nombre de la capa
        :param activation: función de activación
        :param recurrent_activation: función de activación recurrente
        :param return_sequences: indica si se debe devolver una secuencia de salidas
        :param return_state: indica si se debe devolver el estado de la capa
        :param weights: pesos de la capa
        :param recurrent_weights: pesos recurrentes de la capa
        :param bias: bias de la capa
        :param weight_initializer: inicializador de pesos
        :param recurrent_weight_initializer: inicializador de pesos recurrentes
        :param bias_initializer: inicializador de bias
        :param short_term_memory: memoria de corto plazo
        """
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.weights = weights
        self.recurrent_weights = recurrent_weights
        self.bias = bias
        self.weights_grads = None
        self.recurrent_weights_grads = None
        self.bias_grads = None
        self.weights_initializer = weight_initializer
        self.recurrent_weight_initializer = recurrent_weight_initializer
        self.bias_initializer = bias_initializer
        self.short_term_memory = short_term_memory
        self.hist_states = []
        super().__init__(input_shape=input_shape, name=name)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, timesteps, input_dim)
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        raise NotImplementedError

    def backward(self, grad_x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        raise NotImplementedError

    def calculate_output(self, outputs: np.ndarray, states: tuple[np.ndarray, ...]) -> None:
        """
        Retorna los resultados de la propagación hacia adelante segun los parámetros de la capa.

        :param outputs: salidas de la capa
        :param states: estados de la capa
        :return: salidas de la capa
        """
        if self.return_sequences:
            if self.return_state:
                return outputs, *states
            return outputs
        else:
            outputs = outputs[:, -1, :]
            if self.return_state:
                return outputs, *states
            return outputs

class SimpleRNN(RecurrentLayer):
    def __init__(self, units: int, input_shape: tuple[any,...]=None, name: str='simple_rnn', activation: Activation=Tanh(), return_sequences: bool=False, return_state: bool=False, weights: np.array=None, recurrent_weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', recurrent_weight_initializer: str='orthogonal', bias_initializer: str='zeros', short_term_memory: np.array=None) -> None:
        """
        Constructor de una capa RNN simple

        :param units: cantidad de unidades
        :param input_shape: forma de la entrada
        :param name: nombre de la capa
        :param activation: función de activación
        :param return_sequences: indica si se debe devolver una secuencia de salidas
        :param return_state: indica si se debe devolver el estado de la capa
        :param weights: pesos de la capa
        :param recurrent_weights: pesos recurrentes de la capa
        :param bias: bias de la capa
        :param weight_initializer: inicializador de pesos
        :param recurrent_weight_initializer: inicializador de pesos recurrentes
        :param bias_initializer: inicializador de bias
        :param short_term_memory: memoria de corto plazo
        """
        super().__init__(units=units, input_shape=input_shape, name=name, activation=activation, return_sequences=return_sequences, return_state=return_state, weights=weights, recurrent_weights=recurrent_weights, bias=bias, weight_initializer=weight_initializer, recurrent_weight_initializer=recurrent_weight_initializer, bias_initializer=bias_initializer, short_term_memory=short_term_memory)

    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        if self.return_sequences:
            self.output_shape = (None, input_shape[1], self.units)
        else:
            self.output_shape = (None, self.units)
        if self.weights is None:
            self.weights = initialize_parameters(shape=(input_shape[-1], self.units), distribution=self.weights_initializer)
        if self.recurrent_weights is None:
            self.recurrent_weights = initialize_parameters(shape=(self.units, self.units), distribution=self.recurrent_weight_initializer)
        if self.bias is None:
            self.bias = initialize_parameters(shape=(self.units), distribution=self.bias_initializer, is_bias=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        batch_size, timesteps, _ = x.shape # (batches, timesteps, input_dim)

        self.short_term_memory = np.zeros((batch_size, self.units))
        self.hist_states = [self.short_term_memory.copy()]
        outputs = []

        for t in range(timesteps):
            xt = x[:, t]
            z = xt @ self.weights + self.short_term_memory @ self.recurrent_weights + self.bias

            self.short_term_memory = self.activation(z)
            self.hist_states.append(self.short_term_memory.copy())
            outputs.append(self.short_term_memory.copy())


        outputs = np.stack(outputs, axis=1)
        return self.calculate_output(outputs, [self.short_term_memory])

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, timesteps, _ = self.forward_data.shape # (batches, timesteps, input_dim)

        if self.return_sequences:
            dY_list = [x_grad[:, t, :] for t in range(timesteps)]
        else:
            dY_list = [np.zeros((batch_size, self.units)) for _ in range(timesteps)]
            dY_list[-1] = x_grad

        self.weights_grads = np.zeros_like(self.weights)
        self.recurrent_weights_grads = np.zeros_like(self.recurrent_weights)
        self.bias_grads = np.zeros_like(self.bias)

        dh_next = np.zeros((batch_size, self.units))
        self.data_grads = [None] * timesteps

        for t in reversed(range(timesteps)):
            dz = dY_list[t] + dh_next
            h_t = self.hist_states[t + 1]

            saved_fd = self.activation.forward_data
            self.activation.forward_data = h_t
            d_pre = self.activation.backward(dz)
            self.activation.forward_data = saved_fd
            
            self.weights_grads += self.forward_data[:, t, :].T @ d_pre
            h_prev = self.hist_states[t]
            self.recurrent_weights_grads += h_prev.T @ d_pre
            self.bias_grads += np.sum(d_pre, axis=0)

            dh_next = d_pre @ self.recurrent_weights.T
            self.data_grads[t] = d_pre @ self.weights.T

        self.data_grads = np.stack(self.data_grads, axis=1)
        
        return self.data_grads
        
class GRU(RecurrentLayer):
    def __init__(self, units: int, input_shape: tuple[any,...]=None, name: str='gru', activation: Activation=Tanh(), recurrent_activation: Activation=Sigmoid(), return_sequences: bool=False, return_state: bool=False, weights: np.array=None, recurrent_weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', recurrent_weight_initializer: str='orthogonal', bias_initializer: str='zeros', short_term_memory: np.array=None, reset_after: bool=True) -> None:
        """
        Constructor de una capa GRU (Gated Recurrent Unit)

        :param units: cantidad de unidades
        :param input_shape: forma de la entrada
        :param name: nombre de la capa
        :param activation: función de activación
        :param recurrent_activation: función de activación recurrente
        :param return_sequences: indica si se debe devolver una secuencia de salidas
        :param return_state: indica si se debe devolver el estado de la capa
        :param weights: pesos de la capa
        :param recurrent_weights: pesos recurrentes de la capa
        :param bias: bias de la capa
        :param weight_initializer: inicializador de pesos
        :param recurrent_weight_initializer: inicializador de pesos recurrentes
        :param bias_initializer: inicializador de bias
        :param short_term_memory: memoria de corto plazo
        :param reset_after: indica si aplicar el reset gate después de la multiplicación con los pesos recurrentes
        """
        self.reset_after = reset_after
        self.z_list = []
        self.r_list = []
        self.h_tilde_list = []
        super().__init__(units=units, input_shape=input_shape, name=name, activation=activation, recurrent_activation=recurrent_activation, return_sequences=return_sequences, return_state=return_state, weights=weights, recurrent_weights=recurrent_weights, bias=bias, weight_initializer=weight_initializer, recurrent_weight_initializer=recurrent_weight_initializer, bias_initializer=bias_initializer, short_term_memory=short_term_memory)
    
    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        if self.return_sequences:
            self.output_shape = (None, input_shape[1], self.units)
        else:
            self.output_shape = (None, self.units)
        if self.weights is None:
            self.weights = initialize_parameters(shape=(input_shape[-1], 3 * self.units), distribution=self.weights_initializer)
        if self.recurrent_weights is None:
            self.recurrent_weights = initialize_parameters(shape=(self.units, 3 * self.units), distribution=self.recurrent_weight_initializer)
        if self.bias is None:
            if self.reset_after:
                self.bias = initialize_parameters(shape=(2, 3 * self.units), distribution=self.bias_initializer, is_bias=True)
            else:
                self.bias = initialize_parameters(shape=(3 * self.units), distribution=self.bias_initializer, is_bias=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        batch_size, timesteps, _ = x.shape # (batches, timesteps, input_dim)

        W_z, W_r, W_h = np.hsplit(self.weights, 3)
        U_z, U_r, U_h = np.hsplit(self.recurrent_weights, 3)

        if self.reset_after:
            b_in = self.bias[0]
            b_rec = self.bias[1]
            b_z = b_in[:self.units] + b_rec[:self.units]
            b_r = b_in[self.units:2*self.units] + b_rec[self.units:2*self.units]
            b_h = b_in[2*self.units:] + b_rec[2*self.units:]
        else:
            b_z, b_r, b_h = np.split(self.bias, 3)

        self.short_term_memory = np.zeros((batch_size, self.units))
        self.hist_states = [self.short_term_memory.copy()]
        outputs = []

        for t in range(timesteps):
            xt = x[:, t]
            h_prev = self.short_term_memory

            # Update gate
            z = self.recurrent_activation(xt @ W_z + h_prev @ U_z + b_z)
            # Reset gate
            r = self.recurrent_activation(xt @ W_r + h_prev @ U_r + b_r)
            # Candidato
            if self.reset_after:
                h_tilde = self.activation(xt @ W_h + r * (h_prev @ U_h) + b_h)
            else:
                h_tilde = self.activation(xt @ W_h + (r * h_prev) @ U_h + b_h)
            
            self.short_term_memory = z * h_prev + (1 - z) * h_tilde

            outputs.append(self.short_term_memory.copy())
            self.z_list.append(z)
            self.r_list.append(r)
            self.hist_states.append(self.short_term_memory.copy())
            self.h_tilde_list.append(h_tilde)
        
        outputs = np.stack(outputs, axis=1) # (batch, timesteps, units)
        return self.calculate_output(outputs, [self.short_term_memory])

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, timesteps, _ = self.forward_data.shape # (batches, timesteps, input_dim)
        W_z, W_r, W_h = np.hsplit(self.weights, 3)
        U_z, U_r, U_h = np.hsplit(self.recurrent_weights, 3)
        b_rec = self.bias[1]

        if self.return_sequences:
            dY_list = [x_grad[:, t, :] for t in range(timesteps)]
        else:
            dY_list = [np.zeros((batch_size, self.units)) for _ in range(timesteps)]
            dY_list[-1] = x_grad

        self.weights_grads = np.zeros_like(self.weights)
        self.recurrent_weights_grads = np.zeros_like(self.recurrent_weights)
        self.bias_grads = np.zeros_like(self.bias)

        dh_next = np.zeros((batch_size, self.units))
        self.data_grads = [None] * timesteps
        dW_z = np.zeros_like(W_z)
        dW_r = np.zeros_like(W_r)
        dW_h = np.zeros_like(W_h)
        dU_z = np.zeros_like(U_z)
        dU_r = np.zeros_like(U_r)
        dU_h = np.zeros_like(U_h)
        db_in = np.zeros_like(self.bias[0]) if self.reset_after else None
        db_rec = np.zeros_like(self.bias[1]) if self.reset_after else None

        for t in reversed(range(timesteps)):
            dh = dY_list[t] + dh_next
            
            h_prev = self.hist_states[t]
            z = self.z_list[t]
            r = self.r_list[t]
            h_tilde = self.h_tilde_list[t]
            x_t = self.forward_data[:, t]

            dh_tilde = dh * (1 - z)
            dh_tilde_pre = self.activation.backward(dh_tilde, x=h_tilde)

            if self.reset_after:
                dr = dh_tilde_pre * (h_prev @ U_h + b_rec[2*self.units:])
            else:
                dr = (dh_tilde_pre @ U_h.T) * h_prev
            dr_pre = self.recurrent_activation.backward(dr, x=r)

            dz = dh * (h_prev - h_tilde)
            dz_pre = self.recurrent_activation.backward(dz, x=z)

            dW_h += x_t.T @ dh_tilde_pre
            dW_r += x_t.T @ dr_pre
            dW_z += x_t.T @ dz_pre

            if self.reset_after:
                dU_h += h_prev.T @ (dh_tilde_pre * r)
            else:
                dU_h += (r * h_prev).T @ dh_tilde_pre
            dU_r += h_prev.T @ dr_pre
            dU_z += h_prev.T @ dz_pre

            if self.reset_after:
                # bias entrada
                db_in[2*self.units:] += np.sum(dh_tilde_pre, axis=0)
                db_in[self.units:2*self.units] += np.sum(dr_pre, axis=0)
                db_in[:self.units] += np.sum(dz_pre, axis=0)
                # bias recurrente
                db_rec[2*self.units:] += np.sum(dh_tilde_pre * r, axis=0)
                db_rec[self.units:2*self.units] += np.sum(dr_pre, axis=0)
                db_rec[:self.units] += np.sum(dz_pre, axis=0)
            else:
                self.bias_grads[2*self.units:] += np.sum(dh_tilde_pre, axis=0)
                self.bias_grads[self.units:2*self.units] += np.sum(dr_pre, axis=0)
                self.bias_grads[:self.units] += np.sum(dz_pre, axis=0)

            dh_prev = dh * z 
            if self.reset_after:
                dh_prev += (
                    dz_pre @ U_z.T +
                    dr_pre @ U_r.T +
                    (dh_tilde_pre * r) @ U_h.T
                )
            else:
                dh_prev += (
                    dz_pre @ U_z.T +
                    dr_pre @ U_r.T +
                    (dh_tilde_pre @ U_h.T) * r
                )
            dh_next = dh_prev

            dx_t = (
                dz_pre @ W_z.T +
                dr_pre @ W_r.T +
                dh_tilde_pre @ W_h.T
            )
            self.data_grads[t] = dx_t

        self.data_grads = np.stack(self.data_grads, axis=1)
        self.weights_grads = np.hstack([dW_z, dW_r, dW_h])
        self.recurrent_weights_grads = np.hstack([dU_z, dU_r, dU_h])
        if self.reset_after:
            self.bias_grads = np.array([db_in, db_rec])
        
        return self.data_grads

class LSTM(RecurrentLayer):
    def __init__(self, units: int, input_shape: tuple[any,...]=None, name: str='lstm', activation: Activation=Tanh(), recurrent_activation: Activation=Sigmoid(), return_sequences: bool=False, return_state: bool=False, unit_forget_bias: bool=True, weights: np.array=None, recurrent_weights: np.array=None, bias: np.array=None, weight_initializer: str='glorot_uniform', recurrent_weight_initializer: str='orthogonal', bias_initializer: str='zeros', short_term_memory: np.array=None, long_term_memory: np.array=None) -> None:
        """
        Constructor de una capa LSTM

        :param units: cantidad de unidades
        :param input_shape: forma de la entrada
        :param name: nombre de la capa
        :param activation: función de activación
        :param recurrent_activation: función de activación recurrente
        :param return_sequences: indica si se debe devolver una secuencia de salidas
        :param return_state: indica si se debe devolver el estado de la capa
        :param unit_forget_bias: indica si se debe incluir bias para el unit_forget_bias
        :param weights: pesos de la capa
        :param recurrent_weights: pesos recurrentes de la capa
        :param bias: bias de la capa
        :param weight_initializer: inicializador de pesos
        :param recurrent_weight_initializer: inicializador de pesos recurrentes
        :param bias_initializer: inicializador de bias
        :param short_term_memory: memoria de corto plazo
        :param long_term_memory: memoria de largo plazo
        """
        self.unit_forget_bias = unit_forget_bias
        self.long_term_memory = long_term_memory
        self.f_list = []
        self.i_list = []
        self.o_list = []
        self.c_tilde_list = []
        self.c_list = []
        self.hist_states = []
        self.tanh_c_list = []
        super().__init__(units=units, input_shape=input_shape, name=name, activation=activation, recurrent_activation=recurrent_activation, return_sequences=return_sequences, return_state=return_state, weights=weights, recurrent_weights=recurrent_weights, bias=bias, weight_initializer=weight_initializer, recurrent_weight_initializer=recurrent_weight_initializer, bias_initializer=bias_initializer, short_term_memory=short_term_memory)


    def compile(self, input_shape: tuple[int, ...]=None) -> None:
        """
        Compila la capa, inicializando sus parámetros y generando las dimensiones de salida de la capa.

        :param input_shape: tupla con las dimensiones de entrada (batches, channels, height, width)
        """
        self.input_shape = input_shape
        if self.return_sequences:
            self.output_shape = (None, input_shape[1], self.units)
        else:
            self.output_shape = (None, self.units)
        if self.weights is None:
            self.weights = initialize_parameters(shape=(self.input_shape[-1], 4 * self.units), distribution=self.weights_initializer)
        if self.recurrent_weights is None:
            self.recurrent_weights = initialize_parameters(shape=(self.units, 4 * self.units), distribution=self.recurrent_weight_initializer)
        if self.bias is None:
            self.bias = initialize_parameters(shape=(4 * self.units), distribution=self.bias_initializer, is_bias=True)
            if self.unit_forget_bias:
                self.bias[self.units:2*self.units] += 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: matriz de entrada
        :return: matriz de salida de la capa
        """
        self.forward_data = x
        batch_size, timesteps, _ = x.shape # (batches, timesteps, input_dim)
        W_i, W_f, W_c, W_o = np.hsplit(self.weights, 4)
        U_i, U_f, U_c, U_o = np.hsplit(self.recurrent_weights, 4)
        b_i, b_f, b_c, b_o = np.split(self.bias, 4)

        self.long_term_memory = np.zeros((batch_size, self.units))
        self.short_term_memory = np.zeros((batch_size, self.units))
        self.hist_states = [self.short_term_memory.copy()]
        self.c_list = [self.long_term_memory.copy()]
        outputs = []

        for t in range(timesteps):
            xt = x[:, t]

            # forget gate
            f = xt @ W_f + self.short_term_memory @ U_f + b_f
            forget_factor = self.recurrent_activation(f)
            self.long_term_memory *= forget_factor

            # input gate
            i = xt @ W_i + self.short_term_memory @ U_i + b_i
            c_tilde = xt @ W_c + self.short_term_memory @ U_c + b_c
            input_factor = self.recurrent_activation(i)
            candidate_factor = self.activation(c_tilde)
            self.long_term_memory += input_factor * candidate_factor

            # output gate
            o = xt @ W_o + self.short_term_memory @ U_o + b_o
            output_factor = self.recurrent_activation(o)
            self.short_term_memory = output_factor * self.activation(self.long_term_memory)

            self.f_list.append(forget_factor)
            self.i_list.append(input_factor)
            self.o_list.append(output_factor)
            self.c_tilde_list.append(candidate_factor)
            self.c_list.append(self.long_term_memory.copy())
            self.hist_states.append(self.short_term_memory.copy())
            self.tanh_c_list.append(self.activation(self.long_term_memory.copy()))

            outputs.append(self.short_term_memory.copy())
        
        outputs = np.stack(outputs, axis=1) # (batches, timesteps, units)
        return self.calculate_output(outputs, [self.short_term_memory, self.long_term_memory])

    def backward(self, x_grad: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param x_grad: gradientes de la propagación hacia adelante
        :return: gradientes de la propagación hacia atrás
        """
        batch_size, timesteps, _ = self.forward_data.shape

        W_i, W_f, W_c, W_o = np.hsplit(self.weights, 4)
        U_i, U_f, U_c, U_o = np.hsplit(self.recurrent_weights, 4)
        b_i, b_f, b_c, b_o = np.split(self.bias, 4)

        if self.return_sequences:
            dY_list = [x_grad[:, t, :] for t in range(timesteps)]
        else:
            dY_list = [np.zeros((batch_size, self.units)) for _ in range(timesteps)]
            dY_list[-1] = x_grad
    
        self.weights_grads = np.zeros_like(self.weights)
        self.recurrent_weights_grads = np.zeros_like(self.recurrent_weights)
        self.bias_grads = np.zeros_like(self.bias)
        self.data_grads = [None] * timesteps

        dW_i = np.zeros_like(W_i)
        dW_f = np.zeros_like(W_f)
        dW_c = np.zeros_like(W_c)
        dW_o = np.zeros_like(W_o)
        dU_i = np.zeros_like(U_i)
        dU_f = np.zeros_like(U_f)
        dU_c = np.zeros_like(U_c)
        dU_o = np.zeros_like(U_o)
        db_i = np.zeros_like(b_i)
        db_f = np.zeros_like(b_f)
        db_c = np.zeros_like(b_c)
        db_o = np.zeros_like(b_o)

        dh_next = np.zeros((batch_size, self.units))
        dc_next = np.zeros((batch_size, self.units))

        for t in reversed(range(timesteps)):
            dh = dY_list[t] + dh_next

            h_prev = self.hist_states[t]
            c_prev = self.c_list[t]

            i_t = self.i_list[t]
            f_t = self.f_list[t]
            o_t = self.o_list[t]
            c_tilde = self.c_tilde_list[t]
            tanh_c_curr = self.tanh_c_list[t]
            x_t = self.forward_data[:, t]

            dtanh_c = dh * o_t
            dtanh_c_pre = self.activation.backward(dtanh_c, x=tanh_c_curr)

            dc = dtanh_c_pre + dc_next

            do = dh * tanh_c_curr
            do_pre = self.recurrent_activation.backward(do, x=o_t)

            dc_tilde = dc * i_t
            dc_tilde_pre = self.activation.backward(dc_tilde, x=c_tilde)

            di = dc * c_tilde
            di_pre = self.recurrent_activation.backward(di, x=i_t)

            df = dc * c_prev
            df_pre = self.recurrent_activation.backward(df, x=f_t)

            dW_i += x_t.T @ di_pre
            dW_f += x_t.T @ df_pre # mal
            dW_c += x_t.T @ dc_tilde_pre
            dW_o += x_t.T @ do_pre

            dU_i += h_prev.T @ di_pre
            dU_f += h_prev.T @ df_pre
            dU_c += h_prev.T @ dc_tilde_pre
            dU_o += h_prev.T @ do_pre

            db_i += np.sum(di_pre, axis=0)
            db_f += np.sum(df_pre, axis=0) # mal
            db_c += np.sum(dc_tilde_pre, axis=0)
            db_o += np.sum(do_pre, axis=0)

            dh_prev = (
                di_pre @ U_i.T +
                df_pre @ U_f.T +
                dc_tilde_pre @ U_c.T +
                do_pre @ U_o.T
            )

            dc_prev = dc * f_t

            dh_next = dh_prev
            dc_next = dc_prev

            dx_t = (
                di_pre @ W_i.T +
                df_pre @ W_f.T +
                dc_tilde_pre @ W_c.T +
                do_pre @ W_o.T
            )
            self.data_grads[t] = dx_t

        self.weights_grads = np.hstack([dW_i, dW_f, dW_c, dW_o])
        self.recurrent_weights_grads = np.hstack([dU_i, dU_f, dU_c, dU_o])
        self.bias_grads = np.hstack([db_i, db_f, db_c, db_o])

        self.data_grads = np.stack(self.data_grads, axis=1)
        return self.data_grads
    