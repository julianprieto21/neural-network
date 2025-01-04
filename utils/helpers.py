import numpy as np


def initialize_parameters(shape: tuple[int, ...], distribution: str='normal', is_bias: bool=False) -> np.ndarray:
    """
    Inicializa los parametros de una matriz con una distribución normal

    :param shape: tupla de dimensiones de la matriz
    :param name: nombre del peso
    :return parameter: matriz con los parametros inicializados
    """
    def get_fan_in_out(shape):
        if len(shape) == 4:
            _, _, fan_in, fan_out = shape
        elif len(shape) == 2:
            fan_in, fan_out = shape
        else:
            raise ValueError('Dimensiones no soportadas para inicialización de pesos.')
        return fan_in, fan_out
    
    if is_bias:
        if distribution == 'normal':
            return np.random.normal(0, 0.05, shape)
        elif distribution == 'uniform':
            return np.random.uniform(-0.05, 0.05, shape)
        elif distribution == 'zeros':
            return np.zeros(shape)
        elif distribution == 'ones':
            return np.ones(shape)
        else:
            raise ValueError(f'Distribución "{distribution}" no soportada para bias.')
    
    else:
        if distribution == 'normal':
            return np.random.normal(0, 0.05, shape)

        if distribution == 'uniform':
            return np.random.uniform(-0.05, 0.05, shape)
        
        if distribution == 'zeros':
            return np.zeros(shape)
        
        if distribution == 'ones':
            return np.ones(shape)
        
        if distribution in {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}:
            fan_in, fan_out = get_fan_in_out(shape)
            if distribution == 'glorot_normal':
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                return np.random.normal(0, scale, shape)
            
            elif distribution == 'glorot_uniform':
                scale = np.sqrt(6.0 / (fan_in + fan_out))
                return np.random.uniform(-scale, scale, shape)
            
            elif distribution == 'he_normal':
                scale = np.sqrt(2.0 / fan_in)
                return np.random.normal(0, scale, shape)
            
            elif distribution == 'he_uniform':
                scale = np.sqrt(6.0 / fan_in)
                return np.random.uniform(-scale, scale, shape)
            
        else:
            raise ValueError(f'Distribución {distribution} no soportada para pesos.')

def one_hot_encoder(y: np.ndarray, num_classes: int=None) -> np.ndarray:
    """
    Codifica una matriz de etiquetas en una matriz de one-hot

    :param y: matriz de etiquetas
    :param n_classes: cantidad de clases
    :return y: matriz de one-hot
    """
    y = np.array(y)

    if num_classes is None:
        num_classes = np.max(y) + 1

    y = np.eye(num_classes)[y]
    return y