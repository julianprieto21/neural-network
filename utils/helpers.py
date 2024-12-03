import numpy as np


def initialize_parameters(shape: tuple[int, ...], distribution: str='normal') -> np.ndarray:
    """
    Inicializa los parametros de una matriz con una distribución normal

    :param shape: tupla de dimensiones de la matriz
    :param name: nombre del peso
    :return weights: matriz con los parametros inicializados
    """
    if distribution == 'normal':
        parameter = np.random.normal(0, 0.05, shape)
        return parameter
    elif distribution == 'uniform':
        parameter = np.random.uniform(-0.05, 0.05, shape)
        return parameter
    elif distribution == 'glorot_normal':
        fan_in, fan_out = shape
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        parameter = np.random.normal(0, scale, shape)
        return parameter
    elif distribution == 'glorot_uniform':
        fan_in, fan_out = shape
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        parameter = np.random.uniform(-scale, scale, shape)
        return parameter
    elif distribution == 'he_normal':
        fan_in, fan_out = shape
        scale = np.sqrt(2.0 / fan_in)
        parameter = np.random.normal(0, scale, shape)
        return parameter
    elif distribution == 'he_uniform':
        fan_in, fan_out = shape
        scale = np.sqrt(6.0 / fan_in)
        parameter = np.random.uniform(-scale, scale, shape)
        return parameter
    elif distribution == 'zeros':
        parameter = np.zeros(shape)
        return parameter
    elif distribution == 'ones':
        parameter = np.ones(shape)
        return parameter
    
    return None

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