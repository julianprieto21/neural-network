import numpy as np


def initialize_parameters(shape: tuple[int, ...], distribution: str='normal') -> np.ndarray:
    """
    Inicializa los parametros de una matriz con una distribución normal

    :param shape: tupla de dimensiones de la matriz
    :param name: nombre del peso
    :return weights: matriz con los parametros inicializados
    """
    if distribution == 'normal':
        parameter = np.random.normal(0, 0.01, shape)
        return parameter
    
    return None
