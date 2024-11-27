import numpy as np

def SGD(params: list[np.ndarray], grads: list[np.ndarray], learning_rate: float, momentum: float=None) -> list[np.ndarray]:
    """
    Actualiza los parámetros utilizando el método de SGD

    :param params: lista de parámetros
    :param grads: lista de gradientes
    :param learning_rate: tasa de aprendizaje
    :param momentum: valor del Momentum. Por defecto None
    :return updated_params: lista de parámetros actualizados
    """
    updated_params = []
    for param, grad in zip(params, grads):
        v = np.zeros_like(param)
        if momentum is not None:
            v = momentum * v - learning_rate * grad
            param += v
        else:
            param -= learning_rate * grad

    #TODO: Terminar funcion. Actualizar parámetros en variable updated_params.
    
    return updated_params