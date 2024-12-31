import numpy as np

def cross_entropy(target: np.ndarray, output: np.ndarray, reduction: str='mean') -> float: 
    epsilon = 1e-10
    output = np.clip(output, epsilon, 1 - epsilon)
    loss = -np.sum(target * np.log(output) + (1-target) * np.log(1-output), axis=1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Reduccion {reduction} no soportada')
    return loss

def categorical_cross_entropy(target: np.ndarray, output: np.ndarray, reduction: str='mean', axis: int=-1) -> float:
    """
    Calcula el loss de la red utilizando el método de cross entropy

    :param output: salida de la red
    :param target: etiquetas reales
    :param reduction: tipo de reducción. Por defecto 'mean'
    :return loss: loss (pérdida) de la red
    """
    epsilon = 1e-10
    output = output / np.sum(output, axis, keepdims=True)
    output = np.clip(output, epsilon, 1.0 - epsilon)
    loss = -np.sum(target * np.log(output), axis=axis)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Reduccion {reduction} no soportada')
    return loss