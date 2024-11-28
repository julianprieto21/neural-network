import numpy as np

def cross_entropy(output: np.ndarray, target: np.ndarray, reduction: str='mean') -> float: 
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
    return np.round(loss, 3)

def categorical_cross_entropy(output: np.ndarray, target: np.ndarray, reduction: str='mean') -> float:
    """
    Calcula el loss de la red utilizando el método de cross entropy

    :param output: salida de la red
    :param target: etiquetas reales
    :param reduction: tipo de reducción. Por defecto 'mean'
    :return loss: loss (pérdida) de la red
    """
    epsilon = 1e-10
    output = np.clip(output, epsilon, 1 - epsilon)
    loss = -np.sum(target * np.log(output), axis=1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Reduccion {reduction} no soportada')
    return np.round(loss, 3)

def mean_squared_error(output: np.ndarray, target: np.ndarray, reduction: str='mean') -> float:
    """
    Calcula el loss de la red utilizando el método de error cuadrático medio

    :param output: salida de la red
    :param target: etiquetas reales
    :param reduction: tipo de reducción. Por defecto 'mean'
    :return loss: loss (pérdida) de la red
    """
    loss = np.mean((output - target) ** 2)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Reduccion {reduction} no soportada')
    return np.round(loss, 3)

def categorical_mean_squared_error(output: np.ndarray, target: np.ndarray, reduction: str='mean') -> float:
    """
    Calcula el loss de la red utilizando el método de error cuadrático medio

    :param output: salida de la red
    :param target: etiquetas reales
    :param reduction: tipo de reducción. Por defecto 'mean'
    :return loss: loss (pérdida) de la red
    """
    #TODO: Revisar
    epsilon = 1e-10
    output = np.clip(output, epsilon, 1 - epsilon)
    loss = np.mean(np.sum(target * np.log(output) + (1-target) * np.log(1-output), axis=1))
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Reduccion {reduction} no soportada')
    return np.round(loss, 3)

if __name__ == '__main__':
    y_true = np.array([[0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    loss = cross_entropy(y_pred, y_true)
    print(loss)
    pass