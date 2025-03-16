import numpy as np


def accuracy(target: np.ndarray, output: np.ndarray, k: int=1) -> list[float]:
    """
    Calcula el accuracy estándar (top-k) basado en la salida y las etiquetas reales.

    :param target: etiquetas reales
    :param output: etiquetas predichas
    :param k: top-k que se quiere calcular. Por defecto es 1
    :return: accuracy
    """
    output = np.array(output)
    target = np.array(target)
    
    accuracy = 1
    return accuracy

def categorical_accuracy(target: np.ndarray, output: np.ndarray, k: int=1) -> list[float]:
    """
    Calcula el accuracy categórico (top-k) basado en la salida y las etiquetas reales.

    :param target: etiquetas reales
    :param output: etiquetas predichas
    :param k: top-k que se quiere calcular. Por defecto es 1
    :return accuracy
    """
    output = np.array(output)
    target = np.array(target)
    if output.shape[1] != target.shape[1]:
        raise ValueError('"Output" y "target" deben tener el mismo número de columnas.')
    elif k > output.shape[1]:
        raise ValueError('"Top-k" debe ser igual o menor que el número de columnas de "output".')
    elif k < 1:
        raise ValueError('"Top-k" debe ser igual o mayor a 1.')
    
    target = target.argmax(axis=1) # Cantidad de predicciones que se quieren tomar en cuenta
    batch_size = target.shape[0] # Cantidad de entradas en la batch

    topk_ind = np.argsort(output, axis=1)[:, -k:][:, ::-1]
    target = target[:, np.newaxis]
    correct = np.equal(topk_ind, target)

    res = []
    correct_i = correct[:, :k].sum(0) # Suma las predicciones correctas de acuerdo al top-k
    res.append(correct_i / batch_size) # Calcula el porcentaje de correctas de acuerdo al top-k
    return res[-1][0] if len(res[-1]) == 1 else res[-1] # Devuelve la lista con el porcentaje de correctas de acuerdo al top-k

def mean_squared_error():
    pass

def mean_absolute_error():
    pass

def r_squared():
    pass