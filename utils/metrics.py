import numpy as np


def accuracy(target: np.ndarray, output: np.ndarray, k: int=1) -> list[float]:
    """
    Calcula el accuracy top-k de acuerdo a la salida y la etiqueta

    :param output: salida de la red
    :param target: etiquetas reales
    :param topk: top-k que se quiere calcular. Por defecto es el top-1 (accuracy estandar)
    :return accuracy: lista con el accuracy top-k
    """
    if output.shape[1] != target.shape[1]:
        raise ValueError('Output and target must have the same number of columns')
    elif k > output.shape[1]:
        raise ValueError('Top-k must be less than or equal to the number of columns in output')
    elif k < 1:
        raise ValueError('Top-k must be greater than or equal to 1')
    
    target = target.argmax(axis=1) # Cantidad de predicciones que se quieren tomar en cuenta
    batch_size = target.shape[0] # Cantidad de entradas en la batch

    topk_ind = np.argsort(output, axis=1)[:, -k:][:, ::-1]
    target = target[:, np.newaxis]
    correct = np.equal(topk_ind, target)

    res = []
    correct_i = correct[:, :k].sum(0) # Suma las predicciones correctas de acuerdo al top-k
    res.append(correct_i / batch_size) # Calcula el porcentaje de correctas de acuerdo al top-k
    return res[-1][0] if len(res[-1]) == 1 else res[-1] # Devuelve la lista con el porcentaje de correctas de acuerdo al top-k