import numpy as np

class Loss:
    def __init__(self):
        pass

    def __call__(self, target: np.ndarray, output: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, target: np.ndarray, output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CategoricalCrossEntropy(Loss):
    def __init__(self, reduction: str='mean', axis: int=-1):
        self.reduction = reduction
        self.axis = axis
        super().__init__()

    def __call__(self, target: np.ndarray, output: np.ndarray) -> float:
        """
        Calcula el loss de la red utilizando el método de cross entropy

        :param output: salida de la red
        :param target: etiquetas reales
        :param reduction: tipo de reducción. Por defecto 'mean'
        :return loss: loss (pérdida) de la red
        """
        epsilon = 1e-10
        output = output / np.sum(output, self.axis, keepdims=True)
        output = np.clip(output, epsilon, 1.0 - epsilon)
        loss = -np.sum(target * np.log(output), axis=self.axis)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'Reduccion {self.reduction} no soportada')
        return loss

    def backward(self, target, output):
        return output - target

class CrossEntropy(Loss):
    def __init__(self, reduction: str='mean'):
        self.reduction = reduction
        super().__init__()
    
    def __call__(self, target: np.ndarray, output: np.ndarray) -> float:
        epsilon = 1e-10
        output = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.sum(target * np.log(output) + (1-target) * np.log(1-output), axis=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'Reduccion {self.reduction} no soportada')
        return loss

    def backward(self, target: np.ndarray, output: np.ndarray):
        raise NotImplementedError