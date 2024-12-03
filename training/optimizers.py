import numpy as np

class Optimizer:
    def __init__(self, learning_rate: float=0.001, momentum: float=None, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8) -> None:
        """
        Constructor de un optimizador

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param momentum: valor del Momentum. Por defecto None
        :param beta1: valor del beta1. Por defecto 0.9
        :param beta2: valor del beta2. Por defecto 0.999
        :param epsilon: valor del epsilon. Por defecto 1e-8
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.updated_params = []

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización especificado

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float=0.001, momentum: float=None) -> None:
        """
        Constructor de un optimizador SGD

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param momentum: valor del Momentum. Por defecto None
        :param beta1: valor del beta1. Por defecto 0.9
        :param beta2: valor del beta2. Por defecto 0.999
        :param epsilon: valor del epsilon. Por defecto 1e-8
        """
        super().__init__(learning_rate, momentum)
        pass

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización SGD

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        updated_params = []
        for param, grad in zip(params, grads):
            if self.momentum is not None:
                param -= self.learning_rate * grad - self.momentum * param
            else:
                param -= self.learning_rate * grad
            updated_params.append(param)
        return updated_params
