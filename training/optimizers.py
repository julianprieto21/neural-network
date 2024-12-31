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
        self.velocity = None

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

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización SGD

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        updated_params = []
        # if self.velocity is None and self.momentum is not None:
        #     self.velocity = [np.zeros_like(param) for param in params]
        for i, (param, grad) in enumerate(zip(params, grads)):
            # if self.momentum is not None:
            #     self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            #     param += self.velocity[i]
            # else:
            param -= self.learning_rate * grad
            param = np.where(abs(param) < 1e-07, 0, param)
            updated_params.append(param)
        return updated_params
