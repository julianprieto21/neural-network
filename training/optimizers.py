import numpy as np

class Optimizer:
    def __init__(self, learning_rate: float) -> None:
        """
        Constructor de un optimizador

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param momentum: valor del Momentum. Por defecto None
        :param beta1: valor del beta1. Por defecto 0.9
        :param beta2: valor del beta2. Por defecto 0.999
        :param epsilon: valor del epsilon. Por defecto 1e-8
        """
        self.learning_rate = learning_rate
        self.epsilon = 1e-7

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización especificado

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float=0.01, momentum: float=None) -> None:
        """
        Constructor de un optimizador SGD

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param momentum: valor del Momentum. Por defecto None
        """
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.velocities = None

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización SGD

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        updated_params = []
        if self.velocities is None and self.momentum is not None:
            self.velocities = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.momentum is not None:
                self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
                param += self.velocities[i]
            else:
                param -= self.learning_rate * grad
            param = np.where(abs(param) < self.epsilon, 0, param)
            updated_params.append(param)
        return updated_params

class RMSprop(Optimizer):
    def __init__(self, learning_rate: float=0.001, rho: float=0.9, momentum: float=None) -> None:
        """
        Constructor de un optimizador RMSprop

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param beta1: valor del beta1. Por defecto 0.9
        :param beta2: valor del beta2. Por defecto 0.999
        :param epsilon: valor del epsilon. Por defecto 1e-8
        """
        super().__init__(learning_rate=learning_rate)
        self.rho = rho
        self.momentum = momentum
        self.velocities = None
        self.momentums = None

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        """
        Actualiza los parámetros utilizando el método de optimización RMSprop

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        updated_params = []
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]
        if self.momentums is None and self.momentum is not None:
            self.momentums = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = (self.rho * self.velocities[i]) + (1 - self.rho) * (grad**2)
            denominator = self.velocities[i] + self.epsilon
            increment = (self.learning_rate * grad) / np.sqrt(denominator)
            if self.momentum is not None:
                self.momentums[i] = self.momentum * self.momentums[i] + increment
                param -= self.momentums[i]
            else:
                param -= increment
            param = np.where(abs(param) < self.epsilon, 0, param)
            updated_params.append(param)
        return updated_params

class Adam(Optimizer):
    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999) -> None:
        """
        Constructor de un optimizador Adam

        :param learning_rate: tasa de aprendizaje. Por defecto 0.001
        :param beta1: valor del beta1. Por defecto 0.9
        :param beta2: valor del beta2. Por defecto 0.999
        :param epsilon: valor del epsilon. Por defecto 1e-8
        """
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iteration = 0
        self.momentums = None
        self.velocities = None

    def __call__(self, params, grads):
        """
        Actualiza los parámetros utilizando el método de optimización Adam

        :param params: lista de parámetros   
        :param grads: lista de gradientes
        :return updated_params: lista de parámetros actualizados
        """
        raise NotImplementedError
        # updated_params = []
        # if self.velocities is None:
        #     self.velocities = [np.zeros_like(param) for param in params]
        # if self.momentums is None:
        #     self.momentums = [np.zeros_like(param) for param in params]

        # beta1_power = self.beta1 ** (self.iteration + 1)
        # beta2_power = self.beta2 ** (self.iteration + 1)

        # for i, (param, grad) in enumerate(zip(params, grads)):
        #     print(self.velocities[i])
        #     alpha = self.learning_rate * np.sqrt(1 - beta2_power) / (1 - beta1_power)
        #     self.momentums[i] = (grad - self.momentums[i]) * (1 - self.beta1)
        #     self.velocities[i] = (grad ** self.velocities[i]) - (1 - self.beta2)
        #     param -= alpha * self.momentums[i] / (np.sqrt(self.velocities[i]) + self.epsilon)
        #     param = np.where(abs(param) < self.epsilon, 0, param)
        #     updated_params.append(param)
        # return updated_params