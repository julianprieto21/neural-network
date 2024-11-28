import numpy as np
from training.optimizers import Optimizer
from .layers import Layer

class ConvolutionalNetwork:
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int, layers: list[Layer], optimizer: Optimizer, loss: any, metrics: any, verbose: bool=True) -> None:
        """
        Constructor de un modelo de red neuronal convolucional

        :param input_shape: tupla con las dimensiones de entrada (channels, height, width)
        :param num_classes: cantidad de clases
        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        # self.model = None

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, epochs: int=10, batch_size: int=32) -> None:
        """
        Entrena el modelo de red neuronal convolucional

        :param train_data: matriz de entrada de entrenamiento
        :param train_labels: matriz de etiquetas de entrenamiento
        :param epochs: cantidad de epoches
        :param batch_size: tamaño de la batch
        """
        raise NotImplementedError

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Realiza una predicción sobre el modelo de red neuronal convolucional

        :param test_data: matriz de entrada de prueba
        :return: matriz de predicciones
        """
        raise NotImplementedError
