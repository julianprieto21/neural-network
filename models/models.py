import numpy as np
from training.optimizers import Optimizer
from .layers import Layer

class NeuralNetwork:
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss: any, metrics: any, verbose: bool=True) -> None:
        """
        Constructor de un modelo de red neuronal

        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso
        """
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.params = None
    
    def summary(self) -> None:
        """
        Muestra un resumen del modelo
        """
        parameters = 0
        print(f'Modelo de red neuronal con {len(self.layers) + 1} capas:')
        print(f'\tEntrada: {self.layers[0].input_shape}')
        for layer in self.layers:
            if layer.weights is not None: 
                parameters += layer.weights.size + layer.bias.size
            print(f'\t{layer.name}: {layer.__class__.__name__} - {layer.output_shape}')
            # if layer.weights is None: continue
            # parameters += layer.weights.size
            # parameters += layer.bias.size
        print(f'Optimizador: {self.optimizer.__class__.__name__}')
        print(f'Función de pérdida: {self.loss.__name__}')        
        print(f'Función de cálculo de métricas: {self.metrics.__name__}')
        print(f'Cantidad de parámetros: {parameters}')

    def compile(self) -> None:
        """
        Compila el modelo. Recupera los parámetros y genera las dimensiones de salida de las capas
        """
        first_layer = self.layers[0]
        first_layer.compile()
        output_shape = first_layer.output_shape
        for layer in self.layers[1:]:
            layer.input_shape = output_shape
            layer.compile()
            output_shape = layer.output_shape
        self.params = self.get_params()

    def get_params(self) -> list[np.ndarray]:
        """
        Obtiene los parámetros del modelo

        :return params: lista de parámetros
        """
        params = []
        i = 0
        for layer in self.layers:
            if layer.weights is not None and layer.bias is not None:
                i += 1
                params.append(layer.weights)
                params.append(layer.bias)
        return params

    def _forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        for layer in self.layers:
            data = layer.forward(data)
            print(layer.name, data.shape)
        return data
    
    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :return grad_output: gradientes de la propagación hacia atrás
        """        
        raise NotImplementedError
        # for layer in reversed(self.layers):
        #     grad_output = layer.backward(grad_output)
        # return grad_output

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, epochs: int=10, batch_size: int=32) -> None:
        """
        Entrena el modelo de red neuronal

        :param train_data: matriz de entrada de entrenamiento
        :param train_labels: matriz de etiquetas de entrenamiento
        :param epochs: cantidad de epoches
        :param batch_size: tamaño de la batch
        """
        for epoch in range(epochs):
            for batch in range(0, train_data.shape[0], batch_size):
                batch_data = train_data[batch:batch+batch_size]
                batch_labels = train_labels[batch:batch+batch_size]
                preds = self._forward(batch_data)
                loss = self.loss(preds, batch_labels)
                grads = self._backward(loss)
                params = self.get_params()
                params = self.optimizer(params, grads)

    def predict(self, test_data: np.ndarray, probs: bool=False) -> np.ndarray:
        """
        Realiza una predicción sobre el modelo de red neuronal

        :param test_data: matriz de entrada de prueba
        :param probs: indica si se debe devolver las probabilidades de cada clase o la clase ganadora
        :return: matriz de predicciones
        """
        probs = self._forward(test_data)
        return probs if probs else np.argmax(probs, axis=0)
