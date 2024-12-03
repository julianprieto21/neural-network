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
        self.compile()
    
    def summary(self) -> None:
        """
        Muestra un resumen del modelo
        """
        parameters = 0
        print(f'Modelo de red neuronal con {len(self.layers) + 1} capas:')
        print(f'- input: Entrada - {self.layers[0].input_shape}')
        for layer in self.layers:
            layer_parameters = 0
            if layer.weights is not None: 
                layer_parameters = layer.weights.size + layer.bias.size
                parameters += layer_parameters
            print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape} - {layer_parameters} params')
            # if layer.weights is None: continue
            # parameters += layer.weights.size
            # parameters += layer.bias.size
        print(f'Optimizador: {self.optimizer.__class__.__name__}')
        print(f'Función de pérdida: {self.loss.__name__}')        
        print(f'Función de cálculo de métricas: {self.metrics.__name__}')
        print(f'Cantidad de parámetros: {parameters}')
        print('---')
        for layer in self.layers:
            if layer.weights is None: continue
            print(layer.name)
            print('Pesos: ', layer.weights.shape)
            print('Bias: ', layer.bias.shape)

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

    def get_params(self) -> list[np.ndarray]:
        """
        Obtiene los parámetros del modelo

        :return params: lista de parámetros
        """
        params = []
        for layer in self.layers:
            if layer.weights is not None and layer.bias is not None:
                params.append(layer.weights)
                params.append(layer.bias)
        return params

    def _get_grads(self) -> list[np.ndarray]:
        """
        Obtiene los gradientes de los parámetros del modelo

        :return grads: lista de gradientes
        """
        grads = []
        for layer in self.layers:
            if layer.weights is not None and layer.bias is not None:
                grads.append(layer.weights_grads)
                grads.append(layer.bias_grads)
        return grads

    def _forward(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def _backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :param data: tensor de entrada
        :return param_grads: gradientes de los parámetros
        """      

        preds = self._forward(x)
        grad_output = preds - y 
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        param_grads = self._get_grads()
        return param_grads
        
    def _update_params(self, param_grads: list[np.ndarray]) -> None:
        """
        Actualiza los parámetros del modelo

        :param param_grads: gradientes de los parámetros	
        """
        # Actualiza los parámetros del objeto del modelo
        self.params = self.optimizer(self.params, param_grads)

        # Actualiza los parámetros del objeto de cada capa
        i = 0
        for layer in self.layers:
            if layer.weights is not None and layer.bias is not None:
                layer.weights = self.params[i]
                layer.bias = self.params[i+1]
                i += 2

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, validation_data: np.ndarray=None, validation_labels: np.ndarray=None, epochs: int=10, batch_size: int=32) -> None:
        """
        Entrena el modelo de red neuronal

        :param train_data: matriz de entrada de entrenamiento
        :param train_labels: matriz de etiquetas de entrenamiento
        :param epochs: cantidad de epoches
        :param batch_size: tamaño de la batch
        """

        for epoch in range(epochs):
            loss_history = []
            metric_history = []
            self.log(f'Epoch {epoch + 1}/{epochs}')
            for batch in range(0, train_data.shape[0], batch_size):
                batch_data = train_data[batch:batch+batch_size]
                batch_labels = train_labels[batch:batch+batch_size]

                self.params = self.get_params()   
                param_grads = [0 for _ in self.params]
                for i in range(batch_data.shape[0]):
                    grads = self._backward(batch_data[i, :, :], batch_labels[i])
                    for j in range(len(grads)):
                        param_grads[j] += grads[j]

                for i in range(len(param_grads)):
                    param_grads[i] /= batch_data.shape[0]
                self._update_params(param_grads)
                
                loss, metric = self.evaluate(batch_data, batch_labels)
                loss_history.append(loss)
                metric_history.append(metric)
                self.log(f'| Batch {batch//batch_size + 1}/{train_data.shape[0]//batch_size} - Loss: {loss:.4f} - Metric: {metric}')

            self.log(f'Epoch {epoch + 1}/{epochs} finished - Loss: {np.mean(loss_history):.4f} - Metric: {np.mean(metric_history)}')
            if validation_data is not None:
                loss, metric = self.evaluate(validation_data, validation_labels)
                self.log(f'Validation - Loss: {loss:.4f} - Metric: {metric}')

    def predict(self, test_data: np.ndarray, probs: bool=False) -> np.ndarray:
        """
        Realiza una predicción sobre el modelo de red neuronal

        :param test_data: matriz de entrada de prueba
        :param probs: indica si se debe devolver las probabilidades de cada clase o la clase ganadora
        :return: matriz de predicciones
        """
        preds = self._forward(test_data)
        return preds if probs else np.argmax(preds, axis=0)
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Realiza una evaluación sobre el modelo de red neuronal para un lote de datos de prueba

        :param test_data: matriz de entrada de prueba
        :param test_labels: matriz de etiquetas de prueba
        :return: valor de la pérdida y la métrica 
        """

        loss = 0
        metric = 0
        for i in range(test_data.shape[0]):
            pred = self.predict(test_data[i, :, :], probs=True)
            pred = np.array([pred])
            true = np.array([test_labels[i]])
            loss += self.loss(true, pred)
            metric += self.metrics(true, pred)

        loss /= test_data.shape[0]
        metric /= test_data.shape[0]
        return round(loss, 3), round(metric, 3)

    def log(self, text: str) -> None:
        """
        Escribe un mensaje en la consola

        :param text: mensaje a escribir
        """
        if self.verbose:
            print(text)