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
        self.data_grads = None
        self.compile()
    
    def compile(self) -> None:
        """
        Compila el modelo.
        """
        raise NotImplementedError

    def log(self, text: str) -> None:
        """
        Escribe un mensaje en la consola

        :param text: mensaje a escribir
        """
        if self.verbose:
            print(text)
class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss: any, metrics: any, verbose: bool=True) -> None:
        """
        Constructor de un modelo de red neuronal

        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso
        """
        super().__init__(layers, optimizer, loss, metrics, verbose)
    
    def summary(self) -> None:
        """
        Muestra un resumen del modelo
        """
        parameters = sum(arr.size for arr in self.get_params())
        print(f'Modelo de red neuronal con {len(self.layers) + 1} capas:')
        print(f'- input: Entrada - {self.layers[0].input_shape}')
        for layer in self.layers:
            if layer.weights is not None: 
                print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape} - {layer.weights.size + layer.bias.size} params')
                print(f'    Pesos: ', layer.weights.shape)
                print(f'    Bias: ', layer.bias.shape)
            else:
                print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape}')
        print(f'Optimizador: {self.optimizer.__class__.__name__} - Learning rate: {self.optimizer.learning_rate}')
        print(f'Función de pérdida: {self.loss.__name__}')        
        print(f'Función de cálculo de métricas: {self.metrics.__name__}')
        print(f'Cantidad de parámetros: {parameters}')

    def compile(self) -> None:
        """
        Compila el modelo.
        """
        output_shape = self.layers[0].output_shape
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

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante

        :param x: tensor de entrada
        :return y: tensor de salida
        """
        for layer in self.layers:
            data = layer(data)
        return data
    
    def _backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás

        :param grad_output: gradientes de la propagación hacia adelante
        :param data: tensor de entrada
        :return param_grads: gradientes de los parámetros
        """      

        preds = self(x)
        grad_output = preds - y 
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        self.data_grads = grad_output
        return self._get_grads()
        
    def _update_params(self, param_grads: list[np.ndarray]) -> None:
        """
        Actualiza los parámetros del modelo

        :param param_grads: gradientes de los parámetros	
        """
        self.params = self.optimizer(self.params, param_grads)
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
                x_batch = train_data[batch:batch+batch_size]
                y_batch = train_labels[batch:batch+batch_size]

                loss, metric = self.evaluate(x_batch, y_batch)
                loss_history.append(loss)
                metric_history.append(metric)

                param_grads = [np.zeros_like(param) for param in self.params]
                for i in range(x_batch.shape[0]):
                    grads = self._backward(x_batch[i, :, :], y_batch[i])
                    for j, grad in enumerate(grads):
                        param_grads[j] += grad

                param_grads = [
                    np.where(abs(grad / x_batch.shape[0]) < 1e-07, 0, grad / x_batch.shape[0]) 
                    for grad in param_grads
                ]
                self._update_params(param_grads)
                
                self.log(f'| Batch {batch//batch_size + 1}/{train_data.shape[0]//batch_size} - Loss: {np.mean(loss_history):.4f} - Metric: {np.mean(metric_history)}')

            if validation_data is not None:
                loss, metric = self.evaluate(validation_data, validation_labels)
                self.log(f'Validation - Loss: {loss:.4f} - Metric: {metric}')

    def predict(self, test_data: np.ndarray, probs: bool=True) -> np.ndarray:
        """
        Realiza una predicción sobre el modelo de red neuronal

        :param test_data: matriz de entrada de prueba
        :param probs: indica si se debe devolver las probabilidades de cada clase o la clase ganadora
        :return: matriz de predicciones
        """
        preds = self(test_data)
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
            true = np.array([test_labels[i]])
            loss += self.loss(true, pred)
            metric += self.metrics(true, pred)

        loss /= test_data.shape[0]
        metric /= test_data.shape[0]
        return round(loss, 3), round(metric, 3)
