import numpy as np
from training.optimizers import Optimizer
from .layers import Layer
from training.losses import Loss
import h5py

class NeuralNetwork:
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], optimizer: Optimizer, loss: Loss, metrics: any, verbose: bool=True) -> None:
        """
        Constructor de un modelo de red neuronal

        :param input_shape: forma de la entrada
        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso
        """
        self.input_shape = (None,) + input_shape
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.params = None
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
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], optimizer: Optimizer, loss: Loss, metrics: any, verbose: bool=True) -> None:
        """
        Constructor de un modelo de red neuronal

        :param input_shape: forma de la entrada
        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso
        """
        super().__init__(input_shape, layers, optimizer, loss, metrics, verbose)
    
    def summary(self) -> None:
        """
        Muestra un resumen del modelo
        """
        parameters = sum(arr.size for arr in self.get_params())
        print(f'Modelo de red neuronal con {len(self.layers) + 1} capas:')
        print(f'- input: Entrada - {self.input_shape}')
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape} - {layer.weights.size + layer.bias.size} params')
                print(f'    Pesos: ', layer.weights.shape)
                print(f'    Bias: ', layer.bias.shape)
            else:
                print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape}')
        print(f'Optimizador: {self.optimizer.__class__.__name__} - Learning rate: {self.optimizer.learning_rate}')
        print(f'Función de pérdida: {self.loss.__class__.__name__}')        
        print(f'Función de cálculo de métricas: {self.metrics.__name__}')
        print(f'Cantidad de parámetros: {parameters}')

    def compile(self) -> None:
        """
        Compila el modelo.
        """
        self.layers[0].compile(self.input_shape)
        output_shape = self.layers[0].output_shape
        for layer in self.layers[1:]:
            layer.compile(output_shape)
            output_shape = layer.output_shape
        self.params = self.get_params()

    def get_params(self) -> list[np.ndarray]:
        """
        Obtiene los parámetros del modelo

        :return params: lista de parámetros
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                params.append(layer.weights)
                params.append(layer.bias)
        return params

    def _get_parameter_grads(self) -> list[np.ndarray]:
        """
        Obtiene los gradientes de los parámetros del modelo

        :return grads: lista de gradientes
        """
        grads = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
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
        grad_output = self.loss.backward(y, preds)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return self._get_parameter_grads()
        
    def _update_params(self, param_grads: list[np.ndarray]) -> None:
        """
        Actualiza los parámetros del modelo

        :param param_grads: gradientes de los parámetros	
        """
        self.params = self.optimizer(self.params, param_grads)
        i = 0
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
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

        global_loss = []
        global_metric = []
        self.log("Entreando el modelo...")
        for epoch in range(epochs):
            for batch in range(0, train_data.shape[0], batch_size):
                loss_history = []
                metric_history = []
                x_batch = train_data[batch:batch+batch_size]
                y_batch = train_labels[batch:batch+batch_size]

                loss, metric = self.evaluate(x_batch, y_batch)
                loss_history.append(loss)
                metric_history.append(metric)

                grads = self._backward(x_batch, y_batch)
                param_grads = [
                    np.where(abs(grad / x_batch.shape[0]) < 1e-07, 0, grad / x_batch.shape[0]) 
                    for grad in grads
                ]
                self._update_params(param_grads)
                
            self.log(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(loss_history):.4f} - Metric: {np.mean(metric_history)}')
            global_loss.append(np.mean(loss_history))
            global_metric.append(np.mean(metric_history))
            if validation_data is not None:
                loss, metric = self.evaluate(validation_data, validation_labels)
                self.log(f'Validation - Loss: {loss:.4f} - Metric: {metric}')
        
        return global_loss, global_metric

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
        pred = self.predict(test_data, probs=True)
        loss = self.loss(test_labels, pred)
        metric = self.metrics(test_labels, pred)

        return loss, metric
    
    def save_parameters(self, filename: str='parameters') -> None:
        """
        Guarda los parámetros del modelo en un archivo .h5
        """
        with h5py.File(f'parameters/{filename}.h5', 'w') as file:
            for i, param in enumerate(self.params):
                file.create_dataset(f'param_{i}', data=param)

    def load_parameters(self, filename: str='parameters') -> None:
        """
        Carga los parámetros del modelo desde un archivo .h5
        """
        with h5py.File(f'parameters/{filename}.h5', 'r') as file:
            for i, param in enumerate(self.params):
                param[...] = file[f'param_{i}'][...]
                self.params[i] = param
