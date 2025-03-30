from typing import Callable
import numpy as np
from training.optimizers import Optimizer
from .layers import Layer, Conv2D, Pool2D
from training.losses import Loss
import h5py
from PIL import Image
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], optimizer: Optimizer, loss: Loss, metrics: Callable[[np.ndarray, np.ndarray], float], verbose: bool=False) -> None:
        """
        Constructor de un modelo de red neuronal.

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

    def summary(self) -> None:
        """
        Muestra un resumen del modelo.
        """
        parameters = sum(param.size for param in self.get_params())
        print(f'{__class__.__name__} con {len(self.layers) + 1} capas:')
        print(f'- input: Entrada - {self.input_shape}')
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                print(f'- {layer.name}: {layer.__class__.__name__} - {layer.output_shape} - {np.sum([param.size for param in layer.get_params()])} params')
                print(f'    Pesos: ', layer.weights.shape)
                if hasattr(layer, 'recurrent_weights'):
                    print(f'    Pesos Recurrentes: ', layer.recurrent_weights.shape)
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia adelante.

        :param x: tensor de entrada
        :return: tensor de salida
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad_x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Realiza una propagación hacia atrás.

        :param grad_x: gradientes de la propagación hacia adelante
        :param y: tensor de entrada
        :return: gradientes de los parámetros
        """      

        preds = self(grad_x)
        grad_output = self.loss.backward(y, preds)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return self.get_grads()
    
    def get_params(self) -> list[np.ndarray]:
        """
        Obtiene los parámetros del modelo.

        :return: lista de parámetros
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_grads(self) -> list[np.ndarray]:
        """
        Obtiene los gradientes de los parámetros del modelo.

        :return: lista de gradientes
        """
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads
    
    def update_params(self, param_grads: list[np.ndarray]) -> None:
        """
        Actualiza los parámetros del modelo.

        :param: gradientes de los parámetros	
        """
        self.params = self.optimizer(self.params, param_grads)
        i = 0
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                layer.weights = self.params[i]
                if hasattr(layer, 'recurrent_weights'):
                    layer.recurrect_weights = self.params[i+1]
                    layer.bias = self.params[i+2]
                    i += 3
                else:
                    layer.bias = self.params[i+1]
                    i += 2

    def log(self, text: str) -> None:
        """
        Escribe un mensaje en la consola.

        :param text: mensaje a escribir
        """
        if self.verbose:
            print(text)

    def save_parameters(self, filename: str='parameters') -> None:
        """
        Guarda los parámetros del modelo en un archivo formato h5.

        :param filename: nombre del archivo
        """
        with h5py.File(f'parameters/{filename}.h5', 'w') as file:
            for i, param in enumerate(self.params):
                file.create_dataset(f'param_{i}', data=param)

    def load_parameters(self, filename: str='parameters') -> None:
        """
        Carga los parámetros del modelo desde un archivo formato h5.

        :param filename: nombre del archivo
        """
        with h5py.File(f'parameters/{filename}.h5', 'r') as file:
            for i, param in enumerate(self.params):
                param[...] = file[f'param_{i}'][...]
                self.params[i] = param

class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], optimizer: Optimizer, loss: Loss, metrics: Callable[[np.ndarray, np.ndarray], float], verbose: bool=False) -> None:
        """
        Constructor de un modelo de red neuronal.

        :param input_shape: forma de la entrada
        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso y logs
        """
        super().__init__(input_shape, layers, optimizer, loss, metrics, verbose)

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray]=(None, None), epochs: int=10, batch_size: int=32) -> None:
        """
        Entrena el modelo de red neuronal.

        :param train_data: matriz de entrada de entrenamiento
        :param train_labels: matriz de etiquetas de entrenamiento
        :param validation_data: matriz de entrada de validación
        :param epochs: cantidad de epoches
        :param batch_size: tamaño de la batch
        """
        metric_name = self.metrics.__name__

        train_global_loss = []
        train_global_metric = []
        valid_global_loss = []
        valid_global_metric = []
        self.log("Entrenando el modelo...")
        for epoch in range(epochs):
            for batch in range(0, train_data.shape[0], batch_size):
                loss_history = []
                metric_history = []
                x_batch = train_data[batch:batch+batch_size]
                y_batch = train_labels[batch:batch+batch_size]

                loss, metric = self.evaluate(x_batch, y_batch)
                loss_history.append(loss)
                metric_history.append(metric)

                grads = self.backward(x_batch, y_batch)
                param_grads = [
                    np.where(abs(grad / x_batch.shape[0]) < 1e-07, 0, grad / x_batch.shape[0]) 
                    for grad in grads
                ]
                self.update_params(param_grads)
                
            self.log(f'Epoch {epoch + 1}/{epochs} - Loss: {np.mean(loss_history):.4f} - {metric_name}: {np.mean(metric_history)}')
            train_global_loss.append(np.mean(loss_history))
            train_global_metric.append(np.mean(metric_history))
            if validation_data[0] is not None:
                loss, metric = self.evaluate(validation_data[0], validation_data[1])
                self.log(f'Validation - Loss: {loss:.4f} - {metric_name}: {metric}')
                valid_global_loss.append(loss)
                valid_global_metric.append(metric)
        
        return {
            'train': {'loss': train_global_loss, metric_name: train_global_metric},
            'valid': {'loss': valid_global_loss, metric_name: valid_global_metric}
        }

    def predict(self, test_data: np.ndarray, probs: bool=True) -> np.ndarray:
        """
        Realiza una predicción sobre el modelo de red neuronal.

        :param test_data: matriz de entrada de prueba
        :param probs: indica si se debe devolver las probabilidades de cada clase o la clase ganadora
        :return: matriz de predicciones
        """
        preds = self(test_data)
        return preds if probs else np.argmax(preds, axis=0)
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Realiza una evaluación sobre el modelo de red neuronal para un lote de datos de prueba.

        :param test_data: matriz de entrada de prueba
        :param test_labels: matriz de etiquetas de prueba
        :return: valor de la pérdida y la métrica 
        """
        pred = self.predict(test_data, probs=True)
        loss = self.loss(test_labels, pred)
        metric = self.metrics(test_labels, pred)

        return loss, metric

    def get_gradcam(self, data: np.ndarray, alpha: float=1.0) -> np.ndarray:
        """
        EXPERIMENTAL
        Obtiene el Grad-CAM de una imagen.

        :param data: matriz de entrada
        :param alpha: factor de regularización
        :return: matriz de gradiente CAM
        """
        x = data
        if len(x.shape) == 3: x = np.expand_dims(x, axis=0)
        channels, height, width = x.shape[1:]
        pred = self(x)
        num_classes = pred.shape[1]
        class_idx = np.argmax(pred, axis=1)

        layers = self.layers
        last_conv_layer = [layer for layer in layers if isinstance(layer, Conv2D)][-1]
        for layer in layers:
            x = layer(x)
            if layer == last_conv_layer: break
        feature_maps = x

        grad_output = np.zeros(num_classes)
        grad_output[class_idx] = 1

        dense_weights_1 = self.layers[-1].weights
        dense_weights_2 = self.layers[-2].weights
        grad_dense_2 = np.dot(grad_output, dense_weights_1.T)
        grad_dense_1 = np.dot(grad_dense_2, dense_weights_2.T)


        maxpool = [layer for layer in layers if issubclass(layer, Pool2D)][-1].copy()
        print(maxpool.output_shape)
        grad_pooled = grad_dense_1.reshape(maxpool.output_shape[1], 1, maxpool.output_shape[2], maxpool.output_shape[3])
        feature_maps = feature_maps.transpose(1, 0, 2, 3)

        maxpool.forward_data = feature_maps
        grad_feature_maps = maxpool.backward(grad_pooled)
        grad_feature_maps = np.mean(grad_feature_maps, axis=(1, 2, 3))

        heatmap = feature_maps.transpose(1, 0, 2, 3)[0].T @ grad_feature_maps
        heatmap = heatmap.T
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        jet = plt.get_cmap('viridis')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
        jet_heatmap = jet_heatmap.resize((height, width), resample=Image.NEAREST)
        jet_heatmap = np.array(jet_heatmap, dtype=np.float32) / 255.0

        img = data[0].reshape(height, width, channels) / 255
        gradcam = jet_heatmap * alpha + img
        return gradcam

# EXPERIMENTAL
class RecurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], optimizer: Optimizer, loss: Loss, metrics: Callable[[np.ndarray, np.ndarray], float], verbose: bool=False) -> None:
        """
        Constructor de un modelo de red neuronal.

        :param input_shape: forma de la entrada
        :param layers: capas de la red
        :param optimizer: método de optimización
        :param loss: función de pérdida
        :param metrics: función de cálculo de métricas
        :param verbose: indica si se debe mostrar mensajes de progreso y logs
        """
        super().__init__(input_shape, layers, optimizer, loss, metrics, verbose)