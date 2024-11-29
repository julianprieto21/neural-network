from models.models import NeuralNetwork
from training.optimizers import SGD
from training.loss import cross_entropy
from utils.metrics import accuracy
import models.layers as layers
from models.activations import ReLU, Softmax
import numpy as np
import random

np.random.seed(21)
random.seed(21)

if __name__ == '__main__':
    model = NeuralNetwork(
        layers=(
            layers.Conv2D(input_shape=(1, 28, 28), filters=4, filter_size=3, activation=ReLU(), padding='valid'),
            layers.MaxPool2D(pool_size=3),
            # layers.Conv2D(filters=64, filter_size=3, activation=ReLU(), padding='same'),
            # layers.MaxPool2D(pool_size=2, stride=2),
            layers.Flatten(),
            layers.Dense(neurons=32, activation=ReLU()),
            layers.Dense(neurons=10, activation=Softmax())
        ), 
        optimizer=SGD(learning_rate=0.001), 
        loss=cross_entropy, metrics=accuracy)
    
    model.compile()
    model.summary()
    # model._forward(np.random.rand(1, 28, 28))
    preds = model._forward(np.random.rand(1, 28, 28))
    labels = np.random.randint(0, 10, size=(1, 10))
    first_grads = preds - labels
    grads = model._backward(first_grads)

    # params = self.get_params()
    # params = self.optimizer(params, grads)