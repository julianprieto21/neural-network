from models.models import NeuralNetwork
from training.optimizers import SGD
from training.loss import cross_entropy
from utils.metrics import accuracy
import models.layers as layers
from models.activations import ReLU, Softmax
import numpy as np

if __name__ == '__main__':
    model = NeuralNetwork(
        layers=(
            layers.Conv2D(input_shape=(1, 28, 28), filters=4, filter_size=3, activation=ReLU(), padding='same'),
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
    # model.summary()
    model._forward(np.random.rand(1, 28, 28))
    # logits = model.forward(np.random.rand(1, 28, 28))
    # pred = np.argmax(logits, axis=0)
    # print(pred)