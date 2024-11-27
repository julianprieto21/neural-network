from models.cnn import ConvolutionalNetwork
from training.optimizers import SGD
from training.loss import cross_entropy
from utils.metrics import accuracy
import models.layers as layers

if __name__ == '__main__':
    # model = ConvolutionalNetwork(
    #     input_shape=(1, 28, 28), 
    #     num_classes=10, 
    #     layers=(
    #        layers.Flatten(input_shape=(28, 28, 1)) 
    #     ), 
    #     optimizer=SGD(learning_rate=0.001), 
    #     loss=cross_entropy, metrics=accuracy)
    layer = layers.Dense(input_shape=(784,), activation=1, neurons=128)
    print(layer.output_shape)