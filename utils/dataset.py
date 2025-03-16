import numpy as np
import os
from PIL import Image
from utils.helpers import one_hot_encoder


def train_test_split(x, y, img_size, test_size=0.2, validation_size=0):
    """
    Divide los datos en train, test y validación.

    :param x: matriz de entrada
    :param y: matriz de salida
    :param test_size: tamaño de la parte de prueba
    :param validation_size: tamaño de la parte de validación
    :return: lista de los subconjuntos de entrenamiento, prueba y validación (x_train, y_train, x_test, y_test, x_val, y_val)
    """

    channels, height, width = img_size

    x = x.astype(np.float32) / 255.0 # Normalizar a rango [0, 1]
    y = y.astype(np.float32)

    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    test_size = int(len(x) * test_size)
    train_size = len(x) - test_size
    val_size = int(train_size * validation_size)

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[-test_size:]
    y_test = y[-test_size:]
    x_val = np.array([])
    y_val = np.array([])
    
    if validation_size > 0:
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]

    x_train = x_train.reshape(x_train.shape[0], channels, height, width)
    x_test = x_test.reshape(x_test.shape[0], channels, height, width)
    if validation_size > 0:
        x_val = x_val.reshape(x_val.shape[0], channels, height, width)
    
    y_train = y_train.reshape(-1, 1).astype(int).flatten()
    y_train = one_hot_encoder(y_train)
    y_test = y_test.reshape(-1, 1).astype(int).flatten()
    y_test = one_hot_encoder(y_test)
    if validation_size > 0:
        y_val = y_val.reshape(-1, 1).astype(int).flatten()
        y_val = one_hot_encoder(y_val)

    if x_val.shape[0] > 0 and y_val.shape[0] > 0:
        return x_train, y_train, x_test, y_test, x_val, y_val
    else:
        return x_train, y_train, x_test, y_test
    

def load_mnist(validation_size=0, data_dir='data', img_size=(1, 28, 28)):
    """
    Carga los datos de MNIST.

    :param validation_size: tamaño de la parte de validación
    :param data_dir: directorio donde se encuentran los datos
    :param img_size: tamaño de la imagen
    :return: lista de los subconjuntos de entrenamiento, prueba y validación (x_train, y_train, x_test, y_test, x_val, y_val)
    """
    train_data = np.genfromtxt(f'{data_dir}/mnist_train.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt(f'{data_dir}/mnist_test.csv', delimiter=',', skip_header=1)
    data = np.concatenate((train_data, test_data), axis=0)
    x = data[:, 1:]
    y = data[:, 0]

    return train_test_split(x, y, img_size=img_size, test_size=0.2, validation_size=validation_size)


def load_cats_dogs(validation_size=0, data_dir='data/Animal Images', img_size=(1, 150, 150)):
    """
    Carga los datos de Cats vs Dogs.

    :param validation_size: tamaño de la parte de validación
    :param data_dir: directorio donde se encuentran los datos
    :param img_size: tamaño de la imagen
    :return: lista de los subconjuntos de entrenamiento, prueba y validación (x_train, y_train, x_test, y_test, x_val, y_val)
    """
    x = []
    y = []
    class_names = os.listdir(data_dir)  # Cada carpeta es una clase
    class_map = {name: idx for idx, name in enumerate(class_names)}  # Asignar índice a cada clase
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    # Cargar imagen y redimensionar
                    img = Image.open(img_path)
                    img_array = np.array(img).reshape(img_size)
                    x.append(img_array)
                    y.append(class_map[class_name])
                except Exception as e:
                    print(f"Error al cargar {img_path}: {e}")
    return train_test_split(np.array(x), np.array(y), img_size=img_size, test_size=0.2, validation_size=validation_size)