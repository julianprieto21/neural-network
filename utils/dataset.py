import numpy as np
from utils.helpers import one_hot_encoder

def load_dataset(dataset_name):
    train_data = np.genfromtxt(dataset_name + '_train.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt(dataset_name + '_test.csv', delimiter=',', skip_header=1)

    return train_data, test_data

def preprocess_dataset(train_data, test_data, val_size=0.2):
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    
    val_size = int(len(train_data) * val_size)
    train_size = len(train_data) - val_size

    train_data = train_data[:train_size]
    val_data = train_data[-val_size:]

    train_data = train_data[np.random.permutation(len(train_data))]
    val_data = val_data[np.random.permutation(len(val_data))]
    
    x_train = train_data[:, 1:]
    x_val = val_data[:, 1:]
    x_test = test_data[:, 1:]

    y_train = train_data[:, 0]
    y_val = val_data[:, 0]
    y_test = test_data[:, 0]


    x_train = x_train.reshape(x_train.shape[0], -1, 28, 28)
    x_val = x_val.reshape(x_val.shape[0], -1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], -1, 28, 28)

    y_train = y_train.reshape(-1, 1).astype(int).flatten()
    y_val = y_val.reshape(-1, 1).astype(int).flatten()
    y_test = y_test.reshape(-1, 1).astype(int).flatten()

    y_train = one_hot_encoder(y_train)
    y_val = one_hot_encoder(y_val)
    y_test = one_hot_encoder(y_test)

    print(
    f"""
    Train data: {x_train.shape}
    Valid data: {x_val.shape}
    Test data: {x_test.shape}

    Train labels: {y_train.shape}
    Valid labels: {y_val.shape}
    Test labels: {y_test.shape}
    """
    )

    return x_train, y_train, x_val, y_val, x_test, y_test

