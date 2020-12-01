import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from functools import reduce

from config import DATA_DIR


IMG_DIM = 56


def reshape_features(x, new_shape):
    assert reduce(lambda a, b: a * b, new_shape) == x.shape[1]
    return np.reshape(x, (x.shape[0], *new_shape))


def load_data(dir=DATA_DIR):
    file_path = os.path.join(dir, 'train.pkl')
    with open(file_path, 'rb') as file:
        x, y = pickle.load(file)
    x = reshape_features(x, (IMG_DIM, IMG_DIM))
    return x, y


def split_dataset(x, y):
    return train_test_split(x, y, test_size=0.3, random_state=0)
