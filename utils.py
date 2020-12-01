import os
import pickle
from sklearn.model_selection import train_test_split

from config import DATA_DIR


IMG_DIM = 56


def load_data():
    file_path = os.path.join(DATA_DIR, 'train.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def split_dataset(x, y):
    return train_test_split(x, y, test_size=0.3, random_state=0)
