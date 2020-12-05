import tensorflow_addons as tfa
import numpy as np

from tensorflow import keras

from utils import reshape_features, IMG_DIM, NUM_CLASSES
from config import MODEL_PATH


def predict(input_data):
    input_data = reshape_features(input_data, (IMG_DIM, IMG_DIM))[..., np.newaxis]
    model = keras.models.load_model(MODEL_PATH,
                                    custom_objects={'F1Score': tfa.metrics.F1Score(NUM_CLASSES, average='macro')})
    return np.argmax(model.predict(input_data), axis=1)