import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
import numpy as np


class QSABrain:
    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        raise NotImplementedError()

    def predict_batch_actions(self, s: np.ndarray, all_actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        raise NotImplementedError()


class SimpleMLPQSABrain(QSABrain):
    def __init__(self, state_dim: int, action_dim: int,
                 neurons_per_hidden_layer: int = 16,
                 hidden_layers_count: int = 2,
                 lr: float = 0.1):
        state_input = Input(shape=(state_dim,))
        action_input = Input(shape=(action_dim,))

        hidden = Concatenate()([state_input, action_input])
        for _ in range(hidden_layers_count):
            hidden = Dense(neurons_per_hidden_layer, activation=tanh)(hidden)
        output = Dense(1, activation=linear)(hidden)
        self.model = Model([state_input, action_input], output)
        self.model.compile(optimizer=Adam(lr=lr), loss=mean_squared_error)

    def predict_batch_actions(self, s: np.ndarray, all_actions: np.ndarray) -> np.ndarray:
        return self.model.predict([np.tile(s, (len(all_actions), 1)), all_actions])

    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        return self.model.predict([np.array([s]), np.array([a])])[0][0]

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        return self.model.train_on_batch([np.array([s]), np.array([a])], np.array([target]))
