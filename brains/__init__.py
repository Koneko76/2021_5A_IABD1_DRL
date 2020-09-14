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


class VSBrain:
    def predict(self, s: np.ndarray) -> float:
        raise NotImplementedError()

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train_single(self, s: np.ndarray, target: float) -> float:
        raise NotImplementedError()

    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError()


class SimpleMLPVSBrain(VSBrain):
    def __init__(self, state_dim: int,
                 neurons_per_hidden_layer: int = 16,
                 hidden_layers_count: int = 2,
                 lr: float = 0.1):
        state_input = Input(shape=(state_dim,))

        hidden = state_input
        for _ in range(hidden_layers_count):
            hidden = Dense(neurons_per_hidden_layer, activation=tanh)(hidden)
        output = Dense(1, activation=linear)(hidden)
        self.model = Model(state_input, output)
        self.model.compile(optimizer=Adam(lr=lr), loss=mean_squared_error)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self.model.predict(states)

    def predict(self, s: np.ndarray) -> float:
        return self.model.predict(np.array([s]))[0][0]

    def train_single(self, s: np.ndarray, target: float) -> float:
        return self.model.train_on_batch(np.array([s]), np.array([target]))

    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        return self.model.train_on_batch(states, targets)


def vanilla_softmax_with_mask(signals, masks):
    exp_signals = keras.backend.exp(signals) * masks
    return exp_signals / keras.backend.sum(exp_signals, axis=-1, keepdims=True)


def my_softmax_with_mask(signals_and_masks):
    signals = signals_and_masks[0]
    masks = signals_and_masks[1]
    positive_signals = signals - keras.backend.min(signals, axis=-1, keepdims=True)
    masked_positive_signals = positive_signals * masks
    negative_masked_positive_signals = masked_positive_signals - keras.backend.max(masked_positive_signals,
                                                                                   axis=-1,
                                                                                   keepdims=True)
    exp_negative_masked_positive_signals = keras.backend.exp(negative_masked_positive_signals)
    masked_exp_negative_masked_positive_signals = exp_negative_masked_positive_signals * masks
    sum_masked_exp_negative_masked_positive_signals = keras.backend.sum(masked_exp_negative_masked_positive_signals,
                                                                        axis=-1, keepdims=True)
    return masked_exp_negative_masked_positive_signals / sum_masked_exp_negative_masked_positive_signals


# if __name__ == "__main__":
#     import tensorflow as tf
#
#     print(my_softmax_with_mask(tf.constant([[2.0, 2.0, 2000, 2.0]]),
#                                tf.constant([[0.0, 1.0, 0.0, 1.0]])))
#     print(vanilla_softmax_with_mask(tf.constant([[2.0, 2.0, 2000, 2.0]]),
#                                     tf.constant([[0.0, 1.0, 0.0, 1.0]])))

class PiSWithMaskBrain:
    def predict(self, s: np.ndarray, m: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def predict_batch(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train_single(self, s: np.ndarray, m: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError()

    def train_batch(self, states: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError()


class SimpleMLPPiSWithMaskBrainReinforce(PiSWithMaskBrain):
    @staticmethod
    def reinforce_loss(y_true, y_pred):
        return keras.backend.sum(-y_true * keras.backend.log(keras.backend.clip(y_pred, 1e-8, 1.0 - 1e-8)))

    def __init__(self, state_dim: int, max_action_count: int,
                 neurons_per_hidden_layer: int = 16,
                 hidden_layers_count: int = 2,
                 lr: float = 0.1):
        state_input = Input(shape=(state_dim,))
        mask_input = Input(shape=(max_action_count,))

        hidden = state_input
        for _ in range(hidden_layers_count):
            hidden = Dense(neurons_per_hidden_layer, activation=tanh)(hidden)
        signals = Dense(max_action_count, activation=linear)(hidden)
        output = Activation(my_softmax_with_mask)((signals, mask_input))

        self.model = Model([state_input, mask_input], output)
        self.model.compile(optimizer=Adam(lr=lr), loss=self.reinforce_loss)

    def predict_batch(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self.model.predict([states, masks])

    def predict(self, s: np.ndarray, m: np.ndarray) -> np.ndarray:
        return self.model.predict([np.array([s]), np.array([m])])[0]

    def train_single(self, s: np.ndarray, m: np.ndarray, target: np.ndarray) -> float:
        return self.model.train_on_batch([np.array([s]), np.array([m])], np.array([target]))

    def train_batch(self, states: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> float:
        return self.model.train_on_batch([states, masks], targets)


class PiSABrain:
    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        raise NotImplementedError()

    def predict_batch_actions(self, s: np.ndarray, all_actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        raise NotImplementedError()

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError()


class SimpleMLPPiSABrainReinforce(PiSABrain):
    @staticmethod
    def reinforce_loss(y_true, y_pred):
        return keras.backend.sum(-y_true * keras.backend.log(keras.backend.clip(y_pred, 1e-8, 1.0 - 1e-8)))

    def __init__(self, state_dim: int, action_dim: int,
                 neurons_per_hidden_layer: int = 16,
                 hidden_layers_count: int = 2,
                 lr: float = 0.1):
        state_input = Input(shape=(state_dim,))
        action_input = Input(shape=(action_dim,))

        hidden = Concatenate()([state_input, action_input])
        for _ in range(hidden_layers_count):
            hidden = Dense(neurons_per_hidden_layer, activation=tanh)(hidden)
        output = Dense(1, activation=sigmoid)(hidden)
        self.model = Model([state_input, action_input], output)
        self.model.compile(optimizer=Adam(lr=lr), loss=self.reinforce_loss)

    def predict_batch_actions(self, s: np.ndarray, all_actions: np.ndarray) -> np.ndarray:
        return self.model.predict([np.tile(s, (len(all_actions), 1)), all_actions])

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.model.predict([states, actions])

    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        return self.model.predict([np.array([s]), np.array([a])])[0][0]

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        return self.model.train_on_batch([np.array([s]), np.array([a])], np.array([target]))

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        return self.model.train_on_batch([states, actions], targets)


class QSABrain:
    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        raise NotImplementedError()

    def predict_batch_actions(self, s: np.ndarray, all_actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        raise NotImplementedError()

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
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

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.model.predict([states, actions])

    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        return self.model.predict([np.array([s]), np.array([a])])[0][0]

    def train_single(self, s: np.ndarray, a: np.ndarray, target: float) -> float:
        return self.model.train_on_batch([np.array([s]), np.array([a])], np.array([target]))

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        return self.model.train_on_batch([states, actions], targets)
