import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import softmax, relu, linear, tanh

from modern_keras import reinforce
from line_world import LineWorldEnv, A, num_states
import numpy as np

if __name__ == "__main__":

    # tf.compat.v1.disable_eager_execution()
    np.set_printoptions(precision=3, suppress=True)
    env = LineWorldEnv()

    actor_model = Sequential()
    actor_model.add(Dense(64, activation=relu))
    actor_model.add(Dense(len(A), activation=softmax))

    base_line_model = Sequential()
    base_line_model.add(Dense(64, activation=tanh))
    base_line_model.add(Dense(1, activation=linear))

    reinforce(
        actor_model,
        base_line_model,
        env.deep_reset,
        env.deep_get_state,
        env.deep_is_terminal,
        env.deep_step,
        env.deep_get_available_actions,
        episodes_count=500,
    )

    states = env.to_categorical(np.arange(num_states))
    actions = env.deep_get_all_action_description()

    for s_idx, s in enumerate(states):
        print(f'{s} - policy : {actor_model.predict(np.expand_dims(s, 0))}')
        print(f'{s} - baseline : {base_line_model.predict(np.expand_dims(s, 0))}')
