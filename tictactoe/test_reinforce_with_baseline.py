import os

os.environ['TF_DISABLE_MKL'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.activations import softmax, linear, tanh, relu

from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.constraints import maxnorm
from modern_keras import reinforce
from tictactoe import TicTacToeAgainstRandomEnv
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    env = TicTacToeAgainstRandomEnv()

    actor_model = Sequential()
    actor_model.add(Flatten())
    actor_model.add(Dense(256, activation=tanh,
                          bias_regularizer=l2(),
                          kernel_regularizer=l2()))
    actor_model.add(Dense(256, activation=tanh,
                          bias_regularizer=l2(),
                          kernel_regularizer=l2()))
    actor_model.add(Dense(9, activation=softmax,
                          bias_regularizer=l2(),
                          kernel_regularizer=l2()))

    base_line_model = Sequential()
    base_line_model.add(Flatten())
    base_line_model.add(Dense(256, activation=tanh,
                              bias_regularizer=l2(),
                              kernel_regularizer=l2()))
    base_line_model.add(Dense(256, activation=tanh,
                              bias_regularizer=l2(),
                              kernel_regularizer=l2()))
    base_line_model.add(Dense(1, activation=linear,
                              bias_regularizer=l2(),
                              kernel_regularizer=l2()))

    reinforce(
        actor_model,
        base_line_model,
        env.deep_reset_func,
        env.deep_get_state,
        env.deep_is_terminal_func,
        env.deep_step_func,
        env.deep_get_available_actions,
        episodes_count=500000,
        max_steps_per_episode=10
    )

    # states = env.to_categorical(np.arange(num_states))
    # actions = env.deep_get_all_action_description()
    #
    # for s_idx, s in enumerate(states):
    #     print(f'{s} - policy : {actor_model.predict(np.expand_dims(s, 0))}')
    #     print(f'{s} - baseline : {base_line_model.predict(np.expand_dims(s, 0))}')
