import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from algorithms import deep_q_learning_with_experience_replay_control
from brains import SimpleMLPQSABrain
from tictactoe import TicTacToeAgainstRandomEnv
import numpy as np

if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    brain = SimpleMLPQSABrain(27, 9, lr=0.0001)
    env = TicTacToeAgainstRandomEnv()

    deep_q_learning_with_experience_replay_control(
        env.deep_reset_func,
        env.deep_get_state,
        env.deep_get_all_available_action_description,
        env.deep_is_terminal_func,
        env.deep_step_func,
        brain,
        epsilon=0.5,
        episodes_count=10000,
    )

    env = TicTacToeAgainstRandomEnv()
    env.deep_reset_func()
    print(env.board)

    ss = env.deep_get_state()
    aa = env.deep_get_all_available_action_description()
    print(brain.predict_batch_actions(ss, aa))


