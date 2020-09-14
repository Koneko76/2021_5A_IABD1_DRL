import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from algorithms import reinforce_with_mask_with_baseline
from brains import SimpleMLPPiSWithMaskBrainReinforce, SimpleMLPVSBrain
from tictactoe import TicTacToeAgainstRandomEnv
import numpy as np

if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    pi_brain = SimpleMLPPiSWithMaskBrainReinforce(27, 9, lr=0.0001, hidden_layers_count=4,
                                                  neurons_per_hidden_layer=64)
    v_brain = SimpleMLPVSBrain(27, lr=0.001, hidden_layers_count=4,
                               neurons_per_hidden_layer=64)
    env = TicTacToeAgainstRandomEnv()

    reinforce_with_mask_with_baseline(
        env.deep_reset_func,
        env.deep_get_state,
        env.deep_get_action_mask,
        env.deep_is_terminal_func,
        env.deep_step_with_mask_func,
        pi_brain,
        v_brain,
        episodes_count=50000,
    )

    env = TicTacToeAgainstRandomEnv()
    env.deep_reset_func()
    print(env.board)

    s = env.deep_get_state()
    m = env.deep_get_action_mask()
    print(pi_brain.predict(s, m))
    print(v_brain.predict(s))
