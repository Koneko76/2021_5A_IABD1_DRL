import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from algorithms import reinforce_with_mask_without_baseline
from brains import SimpleMLPPiSWithMaskBrainReinforce
from line_world import S, A, LineWorldEnv, num_states
import numpy as np

if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    brain = SimpleMLPPiSWithMaskBrainReinforce(len(S), len(A), lr=0.0001)
    env = LineWorldEnv()

    reinforce_with_mask_without_baseline(
        env.deep_reset,
        env.deep_get_state,
        env.deep_get_action_mask,
        env.deep_is_terminal,
        env.deep_step_with_mask,
        brain,
        episodes_count=5000,
    )

    states = env.to_categorical(np.arange(num_states))
    mask = env.deep_get_action_mask()

    for s_idx, s in enumerate(states):
        print(f'{s}, {brain.predict(s, mask)}')

