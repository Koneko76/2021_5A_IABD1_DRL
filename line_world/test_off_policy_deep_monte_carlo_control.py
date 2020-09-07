import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from algorithms import off_policy_deep_monte_carlo_control
from brains import SimpleMLPQSABrain
from line_world import S, A, LineWorldEnv, num_states
import numpy as np

if __name__ == "__main__":
    brain = SimpleMLPQSABrain(len(S), len(A))
    env = LineWorldEnv()

    off_policy_deep_monte_carlo_control(
        env.deep_reset,
        env.deep_get_state,
        env.deep_get_all_action_description,
        env.deep_is_terminal,
        env.deep_step,
        brain,
        epsilon=0.2,
        episodes_count=500,
    )

    states = env.to_categorical(np.arange(num_states))
    actions = env.deep_get_all_action_description()

    for s_idx, s in enumerate(states):
        print(f'{s}, {brain.predict_batch_actions(s, actions)}')

