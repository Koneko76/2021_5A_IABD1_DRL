import os

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from algorithms import episodic_semi_gradient_sarsa_control
from brains import SimpleMLPQSABrain
from grid_world import S, A, GridWorldEnv, num_states
import numpy as np

if __name__ == "__main__":
    brain = SimpleMLPQSABrain(len(S), len(A), lr=0.0001,
                              hidden_layers_count=3,
                              neurons_per_hidden_layer=64)
    env = GridWorldEnv()

    episodic_semi_gradient_sarsa_control(
        env.deep_reset,
        env.deep_get_state,
        env.deep_get_all_action_description,
        env.deep_is_terminal,
        env.deep_step,
        brain,
        start_epsilon=0.5,
        episodes_count=500,
        max_steps_per_episode=30
    )

    states = env.to_categorical(np.arange(num_states))
    actions = env.deep_get_all_action_description()

    for s_idx, s in enumerate(states):
        print(f'{s}, {brain.predict_batch_actions(s, actions)}')

