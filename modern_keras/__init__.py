from typing import Callable, Tuple
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def reinforce(pi_model: Model,
              base_line_model: Model,
              deep_reset_func: Callable[[], None],
              deep_get_state: Callable[[], np.ndarray],
              deep_is_terminal_func: Callable[[], bool],
              deep_step_func: Callable[[int], Tuple[float, bool]],
              deep_get_available_actions: Callable[[], np.ndarray],
              episodes_count=500,
              max_steps_per_episode: int = 10,
              gamma: float = 0.99, ):
    s_train = []
    a_train = []
    r_train = []
    g_train = []

    policy_optimizer = Adam(lr=0.0001)
    baseline_optimizer = Adam(lr=0.0001)
    rolling_avg = [0.0]

    def act() -> (np.ndarray, int):
        s_chosen = deep_get_state()
        s_batch_chosen = np.expand_dims(s_chosen, 0)
        available_actions = deep_get_available_actions()

        pi_chosen = pi_model.predict(s_batch_chosen)[0][available_actions]
        sum_pi_chosen = np.sum(pi_chosen)
        if np.isnan(sum_pi_chosen):
            exit(42)
        if sum_pi_chosen > 0.0:
            pi_chosen = pi_chosen / sum_pi_chosen
        else:
            pi_chosen = np.ones_like(pi_chosen) / available_actions.shape[0]
        a_chosen = np.random.choice(available_actions, p=pi_chosen)
        return s_chosen, a_chosen

    for episode_id in range(episodes_count):
        deep_reset_func()
        step = 0
        s_train.clear()
        a_train.clear()
        r_train.clear()
        g_train.clear()

        s, a = act()

        while not deep_is_terminal_func() and step < max_steps_per_episode:
            r, t = deep_step_func(a)
            s_train.append(s)
            a_train.append(a)
            r_train.append(r)

            if t:
                break

            s, a = act()

            step += 1

        g = 0
        for r in r_train[::-1]:
            g = gamma * g + r
            g_train.insert(0, g)

        rolling_avg.append(g * 0.05 + (1 - 0.05) * rolling_avg[-1])

        if episode_id % 1000 == 0:
            plt.plot(rolling_avg)
            plt.show()

        g_train_vec = np.array(g_train)
        s_train_vec = np.array(s_train)
        a_train_vec = np.array(a_train)

        with tf.GradientTape(persistent=True) as tape:
            values_predicted = tf.squeeze(base_line_model(s_train_vec))
            predicted = pi_model(s_train_vec)
            expanded_targets = np.zeros_like(predicted)
            expanded_targets[np.arange(expanded_targets.shape[0]), a_train_vec] = -(g_train_vec - values_predicted)
            pi_losses = expanded_targets * tf.math.log(predicted + 1e-10)
            sum_losses = tf.math.reduce_sum(pi_losses) + pi_model.losses
            baseline_loss = tf.math.reduce_sum(
                keras.metrics.mean_squared_error(g_train_vec, values_predicted)) + base_line_model.losses

        pi_grads = tape.gradient(sum_losses, pi_model.trainable_variables)
        baseline_grads = tape.gradient(baseline_loss, base_line_model.trainable_variables)
        policy_optimizer.apply_gradients(zip(pi_grads, pi_model.trainable_variables))
        baseline_optimizer.apply_gradients(zip(baseline_grads, base_line_model.trainable_variables))
