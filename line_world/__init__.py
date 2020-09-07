from __future__ import annotations
from typing import Callable

import numpy as np

num_states = 7
S = np.arange(num_states)
A = np.array([0, 1])  # 0: left, 1 : right
T = np.array([0, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

for s in S[1:-1]:
    P[s, 0, s - 1, 0] = 1.0
    P[s, 1, s + 1, 0] = 1.0
P[1, 0, 0, 1] = -1.0
P[num_states - 2, 1, num_states - 1, 1] = 1.0


def reset() -> int:
    return num_states // 2


def is_terminal(state: int) -> bool:
    return state in T


def step(state: int, a: int) -> (int, float, bool):
    assert (state not in T)
    s_p = np.random.choice(S, p=P[state, a, :, 0])
    r = P[state, a, s_p, 1]
    return s_p, r, (s_p in T)


class LineWorldEnv:
    def __init__(self):
        import tensorflow
        self.s = 0
        self.to_categorical = tensorflow.keras.utils.to_categorical

    def deep_reset(self):
        self.s = reset()

    def deep_get_state(self) -> np.ndarray:
        return self.to_categorical(self.s, num_states)

    def deep_get_available_actions(self) -> np.ndarray:
        return np.array([0, 1])

    def deep_get_all_action_description(self) -> np.ndarray:
        actions = np.array([0, 1])
        return self.to_categorical(actions, 2)

    def deep_is_terminal(self) -> bool:
        return is_terminal(self.s)

    def deep_step(self, a: int) -> (float, bool):
        self.s, r, t = step(self.s, a)
        return r, t
