from __future__ import annotations
import numpy as np

width = 4
height = 4
num_states = width * height
S = np.arange(num_states)
A = np.arange(4)  # 0: left, 1: Right, 2: Up, 3: Down
T = np.array([width - 1, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

for s in S:
    if (s % width) == 0:
        P[s, 0, s, 0] = 1.0
    else:
        P[s, 0, s - 1, 0] = 1.0
    if (s + 1) % width == 0:
        P[s, 1, s, 0] = 1.0
    else:
        P[s, 1, s + 1, 0] = 1.0
    if s < width:
        P[s, 2, s, 0] = 1.0
    else:
        P[s, 2, s - width, 0] = 1.0
    if s >= (num_states - width):
        P[s, 3, s, 0] = 1.0
    else:
        P[s, 3, s + width, 0] = 1.0

P[width - 1, :, :, 0] = 0.0
P[num_states - 1, :, :, 0] = 0.0

P[:, :, width - 1, 1] = -5.0
P[:, :, num_states - 1, 1] = 1.0


def reset() -> int:
    return 0


def is_terminal(state: int) -> bool:
    return state in T


def step(state: int, a: int) -> (int, float, bool):
    assert (state not in T)
    s_p = np.random.choice(S, p=P[state, a, :, 0])
    r = P[state, a, s_p, 1]
    return s_p, r, (s_p in T)


class GridWorldEnv:
    def __init__(self):
        import tensorflow
        self.s = 0
        self.to_categorical = tensorflow.keras.utils.to_categorical

    def deep_reset(self):
        self.s = reset()

    def deep_get_state(self) -> np.ndarray:
        return self.to_categorical(self.s, num_states)

    def deep_get_all_action_description(self) -> np.ndarray:
        actions = A
        return self.to_categorical(actions, len(A))

    def deep_get_available_actions(self) -> np.ndarray:
        return np.array([0, 1, 2, 3])

    def deep_is_terminal(self) -> bool:
        return is_terminal(self.s)

    def deep_step(self, a: int) -> (float, bool):
        self.s, r, t = step(self.s, a)
        return r, t
