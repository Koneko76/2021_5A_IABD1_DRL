import numpy as np


class TicTacToeAgainstRandomEnv:
    def __init__(self):
        import tensorflow
        self.s = 0
        self.to_categorical = tensorflow.keras.utils.to_categorical
        self.board = np.zeros((3, 3), dtype=np.float)
        self.next_player = None
        self.game_over = None
        self.available_actions = None

    def deep_reset_func(self):
        self.board[:, :] = 0
        self.next_player = np.random.randint(0, 2)
        self.game_over = False
        self.available_actions = list(range(9))

        if self.next_player == 1:
            a = np.random.choice(self.available_actions)
            self.available_actions.remove(a)
            self.board[a // 3, a % 3] = 2
            self.next_player = 0

    def deep_get_state(self) -> np.ndarray:
        return self.to_categorical(self.board, 3).flatten()

    def deep_get_available_actions(self) -> np.ndarray:
        return np.array(self.available_actions)

    def deep_get_all_available_action_description(self) -> np.ndarray:
        return self.to_categorical(np.array(self.available_actions), 9)

    def deep_is_terminal_func(self) -> bool:
        return self.game_over

    def deep_step_func(self, a: int) -> (float, bool):
        a = self.available_actions[a]

        a_i, a_j = a // 3, a % 3
        assert (self.board[a_i, a_j] == 0)

        self.board[a_i, a_j] = 1
        self.available_actions.remove(a)

        if self.board[a_i, 0] == self.board[a_i, 1] == self.board[a_i, 2] or \
                self.board[0, a_j] == self.board[1, a_j] == self.board[2, a_j] or \
                self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == 1 or \
                self.board[2, 0] == self.board[1, 1] == self.board[0, 2] == 1:
            self.game_over = True
            return 1.0, True

        if len(self.available_actions) == 0:
            self.game_over = True
            return 0.0, True

        a = np.random.choice(self.available_actions)

        a_i, a_j = a // 3, a % 3
        assert (self.board[a_i, a_j] == 0)

        self.board[a_i, a_j] = 2
        self.available_actions.remove(a)

        if self.board[a_i, 0] == self.board[a_i, 1] == self.board[a_i, 2] or \
                self.board[0, a_j] == self.board[1, a_j] == self.board[2, a_j] or \
                self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == 2 or \
                self.board[2, 0] == self.board[1, 1] == self.board[0, 2] == 2:
            self.game_over = True
            return -1.0, True

        if len(self.available_actions) == 0:
            self.game_over = True
            return 0.0, True

        return 0.0, False
