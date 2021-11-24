import pickle
import os
import numpy as np

from collections import defaultdict
from typing import Tuple
from player import Player


POLICY_DIR = './policy/'


def dd():
    """
    required for pickling default dict
    """
    return 0


class Agent(Player):
    """
    Define the agent class.
    """

    def __init__(self, name: str, epsilon: float = 0.1, learning_rate: float = 0.2, decay_gamma: float = 0.95,
                 board_rows: int = 3, board_cols: int = 3) -> None:
        """
        Initialize the agent.

        :param name: str. Name of the agent.
        :param epsilon: float. Probability of taking a random action.
        :param learning_rate: float. Learning rate.
        :param decay_gamma: float. Discount factor.
        :param board_rows: int. Number of rows in the board.
        :param board_cols: int. Number of columns in the board.
        """
        super().__init__(name)
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.name = name
        self.states = []
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_gamma = decay_gamma
        self.states_value = defaultdict(dd)

    def add_state(self, state) -> None:
        """
        add state taken by the agent to the states list.

        :param state: str. The hash of the board state
        """
        self.states.append(state)

    def choose_action(self, available_positions, board, player_id) -> Tuple:
        """
        Choose an action based on the current state. epsilon-greedy policy.

        :param available_positions: np.array. Available positions of the current board.
        :param board: np.array. The current game board
        :param player_id: int. The player id.
        :return action: Tuple. The action in the form of (row, column)
        """

        # exploration
        if np.random.uniform(0, 1) <= self.epsilon:
            action = available_positions[np.random.choice(len(available_positions))]
        # exploitation
        else:
            # choose the action with the highest value
            action = self._choose_action_with_value(available_positions, board, player_id)

        return tuple(action)

    def _choose_action_with_value(self, available_positions, board, player_id) -> Tuple:
        """
        Choose an action based on the current state. Select the best action base on the utility.
        """
        value_max = 0
        action = available_positions[0]

        for p in available_positions:
            next_board = board.copy()
            next_board[p] = player_id
            next_board_hash = self.get_hash(next_board)
            value = self.states_value.get(next_board_hash, 0)

            if value >= value_max:
                value_max = value
                action = p

        return tuple(action)

    def update_reward(self, reward) -> None:
        """
        Update the reward of the states.
        V(S) = V(S) + alpha * (V(S') * gamma - V(S)) where S' is the next state.

        :param reward: float. The reward of the game.
        """
        for state in reversed(self.states):
            self.states_value[state] += self.learning_rate * (reward * self.decay_gamma - self.states_value[state])
            reward = self.states_value[state]

    def reset(self) -> None:
        """
        Reset the states list.
        """
        self.states = []

    def save_policy(self) -> None:
        """
        Save the policy of the agent.
        """
        if not os.path.exists(POLICY_DIR):
            os.mkdir(POLICY_DIR)

        save_path = os.path.join(POLICY_DIR, f'{self.name}_policy')
        with open(save_path, 'wb') as f:
            pickle.dump(self.states_value, f)

        print(f'Saved policy of {self.name} to {save_path}.')

    def load_policy(self, file) -> None:
        """
        Load the policy of the agent.
        :param file: str. Path of the policy file.
        """
        load_path = os.path.join(POLICY_DIR, file)
        with open(load_path, 'rb') as f:
            self.states_value = pickle.load(f)

        print(f'Loaded policy of {self.name} from {load_path}.')

    @staticmethod
    def get_hash(board):
        return str(board.flatten())
