import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, name: str, epsilon: float = 0.3, learning_rate: float = 0.2, decay_gamma: float = 0.9,
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
        self.num_loses = 0
        self.num_wins = 0
        self.num_draws = 0
        self.running_loses = []
        self.running_wins = []
        self.running_draws = []
        self.running_win_rate = []
        self.running_draw_rate = []
        self.running_lose_rate = []
        self.states_value = defaultdict(dd)

    @staticmethod
    def get_hash(board):
        """
        Get the string representation of the board to be used as the key of the states values.
        """
        return str(board.flatten())

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
        action = tuple(available_positions[0])

        for position in available_positions:
            position = tuple(position)
            next_board = board.copy()
            next_board[position] = player_id
            next_board_hash = self.get_hash(next_board)
            value = self.states_value.get(next_board_hash, 0)

            if value >= value_max:
                value_max = value
                action = position

        return action

    def update_reward(self, reward) -> None:
        """
        Update the reward of the states.
        V(S) = V(S) + alpha * (V(S') * gamma - V(S)) where S' is the next state.

        :param reward: float. The reward of the game.
        """
        for state in reversed(self.states):
            self.states_value[state] += self.learning_rate * (reward * self.decay_gamma - self.states_value[state])
            reward = self.states_value[state]

    def add_evaluation_stats(self, reward) -> None:
        """

        :param reward:
        :return:
        """
        if reward == 1:
            self.num_wins += 1
        elif reward == 0:
            self.num_loses += 1
        else:
            self.num_draws += 1

        self.running_wins.append(self.num_wins)
        self.running_loses.append(self.num_loses)
        self.running_draws.append(self.num_draws)
        self.running_win_rate.append(self.num_wins / (self.num_wins + self.num_loses + self.num_draws))
        self.running_draw_rate.append(self.num_draws / (self.num_wins + self.num_loses + self.num_draws))
        self.running_lose_rate.append(self.num_loses / (self.num_wins + self.num_loses + self.num_draws))

    def plot_statistics(self, games) -> None:
        """
        Plot the statistics of the agent.
        """
        num_games = len(self.running_win_rate)
        x = np.arange(num_games)
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(x, self.running_wins, label='Wins', color='blue')
        axs[0].plot(x, self.running_loses, label='Loses', color='red')
        axs[0].plot(x, self.running_draws, label='Draws', color='green')
        axs[0].legend()
        axs[0].set_ylabel('Number of wins, losses and draws')

        axs[1].plot(x, self.running_win_rate, label='Win rate', color='blue')
        axs[1].plot(x, self.running_draw_rate, label='Draw rate', color='green')
        axs[1].plot(x, self.running_lose_rate, label='Lose rate', color='red')
        axs[1].legend()
        axs[1].set_ylabel('Win rate')

        fig.suptitle(f'Evaluation at {games} games for agent {self.name}')
        plt.show()

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
