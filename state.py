import numpy as np

from typing import Tuple, Optional
from agent import Agent
from player import Player
from tqdm import tqdm


class State:
    """
    Define the state of the environment.
    """

    def __init__(self, p1: Player, p2: Player, board_rows: int = 3, board_cols: int = 3) -> None:
        """
        Initialize the game state.
        :param p1: Player. Player 1
        :param p2: Player. Player 2
        :param board_rows: int. Number of rows in the game board.
        :param board_cols: int. Number of columns in the game board
        """
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board = np.zeros((board_rows, board_cols))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.board_hash = None
        self.player_id = 1

    @staticmethod
    def get_hash(board):
        """
        Get the string representation of the board to be used as the key of the states values.
        """
        return str(board.flatten())

    def is_end_state(self) -> Optional[int]:
        """
        check if the game satisfy the end condition.
        :return: 1 if p1 wins, -1 if p2 wins, 0 if draw, None if not ended
        """
        # check if row is all same
        for row in self.board:
            if np.sum(row) == self.board_rows:
                self.isEnd = True
                return 1
            if np.sum(row) == -self.board_rows:
                self.isEnd = True
                return -1

        # check if column is all same
        for col in self.board.T:
            if np.sum(col) == self.board_cols:
                self.isEnd = True
                return 1
            if np.sum(col) == -self.board_cols:
                self.isEnd = True
                return -1

        # diagonals
        diag1_sum = np.sum(np.diagonal(self.board))  # top left to bottom right
        diag2_sum = np.sum(np.diagonal(np.rot90(self.board)))  # top right to bottom left
        if diag1_sum == max(self.board_cols, self.board_rows) or diag2_sum == max(self.board_cols, self.board_rows):
            self.isEnd = True
            return 1
        if diag1_sum == -max(self.board_cols, self.board_rows) or diag2_sum == -max(self.board_cols, self.board_rows):
            self.isEnd = True
            return -1

        # draw
        if self.available_positions().shape[0] == 0:
            self.isEnd = True
            return 0

        # continue
        self.isEnd = False
        return None

    def available_positions(self) -> np.array:
        """
        get the indices of position on the board that is still empty.

        :return : list of list. [[row, col], ...]
        """
        return np.argwhere(self.board == 0)

    def update_state(self, position: Tuple) -> None:
        """
        Update the state of the board.

        player 1 uses 1, player 2 uses -1
        :param position: tuple. (row, col). The position to place the move.
        """
        # assign player id to the chosen position
        self.board[position] = self.player_id
        # pass the next turn to the other player
        self.player_id = self.player_id * -1

    def give_reward(self) -> None:
        """
        Give the reward to the player depending on the game end state
        """
        result = self.is_end_state()

        # player 1 wins
        if result == 1:
            self.p1.update_reward(1)
            self.p2.update_reward(0)
        # player 2 wins
        elif result == -1:
            self.p1.update_reward(0)
            self.p2.update_reward(1)
        # draw, player 1 started first so player 1 gets less reward
        else:
            self.p1.update_reward(0.1)
            self.p2.update_reward(0.5)

    def reset_game(self) -> None:
        """
        Reset the game to the initial state.
        """
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.board_hash = None
        self.isEnd = False
        self.player_id = 1

    def play(self, train=False, games=1) -> None:
        """
        start the game

        :param train: bool. if True, train the agent
        :param games: int. Number of rounds to train
        """
        for _ in tqdm(range(games), disable=not train):

            while not self.isEnd:

                if not train:
                    self.print_board()

                # get current player
                player = self.p1 if self.player_id == 1 else self.p2

                positions = self.available_positions()
                action = player.choose_action(positions, self.board, self.player_id)

                # update the state of the board
                self.update_state(action)

                if not train:
                    print(f'Player {player.name} takes {action}.')

                if train and isinstance(player, Agent):
                    board_hash = self.get_hash(self.board)
                    player.add_state(board_hash)

                # check board status if it is end
                winner = self.is_end_state()

                if winner is not None:
                    if not train:
                        self.print_board()
                        if winner == 1:
                            print(f'Player {self.p1.name} wins!')
                        elif winner == -1:
                            print(f'Player {self.p2.name} wins!')
                        else:
                            print(f'The game ends with a draw!')

                    # update reward and proceed to next game
                    if train:
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                    self.reset_game()
                    break

    def print_board(self):
        """
        Display the playing board on stdout.

        p1 will be X, p2 will be O
        """
        for i in range(0, self.board_rows):
            print('－－－－－－－－')
            row = '| '
            for j in range(0, self.board_cols):
                token = ' '
                if self.board[i, j] == 1:
                    token = 'X'
                if self.board[i, j] == -1:
                    token = 'O'
                row += token + ' | '
            print(row)
        print('－－－－－－－－')
