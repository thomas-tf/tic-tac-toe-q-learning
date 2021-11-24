from player import Player


class HumanPlayer(Player):
    """
    A human player class.
    """

    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, available_positions, board, player_id):
        """
        Ask the user to choose an action.

        :param available_positions: List. list of possible positions
        :param board: np.array. The board where the action will be performed
        :param player_id: int. The id of the player
        :return action: Tuple. The action chosen by the user in the form of (row, column)
        """

        while True:

            try:
                row = int(input("Input your action row:"))
            except ValueError:
                print("Please input a valid row")
                continue

            try:
                col = int(input("Input your action col:"))
            except ValueError:
                print("Please input a valid col")
                continue

            if row > board.shape[0]:
                print("Row index is out of bounds, please retry.")
                continue

            if col > board.shape[1]:
                print("Row index is out of bounds, please retry.")
                continue

            action = (row, col)
            if action in available_positions:
                return action
            else:
                print(f"Invalid action. Position {action} is already taken.")
