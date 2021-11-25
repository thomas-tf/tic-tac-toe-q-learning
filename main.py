from state import State
from agent import Agent
from humanplayer import HumanPlayer


def init_training(games: int = 50000, resume: bool = False) -> None:
    """
    Initialize the training of two agents.

    :param games: int. The amount of games to play for training
    """

    agent1 = Agent("Agent1")
    agent2 = Agent("Agent2")

    if resume:
        agent1.load_policy('Agent1_policy')
        agent2.load_policy('Agent2_policy')

    state = State(agent1, agent2)

    print('Training started...')
    state.play(train=True, games=games)
    print('Training completed.')

    # save policy
    agent1.save_policy()
    agent2.save_policy()


def play(start_first: bool, player_name: str = 'human', agent_name: str = 'AI') -> None:
    """
    Play a game of Tic-Tac-Toe with the given player and agent.

    :param start_first: bool. Whether the human player is starting first
    :param player_name: str. The name of the human player
    :param agent_name: str. The name of the agent
    """

    human = HumanPlayer(player_name)
    agent = Agent(agent_name, epsilon=0)

    if start_first:
        policy_file = 'Agent2_policy'
    else:
        policy_file = 'Agent1_policy'
    agent.load_policy(policy_file)

    state = State(human, agent) if start_first else State(agent, human)

    state.play()


if __name__ == "__main__":

    # uncomment this line to retrain policy
    # init_training(100000, resume=False)

    while True:
        start_player = input('Would you like to go first? [y/n]')
        if start_player.lower() == 'y':
            start_first = True
            break
        elif start_player.lower() == 'n':
            start_first = False
            break

    play(start_first, player_name='Human', agent_name='AI')
