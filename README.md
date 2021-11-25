# tic-tac-toe-q-learning

This repository implements Q-learning for the tic-tac-toe in python.

Author: Thomas Wong

### Description

The update rule of Q-Learning is given by:

![formula](https://render.githubusercontent.com/render/math?math=\huge%20Q(s,a)\leftarrow%20Q(s,a)%20%2B%20\alpha%20[(r(s,a)%20%2B%20\gamma\underset{{a}'}{max}%20Q({s}',{a}'))%20-%20Q(s,a)])

Where ![formula](https://render.githubusercontent.com/render/math?math=Q(s,a)) is the Q-value of the state-action pair, 
![formula](https://render.githubusercontent.com/render/math?math=r(s,a)) is the reward of the state-action pair, 
![formula](https://render.githubusercontent.com/render/math?math=\alpha) is the learning rate, 
![formula](https://render.githubusercontent.com/render/math?math=\gamma) gamma is the discount factor, and 
![formula](https://render.githubusercontent.com/render/math?math=\underset{{a}'}{max}%20Q({s}',{a}')) is the maximum Q-value of the next state.

The parameters of the Q-learning algorithm are:
- the learning rate alpha (default: 0.2)
- the discount factor gamma (default: 0.9)
- the epsilon-greedy policy (default: 0.3)

They can be set in the agent class or when the agent is being initalized. But the default values are highly recommended as they were the best settings came up after doing several experiments.

### Prerequisites
Python 3.7 was used for this project but any later version should work.

Install the required packages by running:
```bash
pip install -r requirements.txt
```

### Play the game
To play the game, simply run `main.py` by:
```bash
python main.py
```

The program will ask if you want to go first or after, you may choose by typing the letter `y` or `n`.

To move a move, simply input the row number and the column number (0 for the first row or column, 1 for the second row or column, and so on).

### Retrain the agents
To retrain the agents, uncomment following line in the `main.py` file:
```python
# init_training(50000, resume=False)
```
The above line will train 2 agents for 50,000 games where they play against each other.

50,000 seems to be enough to get the agents to converge and they are unbeatable.