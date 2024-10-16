# Objective:
In this project, we implemented two model-free reinforcement learning algorithms. The first algorithm is Monte Carlo (MC), which includes both the first-visit on-policy MC prediction and on-policy MC control for the game of [blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack/). The second algorithm is Temporal-Difference (TD), featuring Sarsa (on-policy) and Q-Learning (off-policy) applied to the [cliff walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) problem.

The file mdp_dp_test.py serves as a test suite for assessing the algorithms. To run the evaluation, execute the command nosetests -v mdp_dp_test.py in the terminal. Make sure that both mdp_dp.py and mdp_dp_test.py are located in the same directory.
