# Objective:
This project focuses on the implementation of fundamental reinforcement learning techniques, including policy evaluation, policy improvement, policy iteration, and value iteration, within the context of the Frozen Lake environment (FrozenLake-v1) provided by OpenAI's Gym. We explore these algorithms to discover optimal navigation strategies for the Frozen Lake under two distinct scenarios.
1) The player may not always move in the intended direction due to the slippery nature of the frozen lake - Stochastic POlicy
2) The player moves under a deterministic policy with slippery nature of the environment set to False.

More details about the project environement can be found here: [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)


The file mdp_dp_test.py serves as a test suite for assessing the algorithms. To run the evaluation, execute the command nosetests -v mdp_dp_test.py in the terminal. Make sure that both mdp_dp.py and mdp_dp_test.py are located in the same directory.
