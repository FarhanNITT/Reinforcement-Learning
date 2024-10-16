#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    current_sum = observation[0]
    if current_sum < 20:
        action = 1
    else:
        action = 0

    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for episode in range(n_episodes):
        # Generate an episode following the policy
        episode = []
        current_state, _ = env.reset()
        terminal = False

        while not terminal:
            # Sample action based on the policy
            current_action = policy(current_state)
            
            # Take action and observe result
            observation, reward, terminal, _, _ = env.step(current_action)
            
            # Store the transition
            episode.append((current_state, reward))
            
            # Move to the next state
            current_state = observation

        # Now we have the episode, let's calculate the returns for each state
        G = 0  # The return (cumulative discounted reward)
        visited_states = set()  # To ensure first-visit MC

        # Work backwards through the episode
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward  # Accumulate return

            # If it's the first visit to the state in this episode
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]  # Update the value function


    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    greedy_action = np.argmax(Q[state])
    random_action = int(np.random.uniform(0,nA))

    action = random.choices([greedy_action,random_action], weights=[1-epsilon,epsilon])[0]
    print(greedy_action,action)
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for episode in range(n_episodes):
        # Generate an episode following the policy
        episode = []
        current_state, _ = env.reset()
        nA = env.action_space.n
        terminal = False

        while not terminal:
            # Sample action based on the policy
            current_action = epsilon_greedy(Q,current_state,nA,epsilon)
            
            # Take action and observe result
            observation, reward, terminal, _, _ = env.step(current_action)
            
            # Store the transition
            episode.append((current_state, current_action, reward))
            
            # Move to the next state
            current_state = observation

        epsilon = max(0.01, epsilon - 0.1 / n_episodes)

        
         # Now we have the episode, let's calculate the returns for each state
        G = 0  # The return (cumulative discounted reward)
        visited_state_action = set()  # To ensure first-visit MC

        # Work backwards through the episode
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward  # Accumulate return
        
         # If it's the first visit to the state in this episode
            if (state,action) not in visited_state_action:
                visited_state_action.add((state,action))
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]  # Update the value function


cd 
    return Q
