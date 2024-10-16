#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
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
        You can use the function from project2-1

    
    """
    greedy_action = np.argmax(Q[state])
    random_action = int(np.random.uniform(0,nA))

    action = random.choices([greedy_action,random_action], weights=[1-epsilon,epsilon])[0]

    return action
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for episode in range(n_episodes):
        current_state = env.reset()[0]

        terminal = False
        nA = env.action_space.n
        current_action = epsilon_greedy(Q,current_state,nA,epsilon)

        while not terminal:
            observation, reward, terminal, _, _ = env.step(current_action)

            next_action = epsilon_greedy(Q,observation,nA,epsilon)

            Q[current_state][current_action] += alpha * (reward + gamma*Q[observation][next_action] - Q[current_state][current_action])

            current_state = observation
            current_action = next_action

            

        epsilon = 0.99 * epsilon


    
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)

    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for episode in range(n_episodes):
        current_state = env.reset()[0]

        terminal = 0
        nA = env.action_space.n
        

        while terminal==0 :
            current_action = epsilon_greedy(Q,current_state,nA,epsilon)
            observation, reward, terminal, _, _ = env.step(current_action)

            # next_action = epsilon_greedy(Q,observation,nA,epsilon)  

            Q[current_state][current_action] += alpha * (reward + gamma*np.max(Q[observation]) - Q[current_state][current_action])

            current_state = observation

            
        
        # epsilon = 0.99 * epsilon

    return Q
