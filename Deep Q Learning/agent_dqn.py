#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import count

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE 

        self.env = env      # first lets setup the environmen
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mode of operation 

    
         # DQN Hyperparameters

        # to pass as input to model to predict Q valuess
        self.in_channels = 4  # Input channels for stacked frames (e.g. Atari Breakout)
        self.num_actions = env.action_space.n  # Number of actions the agent can take


        # HYP to tweek with to get best performance - starting with standard values

        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.eps_start = 1.0  # Initial epsilon for epsilon-greedy
        self.eps_end = 0.1  # Final epsilon
        self.eps_decay = 1e-10  # Epsilon decay rate
        self.epsilon = self.eps_start 
        self.lr = 0.001
        self.tau = 0.005

        # self.target_update = 10  # Target network update frequency (how frequently to change the target network weights)
        
        
        # Initialize networks

        self.policy_net = DQN(self.in_channels, self.num_actions).to(self.device)  # action selection
        self.target_net = DQN(self.in_channels, self.num_actions).to(self.device)   # calculating target Q values
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   # initializing both networks with same weights
        self.target_net.eval()   # set target network to evalaution mode as we dont update its weights


        # Replay buffer
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.memory = deque(maxlen=10000)   # Check this out for training

        # Adam Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        
        # Steps and episodes
        self.steps_done = 0    # counter for actions taken 
        # self.num_episodes = args.num_episodes    # we can either pass num_episodes as arg or pass directly as variable

        if args.test_dqn:
            # Load your model here if testing
            print('loading trained model')
            model_path = 'double_dqn_model.pth'
            self.policy_net.load_state_dict(torch.load(model_path))
            self.policy_net.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        

        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2).to(self.device)  # current state will be stack of 4 frames - numpy array 84x84x4 -- make tensor and add dimension for batch size
        
        
        # Epsilon-greedy action selection
        self.epsilon = self.eps_end + (self.epsilon - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()   # get the best action from greedy policy
        else: 
            return self.env.action_space.sample()

    
    

    def push(self,state,action,next_state,reward,done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        

        #self.memory.append((state, action, next_state, reward, done))  # check training function
        self.memory.append(self.Transition(state, action, next_state, reward))

        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        transitions = random.sample(self.memory, self.batch_size)
        #batch_state, batch_action, batch_next_state, batch_reward,batch_done = zip(*transitions)


        """ batch_state = torch.tensor(batch_state, dtype=torch.float32).to(self.device)"""
        # batch_action = torch.tensor(batch_action).unsqueeze(1).to(self.device)
        """ batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(self.device)
        batch_reward = torch.tensor(batch_reward).to(self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).to(self.device)
     """
        
        return transitions
        
        
    def optimize_model(self):

        if len(self.memory) < 9000:
            return
        
        transitions = self.replay_buffer()
        # Sample replay buffer
        #state_batch, action_batch, next_state_batch, reward_batch,done_batch= self.replay_buffer()

        batch = self.Transition(*zip(*transitions))

        """ state_batch = torch.tensor(state_batch).permute(0,3,1,2).to(self.device)
        
        state_batch = torch.cat(state_batch)
        #action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch) """

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)

        #next_state_batch = torch.tensor(next_state_batch)
        # Convert to tensor with the desired shape
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        #print(f"batch action value{batch.action}")

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #state_batch = state_batch.float()  
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch



        # Compute loss (Huber loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.num_episodes = 500
        else:
            self.num_episodes = 10000

        self.episode_durations = []
        for i_episode in range(self.num_episodes):
        
            state = self.env.reset()  # Initialize the environment and get its state
            #state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2)

            #print(state.shape)
            for t in count():
                #print(f"Episode number: {t}")
                action = self.make_action(state)
                #state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2)

                tensor_action = torch.tensor([action], dtype=torch.int64, device=self.device).unsqueeze(1)
                tensor_reward = torch.tensor([reward],dtype=torch.float32, device=self.device)
                tensor_state = torch.tensor(state,dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2)

                self.push(tensor_state, tensor_action, next_state, tensor_reward, done)
                print("exec")

                # Move to the next state
                state = observation

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # plot_durations()
                    break

                torch.save(policy_net_state_dict, 'double_dqn_model.pth')

        
        ###########################