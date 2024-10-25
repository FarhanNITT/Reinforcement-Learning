#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


# implementing the basic DQN to set things up 

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.


    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # first we implement the paper architecture :

        # input dimension is 4 x 84 x 84
        # we can additinally add batch normalization and dropout layers in FC for better training

        self.conv1 = nn.Conv2d(in_channels,32,kernel_size=8,stride=4)  # 32x20x20
        self.bn1 = nn.BatchNorm2d(32)  # additional 
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)  # 64x9x9
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)  # 64x7x7
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # fully connected layers
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512,num_actions)  



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)


        return x
