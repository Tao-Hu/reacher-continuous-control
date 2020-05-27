import numpy as np
import random

from model import ActorCriticNetwork, PolicyNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
EPSILON = 0.1           # Clip hyperparameter for PPO
ROLLOUT_LEN = 1000      # Rollout length if use GAE
LAMDA = 0.95            # lambda-return for GAE
BETA = 0.01             # Coefficient for entropy loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    def __init__(self, state_size, action_size, seed, 
                 hidden_layers, use_gae = True):
        """Initialize an PPO Agent object.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
           use_gae (logic): Indicator of using GAE, True or False
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_gae = use_gae
        self.seed = random.seed(seed)

        # networks
        if use_gae:
            self.network = ActorCriticNetwork(state_size, action_size, seed, hidden_layers).to(device)
        else:
            self.network = PolicyNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = LR)